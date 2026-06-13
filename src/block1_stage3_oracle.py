# RQ1 stage 3: coreness_ascent + residual_outliers oracle checks (Table IV)
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import polars as pl
import igraph as ig

sys.path.insert(0, "/private/workspace/icde_flow/code")
from tracecore_mvp import TraceCoreMVP

PARQUET = "/private/workspace/icde_flow/data/lanl_2015/flows_minimal.parquet"

def igraph_shell_for_epoch(eid: int, epoch_sec: int) -> dict[str, int]:
    t0, t1 = eid * epoch_sec, (eid + 1) * epoch_sec
    df = (pl.scan_parquet(PARQUET)
            .filter((pl.col("time") >= t0) & (pl.col("time") < t1))
            .filter(pl.col("src_comp") != pl.col("dst_comp"))
            .select(["src_comp", "dst_comp"]).unique().collect())
    if df.height == 0:
        return {}
    nodes = sorted(set(df["src_comp"].to_list()) | set(df["dst_comp"].to_list()))
    node2id = {v: i for i, v in enumerate(nodes)}
    edges = {(min(node2id[u], node2id[v]), max(node2id[u], node2id[v]))
             for u, v in zip(df["src_comp"].to_list(), df["dst_comp"].to_list())}
    g = ig.Graph(n=len(nodes), edges=list(edges), directed=False)
    cor = g.coreness()
    return {nodes[i]: int(cor[i]) for i in range(len(nodes))}

def ascent_events(timeline: dict[int, int], e_first: int, e_last: int,
                  delta: int, p: int) -> set[tuple[int, int, int]]:
    out = set()
    for e in range(e_first + p, e_last + 1):
        c_now = timeline.get(e, 0)
        c_prev = timeline.get(e - p, 0)
        if c_now - c_prev >= delta:
            out.add((e, c_prev, c_now))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mvp", default="/private/workspace/icde_flow/bench/lanl_redteam_range/tracecore_mvp.pkl")
    p.add_argument("--rf", default="/private/workspace/icde_flow/bench/lanl_redteam_range/rolling_fanout.parquet")
    p.add_argument("--n-windows", type=int, default=20, help="1-hour windows for ascent check")
    p.add_argument("--n-hosts-per-window", type=int, default=5)
    p.add_argument("--n-epochs-outliers", type=int, default=50)
    p.add_argument("--z0-list", default="2,3,4")
    p.add_argument("--n-roles", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/private/workspace/icde_flow/results/block1_stage3.json")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    z0_list = [float(z) for z in args.z0_list.split(",")]

    print(f"[load] {args.mvp}")
    tc = TraceCoreMVP.load(args.mvp)
    eids = sorted(tc.shells.keys())
    e_min, e_max = eids[0], eids[-1]
    print(f"[load] {len(eids)} epochs ({e_min}..{e_max}), epoch_sec={tc.epoch_sec}")

    print(f"\n=== coreness_ascent oracle ({args.n_windows} windows x "
          f"{args.n_hosts_per_window} hosts x 2 (delta,p) configs) ===")
    win_len = 12
    ascent_checks = 0
    ascent_mismatches = 0
    window_starts = sorted(random.sample(range(e_min, e_max - win_len), args.n_windows))
    for wi, e0 in enumerate(window_starts):
        e1 = e0 + win_len

        oracle_shells = {e: igraph_shell_for_epoch(e, tc.epoch_sec) for e in range(e0, e1 + 1)}

        active = set()
        for e in range(e0, e1 + 1):
            active |= set(tc.shells.get(e, {}).keys())
        if not active:
            print(f"  W{wi:02d} epochs {e0}..{e1}: EMPTY, skip")
            continue
        hosts = random.sample(sorted(active), min(args.n_hosts_per_window, len(active)))
        w_mis = 0
        for h in hosts:
            idx_tl = {e: tc.shells.get(e, {}).get(h, 0) for e in range(e0, e1 + 1)}
            ora_tl = {e: oracle_shells[e].get(h, 0) for e in range(e0, e1 + 1)}
            for (delta, pp) in [(1, 1), (2, 3)]:
                ev_idx = ascent_events(idx_tl, e0, e1, delta, pp)
                ev_ora = ascent_events(ora_tl, e0, e1, delta, pp)
                ascent_checks += 1
                if ev_idx != ev_ora:
                    ascent_mismatches += 1
                    w_mis += 1
        print(f"  W{wi:02d} epochs {e0}..{e1}: hosts={len(hosts)} mismatches={w_mis}")
    print(f"[ascent] checks={ascent_checks}, mismatches={ascent_mismatches}")

    print(f"\n=== residual_outliers oracle ({args.n_epochs_outliers} epochs x z0 in {z0_list}) ===")

    rf = pl.read_parquet(args.rf)
    rows = []
    for eid, shell_dict in tc.shells.items():
        for host, shell in shell_dict.items():
            rows.append((host, eid, shell))
    shells_df = pl.DataFrame({"host": [r[0] for r in rows],
                              "epoch_id": [r[1] for r in rows],
                              "k_shell": [r[2] for r in rows]})
    full = shells_df.join(rf, on=["host", "epoch_id"], how="left").fill_null(0)
    host_feat = (full.group_by("host")
                     .agg([pl.col("k_shell").median().alias("med_kshell"),
                           pl.col("k_shell").max().alias("max_kshell"),
                           pl.col("fanout_now").median().alias("med_fanout_now"),
                           pl.col("rolling_fanout_12").median().alias("med_rf12"),
                           pl.col("rolling_fanout_72").median().alias("med_rf72"),
                           pl.col("rolling_fanout_288").median().alias("med_rf288"),
                           pl.col("rolling_fanout_2016").median().alias("med_rf2016"),
                           pl.col("epoch_id").count().alias("n_epochs_present")]))
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    feat_cols = ["med_kshell", "max_kshell", "med_fanout_now",
                 "med_rf12", "med_rf72", "med_rf288", "med_rf2016", "n_epochs_present"]
    X = host_feat.select(feat_cols).to_numpy().astype(np.float64)
    for j, col in enumerate(feat_cols):
        if col.startswith("med") or col == "n_epochs_present":
            X[:, j] = np.log1p(np.maximum(X[:, j], 0))
    Xs = StandardScaler().fit_transform(X)
    km = MiniBatchKMeans(n_clusters=args.n_roles, random_state=args.seed,
                          batch_size=1024, n_init=5).fit(Xs)
    host_role = pl.DataFrame({"host": host_feat["host"].to_list(),
                              "role": km.labels_.astype(np.int32).tolist()})
    full = full.join(host_role, on="host", how="left")

    SCORE = "rolling_fanout_72"
    SIGMA_MIN = 0.5

    cohort = (full.group_by(["role", "epoch_id"])
                  .agg([pl.col(SCORE).mean().alias("mu"),
                        pl.col(SCORE).std().alias("sigma")]))
    scored = (full.join(cohort, on=["role", "epoch_id"], how="left")
                  .with_columns(((pl.col(SCORE) - pl.col("mu")) /
                                 pl.col("sigma").fill_null(1.0).clip(SIGMA_MIN, None)).alias("z")))

    sample_epochs = sorted(random.sample(eids, min(args.n_epochs_outliers, len(eids))))
    outlier_checks = 0
    outlier_mismatches = 0
    for eid in sample_epochs:
        ep = scored.filter(pl.col("epoch_id") == eid)
        if ep.height == 0:
            continue

        hosts_e = ep["host"].to_list()
        roles_e = ep["role"].to_list()
        vals_e = ep[SCORE].to_list()
        by_role = defaultdict(list)
        for r, v in zip(roles_e, vals_e):
            by_role[r].append(float(v))
        mu_b, sd_b = {}, {}
        for r, vs in by_role.items():
            arr = np.array(vs, dtype=np.float64)
            mu_b[r] = float(arr.mean())

            sd_b[r] = float(arr.std(ddof=1)) if len(arr) > 1 else 1.0
        z_brute = {}
        for h, r, v in zip(hosts_e, roles_e, vals_e):
            z_brute[h] = (float(v) - mu_b[r]) / max(sd_b[r], SIGMA_MIN)
        z_idx = dict(zip(hosts_e, ep["z"].to_list()))
        for z0 in z0_list:
            s_idx = {h for h, z in z_idx.items() if z is not None and z >= z0}
            s_bru = {h for h, z in z_brute.items() if z >= z0}
            outlier_checks += 1
            if s_idx != s_bru:
                outlier_mismatches += 1
                diff = s_idx.symmetric_difference(s_bru)
                print(f"  MISMATCH eid={eid} z0={z0}: |idx|={len(s_idx)} |brute|={len(s_bru)} symdiff={len(diff)}")
    print(f"[outliers] checks={outlier_checks}, mismatches={outlier_mismatches}")

    summary = {
        "config": vars(args),
        "coreness_ascent": {"n_checks": ascent_checks, "mismatches": ascent_mismatches,
                            "n_windows": args.n_windows, "window_epochs": win_len,
                            "delta_p_configs": [[1, 1], [2, 3]]},
        "residual_outliers": {"n_checks": outlier_checks, "mismatches": outlier_mismatches,
                              "n_epochs": len(sample_epochs), "z0_list": z0_list,
                              "score": SCORE, "sigma_min": SIGMA_MIN},
        "overall_pass": ascent_mismatches == 0 and outlier_mismatches == 0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\n=== Stage 3 summary ===\n{json.dumps(summary['coreness_ascent'])}\n"
          f"{json.dumps(summary['residual_outliers'])}\noverall_pass={summary['overall_pass']}")
    print(f"[save] {args.out}")
    return 0 if summary["overall_pass"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
