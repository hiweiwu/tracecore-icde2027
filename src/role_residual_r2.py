# TraceCore-R: causal role protocols, frozen prefix / weekly refit (Table VIII)
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import polars as pl
import networkit as nk

sys.path.insert(0, "/private/workspace/icde_flow/code")
from tracecore_mvp import TraceCoreMVP
from role_residual_lanl import load_redteam, compute_recall_at_fp
from role_residual_r1 import (FEAT_COLS, COHORT_COLS, FP_TARGETS,
                              host_medians, featurize, score_instant, score_ewma,
                              evaluate)

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

PARQUET = "/private/workspace/icde_flow/data/lanl_2015/flows_minimal.parquet"

def fit_roles_sorted(host_feat: pl.DataFrame, n_roles: int, seed: int):
    host_feat = host_feat.sort("host")
    X = featurize(host_feat)
    scaler = StandardScaler().fit(X)
    km = MiniBatchKMeans(n_clusters=n_roles, random_state=seed,
                          batch_size=1024, n_init=5).fit(scaler.transform(X))
    return scaler, km

def assign_all(hosts: list[str], host_feat: pl.DataFrame, scaler, km) -> pl.DataFrame:
    host_feat = host_feat.sort("host")
    seen = set(host_feat["host"].to_list())
    unseen = sorted(h for h in hosts if h not in seen)
    if unseen:
        zeros = pl.DataFrame({"host": unseen,
                              **{c: [0.0] * len(unseen) for c in FEAT_COLS[:-1]},
                              "n_epochs_present": [0] * len(unseen)})
        zeros = zeros.with_columns([pl.col(c).cast(host_feat[c].dtype) for c in FEAT_COLS])
        all_feat = pl.concat([host_feat, zeros.select(host_feat.columns)]).sort("host")
    else:
        all_feat = host_feat
    labels = km.predict(scaler.transform(featurize(all_feat))).astype(np.int32)
    return pl.DataFrame({"host": all_feat["host"].to_list(), "role": labels.tolist()})

def build_prefix_shells(t1: int, epoch_sec: int) -> pl.DataFrame:
    rows_h, rows_e, rows_k = [], [], []
    n_ep = t1 // epoch_sec
    for eid in range(0, n_ep):
        a, b = eid * epoch_sec, (eid + 1) * epoch_sec
        df = (pl.scan_parquet(PARQUET)
                .filter((pl.col("time") >= a) & (pl.col("time") < b))
                .filter(pl.col("src_comp") != pl.col("dst_comp"))
                .select(["src_comp", "dst_comp"]).unique().collect())
        if df.height == 0:
            continue
        nodes = sorted(set(df["src_comp"].to_list()) | set(df["dst_comp"].to_list()))
        node2id = {v: i for i, v in enumerate(nodes)}
        edges = {(min(node2id[u], node2id[v]), max(node2id[u], node2id[v]))
                 for u, v in zip(df["src_comp"].to_list(), df["dst_comp"].to_list())}
        g = nk.Graph(len(nodes), directed=False)
        for x, y in edges:
            g.addEdge(x, y)
        cd = nk.centrality.CoreDecomposition(g); cd.run()
        sc = cd.scores()
        for i, h in enumerate(nodes):
            rows_h.append(h); rows_e.append(eid); rows_k.append(int(sc[i]))
    return pl.DataFrame({"host": rows_h, "epoch_id": rows_e, "k_shell": rows_k})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mvp", default="/private/workspace/icde_flow/bench/lanl_redteam_range/tracecore_mvp.pkl")
    p.add_argument("--rf", default="/private/workspace/icde_flow/bench/lanl_redteam_range/rolling_fanout.parquet")
    p.add_argument("--prefix-rf", default="/private/workspace/icde_flow/bench/lanl_prefix/rolling_fanout_prefix.parquet")
    p.add_argument("--redteam", default="/private/workspace/icde_flow/data/lanl_2015/redteam.txt.gz")
    p.add_argument("--prefix-t1", type=int, default=150_000)
    p.add_argument("--week-epochs", type=int, default=2016)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/private/workspace/icde_flow/results/role_residual_r2.json")
    args = p.parse_args()

    t_all = time.time()
    tc = TraceCoreMVP.load(args.mvp)
    rf = pl.read_parquet(args.rf)
    rt = load_redteam(args.redteam)
    pos_pairs = {(src, tc.epoch_of(t)) for t, src in rt}

    print("[flatten] campaign shells")
    rows_h, rows_e, rows_k = [], [], []
    for eid, shell_dict in tc.shells.items():
        for host, shell in shell_dict.items():
            rows_h.append(host); rows_e.append(eid); rows_k.append(shell)
    camp = (pl.DataFrame({"host": rows_h, "epoch_id": rows_e, "k_shell": rows_k})
              .join(rf, on=["host", "epoch_id"], how="left").fill_null(0))
    all_hosts = sorted(camp["host"].unique().to_list())
    print(f"[flatten] campaign {camp.height:,} rows, {len(all_hosts):,} hosts")

    print("[prefix] building pre-campaign shells (epochs 0..%d)" % (args.prefix_t1 // tc.epoch_sec - 1))
    t0 = time.time()
    pre_shells = build_prefix_shells(args.prefix_t1, tc.epoch_sec)
    print(f"[prefix] {pre_shells.height:,} shell rows in {time.time()-t0:.0f}s")
    pre_rf = pl.read_parquet(args.prefix_rf)
    prefix = pre_shells.join(pre_rf, on=["host", "epoch_id"], how="left").fill_null(0)
    print(f"[prefix] joined {prefix.height:,} rows, {prefix['host'].n_unique():,} hosts")

    report = {"config": vars(args),
              "prefix_epochs": args.prefix_t1 // tc.epoch_sec,
              "prefix_days": round(args.prefix_t1 / 86400, 2),
              "prefix_hosts": prefix["host"].n_unique(),
              "campaign_hosts": len(all_hosts)}

    print("\n[benign_prefix] roles from pre-campaign window only")
    hf_pre = host_medians(prefix)
    sc_a, km_a = fit_roles_sorted(hf_pre, 16, args.seed)
    roles_a = assign_all(all_hosts, hf_pre, sc_a, km_a)
    fr_a = camp.join(roles_a, on="host", how="left")
    report["benign_prefix_instant"] = evaluate(score_instant(fr_a), pos_pairs)
    for m, v in report["benign_prefix_instant"].items():
        print(f"    {m:<26s} AP={v['AP']:.4f} R@1e-3={v['R_at_FP_0.001']:.3f} R@1e-2={v['R_at_FP_0.01']:.3f}")
    report["benign_prefix_ewma_hl288"] = evaluate(score_ewma(fr_a, 288.0), pos_pairs,
                                                  methods=["rolling_fanout_72_z", "rolling_fanout_288_z"])
    v = report["benign_prefix_ewma_hl288"]["rolling_fanout_288_z"]
    print(f"    ewma_hl288 rf288_z: AP={v['AP']:.4f} R@1e-3={v['R_at_FP_0.001']:.3f} R@1e-2={v['R_at_FP_0.01']:.3f}")

    print("\n[weekly_refit] causal rolling refit at 1-week boundaries")
    eids = sorted(camp["epoch_id"].unique().to_list())
    e_min, e_max = eids[0], eids[-1]
    boundaries = list(range(e_min, e_max + 1, args.week_epochs))
    seg_frames = []
    hist = prefix
    roles_cur, sc_cur, km_cur = roles_a, sc_a, km_a
    for bi, b in enumerate(boundaries):
        seg_end = min(b + args.week_epochs, e_max + 1)
        seg = camp.filter((pl.col("epoch_id") >= b) & (pl.col("epoch_id") < seg_end))
        if seg.height == 0:
            continue
        if bi > 0:

            hist = pl.concat([prefix, camp.filter(pl.col("epoch_id") < b)
                                          .select(prefix.columns)])
            hf_hist = host_medians(hist)
            sc_cur, km_cur = fit_roles_sorted(hf_hist, 16, args.seed)
            roles_cur = assign_all(all_hosts, hf_hist, sc_cur, km_cur)
        seg_scored = score_instant(seg.join(roles_cur, on="host", how="left"))
        seg_frames.append(seg_scored)
        print(f"    week {bi+1}: epochs [{b}, {seg_end}) rows={seg.height:,}")
    common_cols = seg_frames[0].columns
    weekly = pl.concat([s.select(common_cols) for s in seg_frames])
    report["weekly_refit_instant"] = evaluate(weekly, pos_pairs)
    for m, v in report["weekly_refit_instant"].items():
        print(f"    {m:<26s} AP={v['AP']:.4f} R@1e-3={v['R_at_FP_0.001']:.3f} R@1e-2={v['R_at_FP_0.01']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[save] {args.out} (total {time.time()-t_all:.0f}s)")

if __name__ == "__main__":
    main()
