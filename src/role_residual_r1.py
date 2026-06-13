# TraceCore-R: K-sweep and EWMA sensitivity (supplement)
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import polars as pl

sys.path.insert(0, "/private/workspace/icde_flow/code")
from tracecore_mvp import TraceCoreMVP
from role_residual_lanl import load_redteam, compute_recall_at_fp

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

FEAT_COLS = ["med_kshell", "max_kshell", "med_fanout_now",
             "med_rf12", "med_rf72", "med_rf288", "med_rf2016", "n_epochs_present"]
COHORT_COLS = ["rolling_fanout_12", "rolling_fanout_72", "rolling_fanout_288", "k_shell"]
FP_TARGETS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
SIGMA_MIN = 0.5

def host_medians(full: pl.DataFrame) -> pl.DataFrame:
    return (full.group_by("host")
                .agg([pl.col("k_shell").median().alias("med_kshell"),
                      pl.col("k_shell").max().alias("max_kshell"),
                      pl.col("fanout_now").median().alias("med_fanout_now"),
                      pl.col("rolling_fanout_12").median().alias("med_rf12"),
                      pl.col("rolling_fanout_72").median().alias("med_rf72"),
                      pl.col("rolling_fanout_288").median().alias("med_rf288"),
                      pl.col("rolling_fanout_2016").median().alias("med_rf2016"),
                      pl.col("epoch_id").count().alias("n_epochs_present")]))

def featurize(host_feat: pl.DataFrame):
    X = host_feat.select(FEAT_COLS).to_numpy().astype(np.float64)
    for j, col in enumerate(FEAT_COLS):
        if col.startswith("med") or col == "n_epochs_present":
            X[:, j] = np.log1p(np.maximum(X[:, j], 0))
    return X

def fit_roles(host_feat: pl.DataFrame, n_roles: int, seed: int):
    X = featurize(host_feat)
    scaler = StandardScaler().fit(X)
    km = MiniBatchKMeans(n_clusters=n_roles, random_state=seed,
                          batch_size=1024, n_init=5).fit(scaler.transform(X))
    return scaler, km

def assign_roles(host_feat: pl.DataFrame, scaler, km) -> pl.DataFrame:
    labels = km.predict(scaler.transform(featurize(host_feat))).astype(np.int32)
    return pl.DataFrame({"host": host_feat["host"].to_list(), "role": labels.tolist()})

def score_instant(full_roles: pl.DataFrame) -> pl.DataFrame:
    cohort = (full_roles.group_by(["role", "epoch_id"])
                        .agg([pl.col(c).mean().alias(f"mu_{c}") for c in COHORT_COLS] +
                             [pl.col(c).std().alias(f"sigma_{c}") for c in COHORT_COLS]))
    scored = full_roles.join(cohort, on=["role", "epoch_id"], how="left")
    for c in COHORT_COLS:
        scored = scored.with_columns(
            ((pl.col(c) - pl.col(f"mu_{c}")) /
             pl.col(f"sigma_{c}").fill_null(1.0).clip(SIGMA_MIN, None)).alias(f"{c}_z"))
    return scored

def score_ewma(full_roles: pl.DataFrame, half_life_epochs: float) -> pl.DataFrame:
    alpha = 1.0 - 0.5 ** (1.0 / half_life_epochs)
    inst = (full_roles.group_by(["role", "epoch_id"])
                      .agg([pl.col(c).mean().alias(f"m1_{c}") for c in COHORT_COLS] +
                           [(pl.col(c) ** 2).mean().alias(f"m2_{c}") for c in COHORT_COLS])
                      .sort(["epoch_id"]))
    roles = sorted([r for r in full_roles["role"].unique().to_list() if r is not None])
    epochs = sorted(inst["epoch_id"].unique().to_list())

    state = {r: [None, None] for r in roles}
    rows = []
    inst_by_epoch = inst.partition_by("epoch_id", as_dict=True)
    for e in epochs:

        for r in roles:
            m1, m2 = state[r]
            if m1 is None:
                continue
            row = {"role": r, "epoch_id": e}
            for i, c in enumerate(COHORT_COLS):
                mu = m1[i]
                var = max(m2[i] - m1[i] ** 2, 0.0)
                row[f"mu_{c}"] = mu
                row[f"sigma_{c}"] = var ** 0.5
            rows.append(row)

        ep_df = inst_by_epoch.get((e,))
        if ep_df is None:
            ep_df = inst_by_epoch.get(e)
        if ep_df is None:
            continue
        for rec in ep_df.iter_rows(named=True):
            r = rec["role"]
            if r is None:
                continue
            m1v = np.array([rec[f"m1_{c}"] or 0.0 for c in COHORT_COLS])
            m2v = np.array([rec[f"m2_{c}"] or 0.0 for c in COHORT_COLS])
            if state[r][0] is None:
                state[r] = [m1v, m2v]
            else:
                state[r][0] = (1 - alpha) * state[r][0] + alpha * m1v
                state[r][1] = (1 - alpha) * state[r][1] + alpha * m2v
    if not rows:
        raise RuntimeError("EWMA produced no state rows")
    stats = pl.DataFrame(rows)
    scored = full_roles.join(stats, on=["role", "epoch_id"], how="left")
    for c in COHORT_COLS:
        scored = scored.with_columns(
            ((pl.col(c) - pl.col(f"mu_{c}").fill_null(0.0)) /
             pl.col(f"sigma_{c}").fill_null(1.0).clip(SIGMA_MIN, None)).alias(f"{c}_z"))
    return scored

def evaluate(scored: pl.DataFrame, pos_pairs: set, methods=None) -> dict:
    scored = scored.with_columns(
        pl.struct(["host", "epoch_id"]).map_elements(
            lambda s: (s["host"], s["epoch_id"]) in pos_pairs, return_dtype=pl.Boolean
        ).alias("is_positive"))
    y_true = scored["is_positive"].to_numpy().astype(np.int64)
    out = {}
    if methods is None:
        methods = [f"{c}_z" for c in COHORT_COLS]
    for m in methods:
        y = np.nan_to_num(scored[m].fill_null(0.0).to_numpy().astype(np.float64),
                          nan=0.0, posinf=1e9, neginf=-1e9)
        out[m] = compute_recall_at_fp(y_true, y, FP_TARGETS)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mvp", default="/private/workspace/icde_flow/bench/lanl_redteam_range/tracecore_mvp.pkl")
    p.add_argument("--rf", default="/private/workspace/icde_flow/bench/lanl_redteam_range/rolling_fanout.parquet")
    p.add_argument("--redteam", default="/private/workspace/icde_flow/data/lanl_2015/redteam.txt.gz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-list", default="4,8,12,16,24,32,64")
    p.add_argument("--ewma-half-lives", default="72,288,1152")
    p.add_argument("--out", default="/private/workspace/icde_flow/results/role_residual_r1.json")
    args = p.parse_args()

    t_all = time.time()
    print(f"[load] {args.mvp}")
    tc = TraceCoreMVP.load(args.mvp)
    rf = pl.read_parquet(args.rf)
    rt = load_redteam(args.redteam)
    pos_pairs = {(src, tc.epoch_of(t)) for t, src in rt}
    first_attack_epoch = min(tc.epoch_of(t) for t, _ in rt)
    eids = sorted(tc.shells.keys())
    print(f"[load] epochs {eids[0]}..{eids[-1]}; first_attack_epoch={first_attack_epoch} "
          f"({(first_attack_epoch - eids[0])} epochs of benign prefix = "
          f"{(first_attack_epoch - eids[0]) * tc.epoch_sec / 86400:.2f} days)")

    print("[flatten] shells -> long df")
    rows_h, rows_e, rows_k = [], [], []
    for eid, shell_dict in tc.shells.items():
        for host, shell in shell_dict.items():
            rows_h.append(host); rows_e.append(eid); rows_k.append(shell)
    shells_df = pl.DataFrame({"host": rows_h, "epoch_id": rows_e, "k_shell": rows_k})
    full = shells_df.join(rf, on=["host", "epoch_id"], how="left").fill_null(0)
    print(f"[flatten] {full.height:,} rows")

    report = {"config": vars(args), "first_attack_epoch": first_attack_epoch,
              "benign_prefix_epochs": first_attack_epoch - eids[0],
              "benign_prefix_days": round((first_attack_epoch - eids[0]) * tc.epoch_sec / 86400, 2)}

    print("\n[base] K=16 instantaneous cohort stats")
    hf_full = host_medians(full)
    scaler16, km16 = fit_roles(hf_full, 16, args.seed)
    roles16 = assign_roles(hf_full, scaler16, km16)
    fr = full.join(roles16, on="host", how="left")
    report["base_k16_instant"] = evaluate(score_instant(fr), pos_pairs)
    for m, v in report["base_k16_instant"].items():
        print(f"    {m:<26s} AP={v['AP']:.4f} R@1e-3={v['R_at_FP_0.001']:.3f} R@1e-2={v['R_at_FP_0.01']:.3f}")

    print("\n[k_sweep]")
    report["k_sweep"] = {}
    for K in [int(k) for k in args.k_list.split(",")]:
        t0 = time.time()
        sc, km = fit_roles(hf_full, K, args.seed)
        fr_k = full.join(assign_roles(hf_full, sc, km), on="host", how="left")
        ev = evaluate(score_instant(fr_k), pos_pairs,
                      methods=["rolling_fanout_72_z", "rolling_fanout_288_z"])
        report["k_sweep"][str(K)] = ev
        v72, v288 = ev["rolling_fanout_72_z"], ev["rolling_fanout_288_z"]
        print(f"    K={K:<3d} rf72_z: AP={v72['AP']:.4f} R@1e-3={v72['R_at_FP_0.001']:.3f} | "
              f"rf288_z: AP={v288['AP']:.4f} R@1e-3={v288['R_at_FP_0.001']:.3f} ({time.time()-t0:.0f}s)")

    print("\n[past_only] roles from benign prefix only")
    prefix = full.filter(pl.col("epoch_id") < first_attack_epoch)
    print(f"    prefix rows={prefix.height:,} "
          f"hosts={prefix['host'].n_unique():,} of {full['host'].n_unique():,}")
    report["past_only_prefix_hosts"] = prefix["host"].n_unique()
    report["past_only_total_hosts"] = full["host"].n_unique()
    hf_prefix = host_medians(prefix)
    scaler_p, km_p = fit_roles(hf_prefix, 16, args.seed)

    seen = set(hf_prefix["host"].to_list())
    unseen = [h for h in hf_full["host"].to_list() if h not in seen]
    if unseen:
        zeros = pl.DataFrame({"host": unseen,
                              **{c: [0.0] * len(unseen) for c in FEAT_COLS[:-1]},
                              "n_epochs_present": [0] * len(unseen)})
        zeros = zeros.with_columns([pl.col(c).cast(hf_prefix[c].dtype) for c in FEAT_COLS])
        hf_all_prefixfeat = pl.concat([hf_prefix, zeros.select(hf_prefix.columns)])
    else:
        hf_all_prefixfeat = hf_prefix
    roles_p = assign_roles(hf_all_prefixfeat, scaler_p, km_p)
    fr_p = full.join(roles_p, on="host", how="left")
    report["past_only_k16_instant"] = evaluate(score_instant(fr_p), pos_pairs)
    for m, v in report["past_only_k16_instant"].items():
        print(f"    {m:<26s} AP={v['AP']:.4f} R@1e-3={v['R_at_FP_0.001']:.3f} R@1e-2={v['R_at_FP_0.01']:.3f}")

    print("\n[ewma] causal cohort stats")
    report["ewma"] = {}
    for hl in [float(h) for h in args.ewma_half_lives.split(",")]:
        for tag, frame in (("full_roles", fr), ("past_only_roles", fr_p)):
            t0 = time.time()
            ev = evaluate(score_ewma(frame, hl), pos_pairs,
                          methods=["rolling_fanout_72_z", "rolling_fanout_288_z"])
            report["ewma"][f"hl{int(hl)}_{tag}"] = ev
            v = ev["rolling_fanout_288_z"]
            print(f"    hl={int(hl):>5d} {tag:<16s} rf288_z: AP={v['AP']:.4f} "
                  f"R@1e-3={v['R_at_FP_0.001']:.3f} R@1e-2={v['R_at_FP_0.01']:.3f} ({time.time()-t0:.0f}s)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[save] {args.out}  (total {time.time()-t_all:.0f}s)")

if __name__ == "__main__":
    main()
