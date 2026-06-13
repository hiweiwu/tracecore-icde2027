# TraceCore-R: role assignment + cohort residual, retrospective roles (Table VI)
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import polars as pl

sys.path.insert(0, "/private/workspace/icde_flow/code")
from tracecore_mvp import TraceCoreMVP

def load_redteam(path: str):
    df = pl.read_csv(path, has_header=False, new_columns=["time", "user", "src", "dst"])
    return list(zip(df["time"].to_list(), df["src"].to_list()))

def compute_recall_at_fp(y_true: np.ndarray, y_score: np.ndarray, fp_targets):
    order = np.argsort(-y_score)
    y_t = y_true[order].astype(np.int64)
    cum_tp = np.cumsum(y_t)
    cum_fp = np.arange(1, len(y_t) + 1) - cum_tp
    n_pos = int(y_t.sum())
    n_neg = len(y_t) - n_pos
    out = {}
    for fp_t in fp_targets:
        max_fp = int(fp_t * n_neg)
        idx = np.searchsorted(cum_fp, max_fp, side="right") - 1
        out[f"R_at_FP_{fp_t}"] = float(cum_tp[max(0, idx)]) / max(1, n_pos) if idx >= 0 else 0.0

    pos_idx = np.where(y_t == 1)[0]
    ap = float(np.mean(cum_tp[pos_idx] / (pos_idx + 1))) if len(pos_idx) else 0.0
    out["AP"] = ap
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mvp", default="/private/workspace/icde_flow/bench/lanl_redteam_range/tracecore_mvp.pkl")
    p.add_argument("--rf", default="/private/workspace/icde_flow/bench/lanl_redteam_range/rolling_fanout.parquet")
    p.add_argument("--redteam", default="/private/workspace/icde_flow/data/lanl_2015/redteam.txt.gz")
    p.add_argument("--n-roles", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/private/workspace/icde_flow/results/role_residual_lanl.json")
    args = p.parse_args()

    print(f"[load] TraceCore MVP from {args.mvp}")
    tc = TraceCoreMVP.load(args.mvp)
    print(f"[load] {len(tc.shells)} epochs, epoch_sec={tc.epoch_sec}")

    print(f"[load] rolling fanout from {args.rf}")
    rf = pl.read_parquet(args.rf)
    print(f"[load] rolling-fanout: {rf.height:,} rows, cols={rf.columns}")

    rt = load_redteam(args.redteam)
    pos_pairs = {(src, tc.epoch_of(t)) for t, src in rt}
    print(f"[redteam] {len(rt)} events, {len(pos_pairs)} unique (host, epoch) positives")

    print("[shells] flattening per-host per-epoch k_shell")
    t0 = time.time()
    rows = []
    for eid, shell_dict in tc.shells.items():
        for host, shell in shell_dict.items():
            rows.append({"host": host, "epoch_id": eid, "k_shell": shell})
    shells_df = pl.DataFrame(rows)
    print(f"[shells] {shells_df.height:,} (host, epoch, kshell) rows in {time.time()-t0:.1f}s")

    print("[join] shells + rolling_fanout")
    full = shells_df.join(rf, on=["host", "epoch_id"], how="left").fill_null(0)
    print(f"[join] {full.height:,} rows; cols: {full.columns}")

    print("[feat] per-host static features (medians)")
    t0 = time.time()
    host_feat = (full.group_by("host")
                     .agg([pl.col("k_shell").median().alias("med_kshell"),
                           pl.col("k_shell").max().alias("max_kshell"),
                           pl.col("fanout_now").median().alias("med_fanout_now"),
                           pl.col("rolling_fanout_12").median().alias("med_rf12"),
                           pl.col("rolling_fanout_72").median().alias("med_rf72"),
                           pl.col("rolling_fanout_288").median().alias("med_rf288"),
                           pl.col("rolling_fanout_2016").median().alias("med_rf2016"),
                           pl.col("epoch_id").count().alias("n_epochs_present")]))
    n_hosts = host_feat.height
    print(f"[feat] {n_hosts:,} hosts in {time.time()-t0:.1f}s")

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    feat_cols = ["med_kshell", "max_kshell", "med_fanout_now",
                 "med_rf12", "med_rf72", "med_rf288", "med_rf2016", "n_epochs_present"]
    X = host_feat.select(feat_cols).to_numpy().astype(np.float64)

    for j, col in enumerate(feat_cols):
        if col.startswith("med") or col == "n_epochs_present":
            X[:, j] = np.log1p(np.maximum(X[:, j], 0))
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    print(f"[kmeans] clustering {n_hosts} hosts → {args.n_roles} roles")
    km = MiniBatchKMeans(n_clusters=args.n_roles, random_state=args.seed,
                          batch_size=1024, n_init=5).fit(Xs)
    labels = km.labels_.astype(np.int32)
    host_role = pl.DataFrame({"host": host_feat["host"].to_list(),
                              "role": labels.tolist()})

    sizes = np.bincount(labels)
    print(f"[kmeans] cluster sizes: {sizes.tolist()}")

    print("[cohort] per-(role, epoch) mean/std of features")
    full = full.join(host_role, on="host", how="left")
    cohort_cols = ["rolling_fanout_12", "rolling_fanout_72", "rolling_fanout_288", "k_shell"]
    cohort = (full.group_by(["role", "epoch_id"])
                  .agg([pl.col(c).mean().alias(f"mu_{c}") for c in cohort_cols] +
                       [pl.col(c).std().alias(f"sigma_{c}") for c in cohort_cols] +
                       [pl.col("host").count().alias("n_hosts_in_cohort")]))
    print(f"[cohort] {cohort.height:,} (role, epoch) cells")

    print("[score] computing residual z-scores")
    scored = full.join(cohort, on=["role", "epoch_id"], how="left")

    residual_cols = []
    for c in cohort_cols:
        z_name = f"{c}_z"
        scored = scored.with_columns(
            ((pl.col(c) - pl.col(f"mu_{c}")) /
             pl.col(f"sigma_{c}").fill_null(1.0).clip(0.5, None)).alias(z_name))
        residual_cols.append(z_name)

    scored = scored.with_columns([
        (pl.col("rolling_fanout_72") / (pl.col("k_shell") + 1)).alias("raw_lsf72"),
        (pl.col("rolling_fanout_72_z") / (pl.col("k_shell_z").abs() + 1)).alias("residual_lsf72"),
    ])

    scored = scored.with_columns(
        pl.struct(["host", "epoch_id"]).map_elements(
            lambda s: (s["host"], s["epoch_id"]) in pos_pairs, return_dtype=pl.Boolean
        ).alias("is_positive"))
    n_pos = int(scored["is_positive"].sum())
    n_neg = scored.height - n_pos
    print(f"[score] positives={n_pos}, negatives={n_neg:,}")

    fp_targets = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    methods = ["k_shell", "rolling_fanout_12", "rolling_fanout_72",
               "rolling_fanout_288", "raw_lsf72"] + residual_cols + ["residual_lsf72"]
    results = {}
    print("[eval] per-method operating-point curves")
    for m in methods:
        y_true = scored["is_positive"].to_numpy().astype(np.int64)
        y_score = scored[m].fill_null(0.0).to_numpy().astype(np.float64)

        y_score = np.nan_to_num(y_score, nan=0.0, posinf=1e9, neginf=-1e9)
        rec = compute_recall_at_fp(y_true, y_score, fp_targets)
        results[m] = rec
        head = f"R@1e-3={rec['R_at_FP_0.001']:.3f}  R@1e-2={rec['R_at_FP_0.01']:.3f}  R@5e-2={rec['R_at_FP_0.05']:.3f}"
        print(f"    {m:<26s} AP={rec['AP']:.4f}  {head}")

    report = {
        "mvp": args.mvp, "rf": args.rf,
        "n_hosts": n_hosts, "n_roles": args.n_roles,
        "cluster_sizes": sizes.tolist(),
        "n_positives": n_pos, "n_negatives": n_neg,
        "operating_points": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[save] {args.out}")

if __name__ == "__main__":
    main()
