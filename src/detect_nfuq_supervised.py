# RQ4: TraceCore features + LightGBM supervised head-to-head (Table V)
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import polars as pl
import networkit as nk
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, roc_curve,
                              precision_score, recall_score, f1_score)
import lightgbm as lgb

def compute_host_features(flows: pl.DataFrame, src_col="IPV4_SRC_ADDR", dst_col="IPV4_DST_ADDR"):
    edges = (flows.select([src_col, dst_col]).unique()
                  .filter(pl.col(src_col) != pl.col(dst_col))
                  .with_columns([
                     pl.when(pl.col(src_col) < pl.col(dst_col)).then(pl.col(src_col)).otherwise(pl.col(dst_col)).alias("u"),
                     pl.when(pl.col(src_col) < pl.col(dst_col)).then(pl.col(dst_col)).otherwise(pl.col(src_col)).alias("v")])
                  .select(["u", "v"]).unique())
    nodes = sorted(set(edges["u"].to_list()) | set(edges["v"].to_list()))
    if not nodes:
        return pl.DataFrame({"host": [], "k_shell": [], "fanout_in": [], "fanout_out": []})
    node2id = {h: i for i, h in enumerate(nodes)}
    g = nk.Graph(len(nodes), directed=False)
    for u, v in zip(edges["u"].to_list(), edges["v"].to_list()):
        g.addEdge(node2id[u], node2id[v])
    cd = nk.centrality.CoreDecomposition(g); cd.run()
    shell_map = {nodes[i]: int(cd.scores()[i]) for i in range(len(nodes))}

    fan_out = (flows.select([src_col, dst_col]).unique()
                    .group_by(src_col).len().rename({src_col: "host", "len": "fanout_out"}))
    fan_in = (flows.select([src_col, dst_col]).unique()
                   .group_by(dst_col).len().rename({dst_col: "host", "len": "fanout_in"}))
    h = fan_out.join(fan_in, on="host", how="full").fill_null(0)
    h = h.with_columns(pl.coalesce(["host", "host_right"]).alias("host"))
    if "host_right" in h.columns: h = h.drop("host_right")
    shell_df = pl.DataFrame({"host": list(shell_map.keys()),
                             "k_shell": list(shell_map.values())})
    h = h.join(shell_df, on="host", how="left").fill_null(0)
    return h.select(["host", "k_shell", "fanout_in", "fanout_out"])

def make_features(flows: pl.DataFrame, host_feats: pl.DataFrame):
    src_f = host_feats.rename({"host": "IPV4_SRC_ADDR",
                                "k_shell": "src_kshell",
                                "fanout_in": "src_fanout_in",
                                "fanout_out": "src_fanout_out"})
    dst_f = host_feats.rename({"host": "IPV4_DST_ADDR",
                                "k_shell": "dst_kshell",
                                "fanout_in": "dst_fanout_in",
                                "fanout_out": "dst_fanout_out"})
    f = flows.join(src_f, on="IPV4_SRC_ADDR", how="left").fill_null(0)
    f = f.join(dst_f, on="IPV4_DST_ADDR", how="left").fill_null(0)
    feats = f.select([
        "src_kshell", "src_fanout_in", "src_fanout_out",
        "dst_kshell", "dst_fanout_in", "dst_fanout_out",
        "PROTOCOL",
        (pl.col("IN_BYTES").log1p()).alias("log_in_bytes"),
        (pl.col("OUT_BYTES").log1p()).alias("log_out_bytes"),
        "IN_PKTS", "OUT_PKTS",
        (pl.col("FLOW_DURATION_MILLISECONDS").log1p()).alias("log_duration"),
        "TCP_FLAGS",
        ((pl.col("OUT_BYTES") + 1) / (pl.col("IN_BYTES") + 1)).alias("byte_ratio"),
    ])
    return feats, f["Label"]

def report_metrics(y_true, y_score, name):
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fpr, tpr, thresh = roc_curve(y_true, y_score)
    recalls = {}
    for fp_t in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
        idx = np.searchsorted(fpr, fp_t, side="right") - 1
        recalls[f"R_at_FP_{fp_t}"] = float(tpr[max(0, idx)]) if idx >= 0 else 0.0

    y_pred = (y_score >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    return {
        "method": name, "ROC_AUC": float(auc), "PR_AUC": float(ap),
        "F1_at_0.5": float(f1), "precision_at_0.5": float(prec),
        "recall_at_0.5": float(rec), **recalls,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/private/workspace/icde_flow/data/nf_uq_v2/NF-UQ-NIDS-v2.csv")
    p.add_argument("--sub", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--fraction", type=float, default=None,
                   help="if set, subsample to this fraction (stratified by Attack) BEFORE split, matching GraphIDS protocol")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_path = args.out or f"/private/workspace/icde_flow/results/detect_nfuq_supervised/{args.sub}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] reading {args.sub}")
    t0 = time.perf_counter()
    cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "PROTOCOL", "IN_BYTES", "IN_PKTS",
            "OUT_BYTES", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "TCP_FLAGS",
            "Label", "Attack", "Dataset"]
    df = (pl.scan_csv(args.csv, schema_overrides={"FLOW_DURATION_MILLISECONDS": pl.Int64})
            .filter(pl.col("Dataset") == args.sub)
            .filter(pl.col("IPV4_SRC_ADDR") != pl.col("IPV4_DST_ADDR"))
            .select(cols)
            .collect())
    n_full = df.height
    print(f"[load] {n_full:,} flows in {time.perf_counter()-t0:.1f}s")
    if args.fraction is not None and 0 < args.fraction < 1:
        rng = np.random.default_rng(args.seed)

        rows_per_attack = df.group_by("Attack").agg(pl.col("Label").len().alias("n"))
        keep_indices = []
        for attack_name in rows_per_attack["Attack"].to_list():
            attack_mask = (df["Attack"] == attack_name).to_numpy()
            attack_idx = np.where(attack_mask)[0]
            keep_n = int(len(attack_idx) * args.fraction)
            chosen = rng.choice(attack_idx, size=keep_n, replace=False)
            keep_indices.append(chosen)
        keep_indices = np.concatenate(keep_indices)
        rng.shuffle(keep_indices)
        df = df[keep_indices.tolist()]
        print(f"[subsample] fraction={args.fraction}: kept {df.height:,} of {n_full:,} flows")
    n = df.height
    print(f"[final] {n:,} flows; benign={int((df['Label']==0).sum()):,}, attack={int((df['Label']==1).sum()):,}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    test_size = int(args.test_frac * n)
    test_idx = perm[:test_size]
    train_idx = perm[test_size:]
    train_df = df[train_idx.tolist()]
    test_df = df[test_idx.tolist()]
    print(f"[split] train={train_df.height:,} ({int(train_df['Label'].sum()):,} attack), test={test_df.height:,} ({int(test_df['Label'].sum()):,} attack)")

    print("[features] computing per-host structural features on TRAIN graph")
    t0 = time.perf_counter()
    host_feats = compute_host_features(train_df)
    print(f"[features] {host_feats.height:,} hosts in {time.perf_counter()-t0:.1f}s")

    print("[features] joining host features to flows")
    t0 = time.perf_counter()
    X_train, y_train = make_features(train_df, host_feats)
    X_test, y_test = make_features(test_df, host_feats)
    print(f"[features] train shape={X_train.shape}, test shape={X_test.shape}, "
          f"in {time.perf_counter()-t0:.1f}s")

    X_train_np = X_train.to_numpy()
    y_train_np = np.array(y_train.to_list())
    X_test_np = X_test.to_numpy()
    y_test_np = np.array(y_test.to_list())

    print("[train] LightGBM")
    t0 = time.perf_counter()
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63,
                              n_jobs=-1, random_state=args.seed, verbosity=-1)
    clf.fit(X_train_np, y_train_np)
    print(f"[train] done in {time.perf_counter()-t0:.1f}s")

    print("[eval] inference on test set")
    t0 = time.perf_counter()
    y_score = clf.predict_proba(X_test_np)[:, 1]
    print(f"[eval] {(time.perf_counter()-t0):.1f}s")

    feat_names = X_train.columns
    feat_imp = sorted(zip(feat_names, clf.feature_importances_),
                      key=lambda x: -x[1])
    print("[feat_imp] top features:")
    for n_, v in feat_imp[:10]:
        print(f"    {n_:<22s} {v}")

    m = report_metrics(y_test_np, y_score, "TraceCore_features + LightGBM")

    print()
    print("=== Headline metrics (TraceCore features + LightGBM, NF-CSE-CIC-IDS2018-v2 protocol) ===")
    for k, v in m.items():
        if k == "method": continue
        print(f"  {k}: {v:.4f}")

    print()
    print("[baseline] only flow features (no TraceCore features)")
    flow_cols = ["PROTOCOL", "log_in_bytes", "log_out_bytes", "IN_PKTS", "OUT_PKTS",
                 "log_duration", "TCP_FLAGS", "byte_ratio"]
    Xtr = X_train.select(flow_cols).to_numpy()
    Xte = X_test.select(flow_cols).to_numpy()
    clf2 = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63,
                               n_jobs=-1, random_state=args.seed, verbosity=-1)
    clf2.fit(Xtr, y_train_np)
    y_score_flow = clf2.predict_proba(Xte)[:, 1]
    m_flow = report_metrics(y_test_np, y_score_flow, "Flow_features only + LightGBM")

    tc_cols = ["src_kshell", "src_fanout_in", "src_fanout_out",
               "dst_kshell", "dst_fanout_in", "dst_fanout_out"]
    Xtr3 = X_train.select(tc_cols).to_numpy()
    Xte3 = X_test.select(tc_cols).to_numpy()
    clf3 = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63,
                               n_jobs=-1, random_state=args.seed, verbosity=-1)
    clf3.fit(Xtr3, y_train_np)
    y_score_tc = clf3.predict_proba(Xte3)[:, 1]
    m_tc = report_metrics(y_test_np, y_score_tc, "TraceCore_features only + LightGBM")

    print()
    print("=== Ablation: feature-set comparison ===")
    for mm in [m, m_flow, m_tc]:
        print(f"  {mm['method']:<40s} ROC_AUC={mm['ROC_AUC']:.4f}  PR_AUC={mm['PR_AUC']:.4f}  "
              f"F1@0.5={mm['F1_at_0.5']:.4f}  R@FP=1e-3={mm['R_at_FP_0.001']:.4f}  "
              f"R@FP=1e-2={mm['R_at_FP_0.01']:.4f}")

    report = {
        "sub_dataset": args.sub,
        "n_flows": n,
        "n_train": train_df.height, "n_test": test_df.height,
        "attack_rate": float(df["Label"].mean()),
        "feat_importance_top10": [{"feature": fn, "importance": int(v)} for fn, v in feat_imp[:10]],
        "results": {
            "TraceCore_plus_flow": m,
            "Flow_only": m_flow,
            "TraceCore_only": m_tc,
        },
    }
    Path(out_path).write_text(json.dumps(report, indent=2))
    print(f"[save] {out_path}")

if __name__ == "__main__":
    main()
