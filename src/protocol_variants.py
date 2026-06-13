# RQ4: inductive / transductive / host-group split variants (Table IX)
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "/private/workspace/icde_flow/code")
from detect_nfuq_supervised import compute_host_features, make_features, report_metrics

import lightgbm as lgb

CSV = "/private/workspace/icde_flow/data/nf_uq_v2/NF-UQ-NIDS-v2.csv"
COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "PROTOCOL", "IN_BYTES", "IN_PKTS",
        "OUT_BYTES", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "TCP_FLAGS",
        "Label", "Attack", "Dataset"]

def load_sub(sub: str, fraction: float | None, seed: int) -> pl.DataFrame:
    df = (pl.scan_csv(CSV, schema_overrides={"FLOW_DURATION_MILLISECONDS": pl.Int64})
            .filter(pl.col("Dataset") == sub)
            .filter(pl.col("IPV4_SRC_ADDR") != pl.col("IPV4_DST_ADDR"))
            .select(COLS)
            .collect())
    n_full = df.height
    if fraction is not None and 0 < fraction < 1:
        rng = np.random.default_rng(seed)
        keep_indices = []
        for attack_name in df.group_by("Attack").agg(pl.len())["Attack"].to_list():
            attack_idx = np.where((df["Attack"] == attack_name).to_numpy())[0]
            keep_n = int(len(attack_idx) * fraction)
            keep_indices.append(rng.choice(attack_idx, size=keep_n, replace=False))
        keep_indices = np.concatenate(keep_indices)
        rng.shuffle(keep_indices)
        df = df[keep_indices.tolist()]
        print(f"[subsample] fraction={fraction}: kept {df.height:,} of {n_full:,}")
    return df

def fit_eval(train_df: pl.DataFrame, test_df: pl.DataFrame, feat_source_df: pl.DataFrame,
             seed: int, name: str) -> dict:
    t0 = time.perf_counter()
    host_feats = compute_host_features(feat_source_df)
    feat_sec = time.perf_counter() - t0
    X_train, y_train = make_features(train_df, host_feats)
    X_test, y_test = make_features(test_df, host_feats)
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63,
                              n_jobs=-1, random_state=seed, verbosity=-1)
    t0 = time.perf_counter()
    clf.fit(X_train.to_numpy(), np.array(y_train.to_list()))
    fit_sec = time.perf_counter() - t0
    y_score = clf.predict_proba(X_test.to_numpy())[:, 1]
    m = report_metrics(np.array(y_test.to_list()), y_score, name)
    m.update({"n_train": train_df.height, "n_test": test_df.height,
              "n_hosts_in_feature_graph": host_feats.height,
              "feature_sec": round(feat_sec, 1), "fit_sec": round(fit_sec, 1),
              "test_attack_rate": float(np.array(y_test.to_list()).mean())})
    return m

def overlap_stats(train_df: pl.DataFrame, test_df: pl.DataFrame) -> dict:
    train_hosts = set(train_df["IPV4_SRC_ADDR"].to_list()) | set(train_df["IPV4_DST_ADDR"].to_list())
    test_src = test_df["IPV4_SRC_ADDR"].to_list()
    test_dst = test_df["IPV4_DST_ADDR"].to_list()
    src_in = np.array([s in train_hosts for s in test_src])
    dst_in = np.array([d in train_hosts for d in test_dst])
    test_hosts = set(test_src) | set(test_dst)
    return {
        "n_train_hosts": len(train_hosts),
        "n_test_hosts": len(test_hosts),
        "n_test_only_hosts": len(test_hosts - train_hosts),
        "pct_test_flows_src_in_train_graph": round(float(src_in.mean()) * 100, 2),
        "pct_test_flows_dst_in_train_graph": round(float(dst_in.mean()) * 100, 2),
        "pct_test_flows_both_in_train_graph": round(float((src_in & dst_in).mean()) * 100, 2),
        "pct_test_flows_neither_in_train_graph": round(float((~src_in & ~dst_in).mean()) * 100, 2),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sub", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fraction", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_path = args.out or f"/private/workspace/icde_flow/results/protocol_variants/{args.sub}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.sub}")
    df = load_sub(args.sub, args.fraction, args.seed)
    n = df.height
    print(f"[load] {n:,} flows after subsample")

    report = {"sub": args.sub, "fraction": args.fraction, "seed": args.seed,
              "n_flows": n, "variants": {}}

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    test_size = int(args.test_frac * n)
    test_idx, train_idx = perm[:test_size], perm[test_size:]
    train_df, test_df = df[train_idx.tolist()], df[test_idx.tolist()]
    print(f"[standard] train={train_df.height:,} test={test_df.height:,}")

    print("[standard] host-overlap statistics")
    report["host_overlap_standard_split"] = overlap_stats(train_df, test_df)
    print(json.dumps(report["host_overlap_standard_split"], indent=2))

    print("[standard] train-only feature graph (paper protocol)")
    report["variants"]["standard_train_only_graph"] = fit_eval(
        train_df, test_df, train_df, args.seed, "standard_train_only_graph")

    print("[transductive] full-graph features (NOT the paper protocol; reference only)")
    report["variants"]["transductive_full_graph"] = fit_eval(
        train_df, test_df, df, args.seed, "transductive_full_graph")

    print("[host_group] splitting by src host")
    src_attack = (df.group_by("IPV4_SRC_ADDR")
                    .agg([pl.col("Label").max().alias("has_attack"), pl.len().alias("n_flows")]))
    rng = np.random.default_rng(args.seed)
    held_out = []
    for ha in (0, 1):
        hosts = src_attack.filter(pl.col("has_attack") == ha)["IPV4_SRC_ADDR"].to_list()
        k = max(1, int(0.2 * len(hosts))) if hosts else 0
        if k:
            held_out.extend(rng.choice(np.array(hosts, dtype=object), size=k, replace=False).tolist())
    held_set = set(held_out)
    test_mask = np.array([s in held_set for s in df["IPV4_SRC_ADDR"].to_list()])
    gtest_df, gtrain_df = df.filter(pl.Series(test_mask)), df.filter(pl.Series(~test_mask))
    print(f"[host_group] held-out src hosts={len(held_set):,}; "
          f"train={gtrain_df.height:,} test={gtest_df.height:,} "
          f"(test attack rate={float(gtest_df['Label'].mean()):.4f})")
    if gtest_df.height == 0 or int(gtest_df["Label"].sum()) == 0:
        report["variants"]["host_group_split"] = {
            "skipped": True,
            "reason": f"degenerate: test flows={gtest_df.height}, attack flows={int(gtest_df['Label'].sum()) if gtest_df.height else 0}"}
    else:
        report["variants"]["host_group_split"] = fit_eval(
            gtrain_df, gtest_df, gtrain_df, args.seed, "host_group_split")
        report["variants"]["host_group_split"]["n_held_out_src_hosts"] = len(held_set)

    Path(out_path).write_text(json.dumps(report, indent=2))
    print(f"[save] {out_path}")
    for vname, v in report["variants"].items():
        if v.get("skipped"):
            print(f"  {vname}: SKIPPED ({v['reason']})")
        else:
            print(f"  {vname:<28s} F1={v['F1_at_0.5']:.4f} PR={v['PR_AUC']:.4f} "
                  f"ROC={v['ROC_AUC']:.4f} R@1e-3={v['R_at_FP_0.001']:.4f}")

if __name__ == "__main__":
    main()
