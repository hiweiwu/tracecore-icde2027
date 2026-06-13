# RQ4: per-attack-class recall on NF-ToN-IoT-v2 (supplement)
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb

ROOT = Path("/private/workspace/icde_flow")
CSV = ROOT / "data/nf_uq_v2/NF-UQ-NIDS-v2.csv"

SEED = 42
TEST_FRAC = 0.2

print("[load] NF-ToN-IoT-v2")
t0 = time.time()
cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "PROTOCOL", "IN_BYTES", "IN_PKTS",
        "OUT_BYTES", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "TCP_FLAGS",
        "Label", "Attack", "Dataset"]
df = (pl.scan_csv(CSV, schema_overrides={"FLOW_DURATION_MILLISECONDS": pl.Int64})
        .filter(pl.col("Dataset") == "NF-ToN-IoT-v2")
        .filter(pl.col("IPV4_SRC_ADDR") != pl.col("IPV4_DST_ADDR"))
        .select(cols)
        .collect())
print(f"  {df.height:,} flows in {time.time()-t0:.1f}s")
print("attack classes:", df["Attack"].unique().to_list())

rng = np.random.default_rng(SEED)
perm = rng.permutation(df.height)
n_test = int(TEST_FRAC * df.height)
test_idx = perm[:n_test]
train_idx = perm[n_test:]
train_df = df[train_idx.tolist()]
test_df = df[test_idx.tolist()]

import sys
sys.path.insert(0, str(ROOT / "code"))
from detect_nfuq_supervised import compute_host_features, make_features

print("[features] computing structural features on TRAIN")
host_feats = compute_host_features(train_df)

print("[features] joining to flows")
X_train, y_train = make_features(train_df, host_feats)
X_test, y_test = make_features(test_df, host_feats)
X_train_np = X_train.to_numpy()
y_train_np = np.array(y_train.to_list())
X_test_np = X_test.to_numpy()
y_test_np = np.array(y_test.to_list())

print("[train] LightGBM (TraceCore+Flow)")
clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63,
                         n_jobs=-1, random_state=SEED, verbosity=-1)
clf.fit(X_train_np, y_train_np)
score_tc = clf.predict_proba(X_test_np)[:, 1]

print("[train] LightGBM (Flow only)")
flow_cols = ["PROTOCOL", "log_in_bytes", "log_out_bytes", "IN_PKTS", "OUT_PKTS",
             "log_duration", "TCP_FLAGS", "byte_ratio"]
Xtr_flow = X_train.select(flow_cols).to_numpy()
Xte_flow = X_test.select(flow_cols).to_numpy()
clf2 = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63,
                          n_jobs=-1, random_state=SEED, verbosity=-1)
clf2.fit(Xtr_flow, y_train_np)
score_flow = clf2.predict_proba(Xte_flow)[:, 1]

def threshold_at_fp(scores, labels, fp_rate):
    neg_scores = scores[labels == 0]
    neg_scores_sorted = np.sort(neg_scores)[::-1]
    n = max(1, int(fp_rate * len(neg_scores)))
    return neg_scores_sorted[n - 1]

attack_classes = test_df["Attack"].unique().to_list()
test_attacks = np.array(test_df["Attack"].to_list())

results = {}
for fp_rate in [0.01, 0.001]:
    tc_thresh = threshold_at_fp(score_tc, y_test_np, fp_rate)
    flow_thresh = threshold_at_fp(score_flow, y_test_np, fp_rate)
    per_class = {}
    for cls in attack_classes:
        mask = (test_attacks == cls) & (y_test_np == 1)
        if mask.sum() == 0:
            continue
        tc_recall = float((score_tc[mask] >= tc_thresh).mean())
        flow_recall = float((score_flow[mask] >= flow_thresh).mean())
        per_class[cls] = {"n_positives": int(mask.sum()),
                          "tracecore_recall": tc_recall,
                          "flow_only_recall": flow_recall,
                          "delta_pp": round((tc_recall - flow_recall) * 100, 1)}
    results[f"FP_{fp_rate}"] = per_class
    print(f"\n=== Per-class recall @ FP={fp_rate} ===")
    print(f"{'class':<20} {'n_pos':>8} {'TraceCore':>10} {'Flow-only':>10} {'Δ pp':>8}")
    for cls, r in sorted(per_class.items(), key=lambda x: -x[1]["n_positives"]):
        print(f"{cls:<20} {r['n_positives']:>8} {r['tracecore_recall']:>10.4f} {r['flow_only_recall']:>10.4f} {r['delta_pp']:>+8.1f}")

OUT = ROOT / "results" / "per_attack_toniot.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2))
print(f"\n[save] {OUT}")
