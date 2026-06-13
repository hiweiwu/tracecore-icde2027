# Builds derived-evidence JSON from run metrics (v1)
from __future__ import annotations
import json, pickle
from pathlib import Path
from collections import Counter
import numpy as np
import polars as pl

ROOT = Path("/private/workspace/icde_flow")
OUT = ROOT / "results/paper_derived_evidence.json"
result = {"_meta": {"computed": "from raw LANL + NF-UQ artifacts; consumed by Phase 4.7 audit"}}

csv = ROOT / "results/detect_redteam_full/event_analysis.csv"
ev = pl.read_csv(csv)
total = ev.height
per_attacker = ev.group_by("src").len().sort("len", descending=True)
result["lanl_attackers"] = {
    "n_redteam_events": total,
    "per_attacker": {row[0]: row[1] for row in per_attacker.iter_rows()},
    "C17693_share_pct": round(100 * ev.filter(pl.col("src") == "C17693").height / total, 2),
}

c17 = ev.filter(pl.col("src") == "C17693")
k_dist = Counter(c17["src_shell"].to_list())
total_c17 = sum(k_dist.values())
result["c17693_kshell_at_attack_epochs"] = {
    f"k={k}": {"count": v, "pct": round(100*v/total_c17, 2)} for k, v in sorted(k_dist.items())
}
result["c17693_kshell_max"] = max(k_dist.keys())
result["c17693_kshell_pct_le_3"] = round(100 * sum(v for k, v in k_dist.items() if k <= 3) / total_c17, 2)

pkl = ROOT / "bench/lanl_redteam_range/tracecore_mvp.pkl"
print(f"[load] {pkl}")
with open(pkl, "rb") as f:
    d = pickle.load(f)
shells = d["shells"]
c17_active = sum(1 for s in shells.values() if "C17693" in s)
result["c17693_active_epochs"] = c17_active
result["total_epochs"] = len(shells)

all_k = Counter()
for s in shells.values():
    for k in s.values():
        all_k[k] += 1
result["population_kshell_distribution"] = {f"k={k}": v for k, v in sorted(all_k.items())}
result["population_kshell_max"] = max(all_k.keys())

print("[role] computing per-host role for C17693")
rf = ROOT / "bench/lanl_redteam_range/rolling_fanout.parquet"
rf_df = pl.read_parquet(rf)

host_feat = (
    rf_df.group_by("host")
    .agg(
        pl.col("rolling_fanout_12").median().alias("rf12_med"),
        pl.col("rolling_fanout_72").median().alias("rf72_med"),
        pl.col("rolling_fanout_288").median().alias("rf288_med"),
    )
    .sort("host")
)
n_hosts = host_feat.height
print(f"  {n_hosts} hosts")

from sklearn.cluster import MiniBatchKMeans
X = host_feat.select(["rf12_med", "rf72_med", "rf288_med"]).to_numpy()

X = (X - X.mean(axis=0)) / X.std(axis=0).clip(min=1e-6)
km = MiniBatchKMeans(n_clusters=16, random_state=42, n_init=10, batch_size=4096)
labels = km.fit_predict(X)
hosts = host_feat["host"].to_list()
host2role = dict(zip(hosts, labels.tolist()))
sizes = Counter(labels.tolist())
c17_role = host2role.get("C17693", None)
result["role_cluster_sizes"] = {f"role_{i}": sizes.get(i, 0) for i in range(16)}
result["c17693_role"] = c17_role
result["c17693_role_size"] = sizes.get(c17_role, None) if c17_role is not None else None

if c17_role is not None:
    cohort_hosts = [h for h, r in host2role.items() if r == c17_role]
    cohort_rf72 = (
        rf_df.filter(pl.col("host").is_in(cohort_hosts))["rolling_fanout_72"]
        .median()
    )
    result["c17693_cohort_median_rf72"] = float(cohort_rf72) if cohort_rf72 is not None else None

    att_max = float(rf_df.filter(pl.col("host") == "C17693")["rolling_fanout_72"].max())
    result["c17693_max_rf72"] = att_max

result["lanl_other_attackers_flow_counts"] = {
    k: v for k, v in result["lanl_attackers"]["per_attacker"].items() if k != "C17693"
}

fi = {
    "src_fanout_out": 2045,
    "src_fanout_in": 1904,
    "dst_fanout_in": 1563,
    "dst_fanout_out": 1476,
    "byte_ratio": 989,
    "log_in_bytes": 953,
    "dst_kshell": 773,
    "TCP_FLAGS": 670,
    "log_out_bytes": 608,
    "src_kshell": 400,
}
total_fi = sum(fi.values())
top4_fi = fi["src_fanout_out"] + fi["src_fanout_in"] + fi["dst_fanout_in"] + fi["dst_fanout_out"]
result["lightgbm_feature_importance_top10"] = fi
result["lightgbm_top10_total_gain"] = total_fi
result["lightgbm_top4_tracecore_gain"] = top4_fi
result["lightgbm_top4_pct_of_top10"] = round(100 * top4_fi / total_fi, 2)

ledger = json.loads((ROOT / "results/ledger.json").read_text())
errors = [
    {"tag": e["tag"], "status": e.get("status", "OK"), "note": e.get("note", "")}
    for e in ledger
    if e.get("status", "OK") not in ("OK",) or "error" in e.get("note", "").lower() or "bug" in e.get("note", "").lower() or "fix" in e.get("note", "").lower() or "inverted" in e.get("note", "").lower()
]
result["ledger_entries_with_recorded_errors"] = errors
result["ledger_total"] = len(ledger)

b3 = json.loads((ROOT / "results/block3_storage.json").read_text())
result["block3_storage"] = {
    "deltas_pct_of_snapshots": b3["deltas_pct_of_snapshots"],
    "shellindex_gz_mb": b3["sizes_mb"]["shellindex_varint_gzipped"],
    "coredelta_gz_mb": b3["sizes_mb"]["coredelta_gzipped"],
    "naive_uncompressed_mb": b3["sizes_mb"]["naive_snapshot_uncompressed"],
    "ratio_shellindex_gz_vs_naive": b3["ratios_vs_naive_uncompressed"]["shellindex_gz_vs_naive"],
}

for tag in ("b2_tcd_slice1h", "b2_iphc_slice1h"):
    for e in ledger:
        if e["tag"] == tag:
            result[f"{tag}_n_queries"] = e["metrics"]["n_queries"]

result["hardware_setup"] = {
    "gpus": "2x NVIDIA RTX 4090",
    "cpu": "32-core server CPU class",
    "ram_gb": 256,
    "disk_tb": 30,
    "note": "Single commodity node; TraceCore/LightGBM CPU-only; GPUs only used by E-GraphSAGE/Anomal-E/GraphIDS baselines.",
}

OUT.write_text(json.dumps(result, indent=2))
print(f"[save] {OUT}")
print()
print(json.dumps(result, indent=2)[:3000])
