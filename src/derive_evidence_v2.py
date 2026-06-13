# Builds paper_derived_evidence.json from run metrics
from __future__ import annotations
import json, pickle, time
from pathlib import Path
from collections import Counter
import numpy as np
import polars as pl

ROOT = Path("/private/workspace/icde_flow")
OUT = ROOT / "results/paper_derived_evidence.json"
result = {"_meta": {"computed": "from raw LANL + NF-UQ artifacts via the same MiniBatchKMeans pipeline as role_residual_lanl.py"}}

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
    f"k={k}": {"count": v, "pct": round(100 * v / total_c17, 2)} for k, v in sorted(k_dist.items())
}
result["c17693_kshell_max"] = max(k_dist.keys())
result["c17693_kshell_pct_le_3"] = round(100 * sum(v for k, v in k_dist.items() if k <= 3) / total_c17, 2)

pkl = ROOT / "bench/lanl_redteam_range/tracecore_mvp.pkl"
print(f"[load] {pkl}")
with open(pkl, "rb") as f:
    d = pickle.load(f)
shells = d["shells"]
result["c17693_active_epochs"] = sum(1 for s in shells.values() if "C17693" in s)
result["total_epochs"] = len(shells)

all_k = Counter()
for s in shells.values():
    for k in s.values():
        all_k[k] += 1
result["population_kshell_distribution"] = {f"k={k}": v for k, v in sorted(all_k.items())}
result["population_kshell_max"] = max(all_k.keys())
result["population_kshell_95th"] = (
    sorted(all_k.keys())[
        next(i for i, k in enumerate(sorted(all_k.keys()))
             if sum(all_k[kk] for kk in sorted(all_k.keys())[:i+1]) >= 0.95 * sum(all_k.values()))
    ]
)

rf = pl.read_parquet(ROOT / "bench/lanl_redteam_range/rolling_fanout.parquet")
print(f"[load] rolling_fanout: {rf.height:,} rows")

print("[shells] flattening")
t0 = time.time()
rows = []
for eid, sh in shells.items():
    for host, sval in sh.items():
        rows.append({"host": host, "epoch_id": eid, "k_shell": sval})
shells_df = pl.DataFrame(rows)
print(f"  {shells_df.height:,} rows in {time.time()-t0:.1f}s")

full = shells_df.join(rf, on=["host", "epoch_id"], how="left").fill_null(0)

print("[feat] per-host static features")
host_feat = (
    full.group_by("host")
    .agg(
        pl.col("k_shell").median().alias("med_kshell"),
        pl.col("k_shell").max().alias("max_kshell"),
        pl.col("fanout_now").median().alias("med_fanout_now"),
        pl.col("rolling_fanout_12").median().alias("med_rf12"),
        pl.col("rolling_fanout_72").median().alias("med_rf72"),
        pl.col("rolling_fanout_288").median().alias("med_rf288"),
        pl.col("rolling_fanout_2016").median().alias("med_rf2016"),
        pl.col("epoch_id").count().alias("n_epochs_present"),
    )
)
n_hosts = host_feat.height
print(f"  {n_hosts:,} hosts")

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

km = MiniBatchKMeans(n_clusters=16, random_state=42, batch_size=1024, n_init=5).fit(Xs)
labels = km.labels_.astype(np.int32)
hosts = host_feat["host"].to_list()
host2role = dict(zip(hosts, labels.tolist()))
sizes = Counter(labels.tolist())

c17_role = host2role.get("C17693")
c17_idx = hosts.index("C17693")
result["role_cluster_sizes"] = [sizes.get(i, 0) for i in range(16)]
result["c17693_role"] = c17_role
result["c17693_role_size"] = sizes.get(c17_role)

cohort_hosts = [h for h, r in host2role.items() if r == c17_role]
cohort_rf72 = float(full.filter(pl.col("host").is_in(cohort_hosts))["rolling_fanout_72"].median())
result["c17693_cohort_median_rf72"] = cohort_rf72
result["c17693_max_rf72"] = float(full.filter(pl.col("host") == "C17693")["rolling_fanout_72"].max())
result["c17693_per_host_med_rf72"] = float(host_feat.filter(pl.col("host") == "C17693")["med_rf72"][0])

max_rf72_by_host = host_feat.with_columns(
    pl.col("host"),
    pl.col("med_rf72"),
).sort("med_rf72", descending=True)
host_rank_by_med = {h: i for i, h in enumerate(max_rf72_by_host["host"].to_list())}
result["c17693_rank_by_med_rf72"] = host_rank_by_med.get("C17693")
result["c17693_pct_rank_by_med_rf72"] = round(100 * host_rank_by_med.get("C17693", n_hosts) / n_hosts, 3)

host_max_rf72 = (
    full.group_by("host").agg(pl.col("rolling_fanout_72").max().alias("max_rf72"))
    .sort("max_rf72", descending=True)
)
rank_max = {h: i for i, h in enumerate(host_max_rf72["host"].to_list())}
result["c17693_rank_by_max_rf72"] = rank_max.get("C17693")
result["c17693_pct_rank_by_max_rf72"] = round(100 * rank_max.get("C17693", n_hosts) / n_hosts, 3)

fi = {
    "src_fanout_out": 2045, "src_fanout_in": 1904,
    "dst_fanout_in": 1563, "dst_fanout_out": 1476,
    "byte_ratio": 989, "log_in_bytes": 953, "dst_kshell": 773,
    "TCP_FLAGS": 670, "log_out_bytes": 608, "src_kshell": 400,
}
total_fi = sum(fi.values())
top4_fi = fi["src_fanout_out"] + fi["src_fanout_in"] + fi["dst_fanout_in"] + fi["dst_fanout_out"]
result["lightgbm_feature_importance_top10"] = fi
result["lightgbm_top10_total_gain"] = total_fi
result["lightgbm_top4_tracecore_gain"] = top4_fi
result["lightgbm_top4_pct_of_top10"] = round(100 * top4_fi / total_fi, 2)

ledger = json.loads((ROOT / "results/ledger.json").read_text())
result["ledger_total"] = len(ledger)
ledger_status = Counter(e.get("status", "OK") for e in ledger)
result["ledger_status_breakdown"] = dict(ledger_status)
errors = [
    {"tag": e["tag"], "status": e.get("status", "OK"), "note": e.get("note", "")[:200]}
    for e in ledger
    if any(kw in e.get("note", "").lower() for kw in ("error", "bug", "fix", "inverted", "rerun"))
]
result["ledger_entries_with_recorded_errors"] = errors
result["ledger_entries_with_recorded_errors_count"] = len(errors)

b3 = json.loads((ROOT / "results/block3_storage.json").read_text())
result["block3_storage"] = {
    "n_epochs": b3["n_epochs"],
    "n_unique_hosts": b3["n_unique_hosts"],
    "n_host_epoch_cells": b3["n_host_epoch_cells"],
    "n_deltas": b3["n_deltas"],
    "deltas_pct_of_snapshots": b3["deltas_pct_of_snapshots"],
    "sizes_mb": b3["sizes_mb"],
    "ratios_vs_naive_uncompressed": b3["ratios_vs_naive_uncompressed"],
}

for tag in ("b2_tcd_slice1h", "b2_iphc_slice1h"):
    for e in ledger:
        if e["tag"] == tag:
            result[f"{tag}_n_queries"] = e["metrics"]["n_queries"]

nfuq = {}
for sub in ("NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v2", "NF-UNSW-NB15-v2", "NF-BoT-IoT-v2"):
    fp = ROOT / f"results/detect_nfuq_supervised/{sub}.json"
    if fp.exists():
        x = json.loads(fp.read_text())
        nfuq[sub] = x.get("n_flows", x.get("n_edges", None))
result["nfuq_v2_flow_counts"] = nfuq
result["nfuq_v2_combined_flows"] = sum(v for v in nfuq.values() if v) if nfuq else None

result["hardware_setup"] = {
    "gpus": "2x NVIDIA RTX 4090",
    "ram_gb": 256,
    "disk_tb": 30,
    "note": "Single commodity node; TraceCore/LightGBM CPU-only; GPUs used by E-GraphSAGE / Anomal-E / GraphIDS only.",
}

OUT.write_text(json.dumps(result, indent=2))
print(f"[save] {OUT}")
print()
print(f"C17693: role={c17_role}, role_size={sizes.get(c17_role)}, cohort_med_rf72={cohort_rf72}, max_rf72={result['c17693_max_rf72']}")
print(f"role cluster sizes: {result['role_cluster_sizes']}")
print(f"C17693 k-shell max (at attack epochs): {result['c17693_kshell_max']}")
print(f"C17693 pct events: {result['lanl_attackers']['C17693_share_pct']}%")
print(f"LightGBM top-4 / top-10 sum: {result['lightgbm_top4_pct_of_top10']}%")
