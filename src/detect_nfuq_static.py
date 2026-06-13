# RQ4: static-snapshot sanity check (supplement)
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import polars as pl
import numpy as np
import networkit as nk

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/private/workspace/icde_flow/data/nf_uq_v2/NF-UQ-NIDS-v2.csv")
    p.add_argument("--sub", required=True,
                   choices=["NF-BoT-IoT-v2", "NF-ToN-IoT-v2",
                            "NF-UNSW-NB15-v2", "NF-CSE-CIC-IDS2018-v2", "ALL"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_path = args.out or f"/private/workspace/icde_flow/results/detect_nfuq_static/{args.sub}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] reading sub-dataset {args.sub} from {args.csv}")
    t0 = time.perf_counter()
    lf = pl.scan_csv(args.csv,
                     schema_overrides={"FLOW_DURATION_MILLISECONDS": pl.Int64})
    if args.sub != "ALL":
        lf = lf.filter(pl.col("Dataset") == args.sub)
    df = (lf.select(["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label"])
            .filter(pl.col("IPV4_SRC_ADDR") != pl.col("IPV4_DST_ADDR"))
            .collect())
    print(f"[load] {df.height:,} flows in {time.perf_counter()-t0:.1f}s")

    print("[agg] per-host stats")
    t0 = time.perf_counter()

    src_stats = (df.group_by("IPV4_SRC_ADDR")
                   .agg([pl.len().alias("n_total_as_src"),
                         pl.col("Label").sum().alias("n_attack_as_src")])
                   .rename({"IPV4_SRC_ADDR": "host"}))

    fanout_out = (df.select(["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]).unique()
                    .group_by("IPV4_SRC_ADDR").len().rename({"IPV4_SRC_ADDR": "host", "len": "fanout_out"}))
    fanout_in = (df.select(["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]).unique()
                   .group_by("IPV4_DST_ADDR").len().rename({"IPV4_DST_ADDR": "host", "len": "fanout_in"}))
    h1 = src_stats.join(fanout_out, on="host", how="full").fill_null(0)
    h1 = h1.with_columns(pl.coalesce(["host", "host_right"]).alias("host")).drop("host_right")
    host_stats = h1.join(fanout_in, on="host", how="full").fill_null(0)
    host_stats = host_stats.with_columns(pl.coalesce(["host", "host_right"]).alias("host")).drop("host_right")
    host_stats = host_stats.with_columns(
        (pl.col("fanout_out") + pl.col("fanout_in")).alias("fanout"))
    print(f"[agg] {host_stats.height:,} unique hosts in {time.perf_counter()-t0:.1f}s")

    print("[graph] building undirected graph from unique (src, dst) pairs")
    t0 = time.perf_counter()
    edges = (df.select(["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]).unique()
               .with_columns([
                  pl.when(pl.col("IPV4_SRC_ADDR") < pl.col("IPV4_DST_ADDR"))
                    .then(pl.col("IPV4_SRC_ADDR")).otherwise(pl.col("IPV4_DST_ADDR")).alias("u"),
                  pl.when(pl.col("IPV4_SRC_ADDR") < pl.col("IPV4_DST_ADDR"))
                    .then(pl.col("IPV4_DST_ADDR")).otherwise(pl.col("IPV4_SRC_ADDR")).alias("v")])
               .select(["u", "v"]).unique())
    nodes = sorted(set(edges["u"].to_list()) | set(edges["v"].to_list()))
    node2id = {h: i for i, h in enumerate(nodes)}
    g = nk.Graph(len(nodes), directed=False)
    for u, v in zip(edges["u"].to_list(), edges["v"].to_list()):
        g.addEdge(node2id[u], node2id[v])
    print(f"[graph] |V|={len(nodes):,}, |E|={edges.height:,} in {time.perf_counter()-t0:.1f}s")

    print("[kshell] running CoreDecomposition")
    t0 = time.perf_counter()
    cd = nk.centrality.CoreDecomposition(g); cd.run()
    shell = {nodes[i]: int(cd.scores()[i]) for i in range(len(nodes))}
    print(f"[kshell] max_shell={max(shell.values())} in {time.perf_counter()-t0:.1f}s")

    shell_df = pl.DataFrame({"host": list(shell.keys()), "k_shell": list(shell.values())})
    h = host_stats.join(shell_df, on="host", how="left").fill_null(0)
    h = h.with_columns([
        (pl.col("fanout") / (pl.col("k_shell") + 1)).alias("low_shell_fanout"),
        (pl.col("fanout") - 10 * pl.col("k_shell")).alias("penalty_score"),
        (pl.col("n_attack_as_src") >= 1).alias("is_positive"),
    ])
    n_pos = int(h["is_positive"].sum())
    n_neg = h.height - n_pos
    print(f"[label] hosts with ≥1 attack-as-src (positive): {n_pos:,}")
    print(f"[label] hosts pure benign / victim-only (negative): {n_neg:,}")
    h.write_parquet(Path(out_path).with_suffix(".parquet"), compression="zstd")

    if n_pos == 0:
        print("[op] no positives — cannot compute operating points")
        return

    print("[op] operating-point curves")
    fp_targets = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    methods = ["k_shell", "fanout", "low_shell_fanout", "penalty_score"]
    op = {}
    for m in methods:
        sd = h.sort(m, descending=True)
        pos_arr = sd["is_positive"].to_numpy()
        cum_tp = np.cumsum(pos_arr).astype(np.int64)
        cum_fp = np.arange(1, len(pos_arr) + 1) - cum_tp
        recalls = {}
        for fp_t in fp_targets:
            max_fp = int(fp_t * n_neg)
            idx = np.searchsorted(cum_fp, max_fp, side="right") - 1
            recalls[f"recall_at_FP_{fp_t}"] = float(cum_tp[max(0, idx)]) / max(1, n_pos) if idx >= 0 else 0.0

        pos_idx = np.where(pos_arr)[0]
        ap = float(np.mean(cum_tp[pos_idx] / (pos_idx + 1))) if len(pos_idx) else 0.0
        op[m] = {**recalls, "AP": ap}
        print(f"    {m:<20s} AP={ap:.3f}  "
              f"R@1e-3={recalls['recall_at_FP_0.001']:.3f}  "
              f"R@1e-2={recalls['recall_at_FP_0.01']:.3f}  "
              f"R@5e-2={recalls['recall_at_FP_0.05']:.3f}  "
              f"R@1e-1={recalls['recall_at_FP_0.1']:.3f}")

    report = {
        "sub_dataset": args.sub,
        "n_flows": df.height, "n_hosts": h.height, "n_edges": edges.height,
        "max_kshell": max(shell.values()),
        "n_positives": n_pos, "n_negatives": n_neg,
        "operating_points": op,
    }
    Path(out_path).write_text(json.dumps(report, indent=2))
    print(f"[save] {out_path}")

if __name__ == "__main__":
    main()
