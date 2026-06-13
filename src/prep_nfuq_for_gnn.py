# Preprocessing: NF-UQ-NIDS-v2 inputs for the GNN baselines
from __future__ import annotations
import argparse, time
from pathlib import Path

import numpy as np
import polars as pl

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/private/workspace/icde_flow/data/nf_uq_v2/NF-UQ-NIDS-v2.csv")
    p.add_argument("--sub", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    t0 = time.time()
    print(f"[load] reading {args.sub}")
    df = (pl.scan_csv(args.csv, schema_overrides={"FLOW_DURATION_MILLISECONDS": pl.Int64})
            .filter(pl.col("Dataset") == args.sub)
            .filter(pl.col("IPV4_SRC_ADDR") != pl.col("IPV4_DST_ADDR"))
            .select(["IPV4_SRC_ADDR", "IPV4_DST_ADDR",
                     "PROTOCOL", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
                     "FLOW_DURATION_MILLISECONDS", "TCP_FLAGS", "Label"])
            .collect())
    print(f"[load] {df.height:,} flows in {time.time()-t0:.1f}s")

    nodes = sorted(set(df["IPV4_SRC_ADDR"].to_list()) | set(df["IPV4_DST_ADDR"].to_list()))
    node2id = {h: i for i, h in enumerate(nodes)}
    print(f"[graph] {len(nodes):,} nodes")

    src_ids = np.array([node2id[h] for h in df["IPV4_SRC_ADDR"].to_list()], dtype=np.uint32)
    dst_ids = np.array([node2id[h] for h in df["IPV4_DST_ADDR"].to_list()], dtype=np.uint32)
    edge_attr = np.column_stack([
        df["PROTOCOL"].to_numpy().astype(np.float32),
        np.log1p(np.maximum(df["IN_BYTES"].to_numpy().astype(np.float32), 0)),
        np.log1p(np.maximum(df["OUT_BYTES"].to_numpy().astype(np.float32), 0)),
        df["IN_PKTS"].to_numpy().astype(np.float32),
        df["OUT_PKTS"].to_numpy().astype(np.float32),
        np.log1p(np.maximum(df["FLOW_DURATION_MILLISECONDS"].to_numpy().astype(np.float32), 0)),
        df["TCP_FLAGS"].to_numpy().astype(np.float32),
    ])
    labels = df["Label"].to_numpy().astype(np.uint8)
    print(f"[arrays] edge_attr shape={edge_attr.shape}, attack_rate={labels.mean():.3f}")

    np.savez_compressed(
        args.out,
        src_ids=src_ids, dst_ids=dst_ids, edge_attr=edge_attr,
        labels=labels, n_nodes=np.int32(len(nodes)),
    )
    sz = Path(args.out).stat().st_size / 1e6
    print(f"[save] {args.out} ({sz:.1f} MB) in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
