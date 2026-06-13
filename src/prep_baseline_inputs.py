# Preprocessing: raw corpora to parquet inputs
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path

import polars as pl

PARQUET = "/private/workspace/icde_flow/data/lanl_2015/flows_minimal.parquet"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t0", type=int, default=100_000, help="start timestamp")
    p.add_argument("--window-sec", type=int, default=3600, help="slice duration in sec (default 1 h)")
    p.add_argument("--n-queries", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="/private/workspace/icde_flow/bench/slice1h")
    args = p.parse_args()
    random.seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t1 = args.t0 + args.window_sec
    tic = time.time()
    df = (pl.scan_parquet(PARQUET)
            .filter((pl.col("time") >= args.t0) & (pl.col("time") < t1))
            .filter(pl.col("src_comp") != pl.col("dst_comp"))
            .select(["src_comp", "dst_comp", "time"])
            .sort("time")
            .collect())
    print(f"Read {df.height:,} flows in {time.time()-tic:.2f}s")

    nodes = sorted(set(df["src_comp"].to_list()) | set(df["dst_comp"].to_list()))
    node2id = {v: i+1 for i, v in enumerate(nodes)}

    edges = []
    seen = set()
    for u_str, v_str, t in zip(df["src_comp"].to_list(),
                               df["dst_comp"].to_list(),
                               df["time"].to_list()):
        u, v = node2id[u_str], node2id[v_str]
        if u > v: u, v = v, u
        key = (u, v)
        if key in seen:
            continue
        seen.add(key)
        edges.append((u, v, t))
    print(f"Unique edges: {len(edges):,}; |V| = {len(nodes):,}")

    graph_path = out / "graph.txt"
    with open(graph_path, "w") as f:
        for u, v, t in edges:
            f.write(f"{u} {v} {t}\n")
    print(f"Wrote {graph_path} ({graph_path.stat().st_size/1e6:.2f} MB)")

    k_choices = [2, 3, 4, 5, 6, 8, 10]

    queries = []
    t_min, t_max = args.t0, t1 - 1
    for _ in range(args.n_queries):
        mode = random.choice(["full", "half", "narrow"])
        if mode == "full":
            ts, te = t_min, t_max
        elif mode == "half":
            mid = random.randint(t_min, t_max)
            half = (t_max - t_min) // 4
            ts = max(t_min, mid - half)
            te = min(t_max, mid + half)
        else:
            mid = random.randint(t_min, t_max)
            narrow = (t_max - t_min) // 20
            ts = max(t_min, mid - narrow)
            te = min(t_max, mid + narrow)
        k = random.choice(k_choices)
        queries.append((ts, te, k))

    queries_path = out / "queries.txt"
    with open(queries_path, "w") as f:
        for ts, te, k in queries:
            f.write(f"{ts}\t{te}\t{k}\n")
    print(f"Wrote {queries_path} with {len(queries)} queries")

    meta = {
        "source": str(PARQUET),
        "t0": args.t0, "t1": t1, "window_sec": args.window_sec,
        "n_flows_raw": df.height,
        "n_unique_edges": len(edges),
        "n_nodes": len(nodes),
        "k_choices": k_choices,
        "n_queries": len(queries),
        "graph_format": "u v t (SPACE-separated)",
        "query_format": "ts\\tte\\tk (TAB-separated)",
        "node_id_range": [1, len(nodes)],
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out / 'meta.json'}")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
