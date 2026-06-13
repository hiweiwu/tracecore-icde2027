# RQ1 stage 1: point_coreness vs NetworKit/igraph oracles (Table IV)
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path

import polars as pl
import networkit as nk
import igraph as ig

LANL_PATH = "/private/workspace/icde_flow/data/lanl_2015/flows.txt.gz"
SCHEMA = ["time", "duration", "src_comp", "src_port", "dst_comp", "dst_port",
          "protocol", "packet_count", "byte_count"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-windows", type=int, default=5,
                   help="number of randomly-sampled windows to evaluate")
    p.add_argument("--window-sec", type=int, default=300,
                   help="window length in seconds (default 5 min)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/private/workspace/icde_flow/results/block1_stage1.json")
    p.add_argument("--t-max", type=int, default=3_126_928,
                   help="max timestamp in LANL flows.txt (probed earlier)")
    args = p.parse_args()

    random.seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"Block 1 Stage 1 — {args.n_windows} windows × {args.window_sec}s")
    print(f"LANL source: {LANL_PATH}")

    starts = sorted(random.sample(range(0, args.t_max - args.window_sec),
                                  args.n_windows))
    print(f"Sampled starts: {starts[:5]}{'...' if len(starts) > 5 else ''}")

    lf = pl.scan_csv(LANL_PATH, has_header=False, new_columns=SCHEMA,
                     schema_overrides={"time": pl.Int64})

    results = []
    overall_pass = True
    for i, t0 in enumerate(starts):
        t1 = t0 + args.window_sec
        tic = time.time()

        df = (lf.filter((pl.col("time") >= t0) & (pl.col("time") < t1))
                .select(["src_comp", "dst_comp"])
                .filter(pl.col("src_comp") != pl.col("dst_comp"))
                .unique()
                .collect())

        n_edges_raw = df.height
        if n_edges_raw == 0:
            print(f"  W{i:02d} [t={t0}..{t1}] EMPTY — skipping")
            results.append({"window": i, "t0": t0, "t1": t1,
                            "n_edges": 0, "n_nodes": 0, "skipped": True})
            continue

        nodes = sorted(set(df["src_comp"].to_list()) | set(df["dst_comp"].to_list()))
        node2id = {v: i for i, v in enumerate(nodes)}
        n_nodes = len(nodes)

        edges_int = set()
        for u, v in zip(df["src_comp"].to_list(), df["dst_comp"].to_list()):
            a, b = node2id[u], node2id[v]
            if a > b: a, b = b, a
            edges_int.add((a, b))
        edges_int = list(edges_int)

        g_nk = nk.Graph(n_nodes, directed=False)
        for a, b in edges_int:
            g_nk.addEdge(a, b)
        cd = nk.centrality.CoreDecomposition(g_nk)
        cd.run()
        shell_nk = cd.scores()

        g_ig = ig.Graph(n=n_nodes, edges=edges_int, directed=False)
        shell_ig = g_ig.coreness()

        assert len(shell_nk) == len(shell_ig) == n_nodes
        mismatches = [(v, shell_nk[v], shell_ig[v])
                      for v in range(n_nodes)
                      if int(shell_nk[v]) != int(shell_ig[v])]
        match_rate = (n_nodes - len(mismatches)) / n_nodes

        elapsed = time.time() - tic
        max_shell = max(int(s) for s in shell_nk) if shell_nk else 0

        verdict = "✅ MATCH" if not mismatches else f"❌ {len(mismatches)} MISMATCH"
        print(f"  W{i:02d} [t={t0}..{t1}] |V|={n_nodes:>6} |E|={len(edges_int):>7}"
              f"  max_shell={max_shell:>3}  {verdict}  ({elapsed:.2f}s)")

        if mismatches:
            overall_pass = False
            print(f"      First 5 mismatches: {mismatches[:5]}")

        results.append({
            "window": i, "t0": t0, "t1": t1,
            "n_edges_raw": n_edges_raw,
            "n_edges_unique": len(edges_int),
            "n_nodes": n_nodes,
            "max_shell": max_shell,
            "match_rate": match_rate,
            "mismatches": len(mismatches),
            "elapsed_sec": round(elapsed, 3),
        })

    summary = {
        "n_windows_evaluated": sum(1 for r in results if not r.get("skipped")),
        "n_windows_skipped": sum(1 for r in results if r.get("skipped")),
        "overall_pass": overall_pass,
        "min_match_rate": min((r["match_rate"] for r in results if "match_rate" in r), default=None),
        "total_nodes_compared": sum(r.get("n_nodes", 0) for r in results),
        "total_mismatches": sum(r.get("mismatches", 0) for r in results),
        "config": {"n_windows": args.n_windows, "window_sec": args.window_sec, "seed": args.seed},
    }
    print()
    print(f"=== Block 1 Stage 1 Summary ===")
    print(json.dumps(summary, indent=2))

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "windows": results}, f, indent=2)
    print(f"Wrote {args.out}")

    if not overall_pass:
        print("\n⚠️  Oracle mismatch — NetworKit and igraph disagree. Investigate before proceeding.")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
