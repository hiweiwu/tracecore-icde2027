# RQ1 stage 2: core_threshold / core_component / shell_diff set checks vs oracles (Table IV)
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path

import polars as pl
import networkit as nk
import igraph as ig

PARQUET = "/private/workspace/icde_flow/data/lanl_2015/flows_minimal.parquet"

def build_window_graph(t0: int, t1: int):
    df = (pl.scan_parquet(PARQUET)
            .filter((pl.col("time") >= t0) & (pl.col("time") < t1))
            .select(["src_comp", "dst_comp"])
            .filter(pl.col("src_comp") != pl.col("dst_comp"))
            .unique()
            .collect())
    if df.height == 0:
        return [], [], {}
    nodes = sorted(set(df["src_comp"].to_list()) | set(df["dst_comp"].to_list()))
    node2id = {v: i for i, v in enumerate(nodes)}
    edges = set()
    for u, v in zip(df["src_comp"].to_list(), df["dst_comp"].to_list()):
        a, b = node2id[u], node2id[v]
        if a > b: a, b = b, a
        edges.add((a, b))
    return nodes, list(edges), node2id

def kshell_nk(n_nodes: int, edges: list) -> list[int]:
    g = nk.Graph(n_nodes, directed=False)
    for a, b in edges:
        g.addEdge(a, b)
    cd = nk.centrality.CoreDecomposition(g)
    cd.run()
    return [int(s) for s in cd.scores()]

def kshell_ig(n_nodes: int, edges: list) -> list[int]:
    g = ig.Graph(n=n_nodes, edges=edges, directed=False)
    return list(g.coreness())

def core_threshold(shell: list[int], k: int) -> set[int]:
    return {v for v, s in enumerate(shell) if s >= k}

def core_component_nk(n_nodes: int, edges: list, shell: list[int],
                      seed: int, k: int) -> set[int]:
    if shell[seed] < k:
        return set()
    keep = {v for v, s in enumerate(shell) if s >= k}

    adj = [[] for _ in range(n_nodes)]
    for a, b in edges:
        if a in keep and b in keep:
            adj[a].append(b)
            adj[b].append(a)
    seen = {seed}
    stack = [seed]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen

def core_component_ig(n_nodes: int, edges: list, shell: list[int],
                      seed: int, k: int) -> set[int]:
    if shell[seed] < k:
        return set()
    keep = [v for v, s in enumerate(shell) if s >= k]
    keep_set = set(keep)
    sub_edges = [(a, b) for a, b in edges if a in keep_set and b in keep_set]

    remap = {v: i for i, v in enumerate(keep)}
    sub_edges_remapped = [(remap[a], remap[b]) for a, b in sub_edges]
    g = ig.Graph(n=len(keep), edges=sub_edges_remapped, directed=False)
    components = g.connected_components()
    if seed not in remap:
        return set()
    seed_remap = remap[seed]
    comp_id = components.membership[seed_remap]
    return {keep[i] for i, c in enumerate(components.membership) if c == comp_id}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-windows", type=int, default=20)
    p.add_argument("--window-sec", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-list", default="2,3,4,6,8",
                   help="comma-separated k values for core_threshold / core_component")
    p.add_argument("--n-seeds-per-window", type=int, default=5,
                   help="hosts to sample per window for core_component check")
    p.add_argument("--out", default="/private/workspace/icde_flow/results/block1_stage2.json")
    p.add_argument("--t-max", type=int, default=3_126_928)
    args = p.parse_args()

    random.seed(args.seed)
    k_list = [int(k) for k in args.k_list.split(",")]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"Block 1 Stage 2 — n_windows={args.n_windows}, k_list={k_list}, "
          f"n_seeds_per_window={args.n_seeds_per_window}")

    starts = sorted(random.sample(range(0, min(args.t_max, 1_400_000) - args.window_sec),
                                  args.n_windows))

    results_per_window = []
    overall_ok = True

    for i, t0 in enumerate(starts):
        t1 = t0 + args.window_sec
        t2 = t1 + args.window_sec
        tic = time.time()

        nodes_a, edges_a, _ = build_window_graph(t0, t1)
        if not nodes_a:
            print(f"  W{i:02d} [t={t0}..{t1}] EMPTY — skip")
            continue
        nodes_b, edges_b, _ = build_window_graph(t1, t2)

        n_a = len(nodes_a)
        shell_nk_a = kshell_nk(n_a, edges_a)
        shell_ig_a = kshell_ig(n_a, edges_a)

        pc_mismatches = sum(1 for v in range(n_a) if shell_nk_a[v] != shell_ig_a[v])

        ct_results = {}
        for k in k_list:
            s_nk = core_threshold(shell_nk_a, k)
            s_ig = core_threshold(shell_ig_a, k)
            ct_results[k] = {
                "n_nk": len(s_nk), "n_ig": len(s_ig),
                "match": s_nk == s_ig,
                "sym_diff": len(s_nk.symmetric_difference(s_ig)),
            }

        eligible = [v for v in range(n_a) if shell_nk_a[v] >= min(k_list)]
        seeds_to_test = random.sample(eligible, min(args.n_seeds_per_window, len(eligible))) if eligible else []
        cc_results = []
        for seed in seeds_to_test:
            k = max(min(k_list), shell_nk_a[seed] - 1)
            c_nk = core_component_nk(n_a, edges_a, shell_nk_a, seed, k)
            c_ig = core_component_ig(n_a, edges_a, shell_ig_a, seed, k)
            cc_results.append({
                "seed": seed, "k": k,
                "size_nk": len(c_nk), "size_ig": len(c_ig),
                "match": c_nk == c_ig,
            })

        if nodes_b:
            n_b = len(nodes_b)
            shell_nk_b = kshell_nk(n_b, edges_b)
            shell_ig_b = kshell_ig(n_b, edges_b)

            id2name_a = {i: nodes_a[i] for i in range(n_a)}
            id2name_b = {i: nodes_b[i] for i in range(n_b)}

            def hosts_at(shell, id2name, k):
                return {id2name[v] for v, s in enumerate(shell) if s >= k}
            sd_results = {}
            for k in k_list:
                a_nk = hosts_at(shell_nk_a, id2name_a, k)
                a_ig = hosts_at(shell_ig_a, id2name_a, k)
                b_nk = hosts_at(shell_nk_b, id2name_b, k)
                b_ig = hosts_at(shell_ig_b, id2name_b, k)
                entrants_nk = b_nk - a_nk
                entrants_ig = b_ig - a_ig
                exits_nk = a_nk - b_nk
                exits_ig = a_ig - b_ig
                sd_results[k] = {
                    "entrants_match": entrants_nk == entrants_ig,
                    "exits_match": exits_nk == exits_ig,
                    "n_entrants": len(entrants_nk),
                    "n_exits": len(exits_nk),
                }
        else:
            sd_results = {"skipped_no_window_B": True}

        elapsed = time.time() - tic

        ok_pc = pc_mismatches == 0
        ok_ct = all(v["match"] for v in ct_results.values())
        ok_cc = all(v["match"] for v in cc_results)
        ok_sd = (isinstance(sd_results, dict)
                 and sd_results.get("skipped_no_window_B")
                 or all(v["entrants_match"] and v["exits_match"]
                        for k, v in sd_results.items() if isinstance(v, dict) and "entrants_match" in v))
        window_ok = ok_pc and ok_ct and ok_cc and ok_sd
        if not window_ok:
            overall_ok = False
        verdict = "✅" if window_ok else "❌"
        print(f"  W{i:02d} |V|={n_a:>5} |E|={len(edges_a):>6} "
              f"pc={'✅' if ok_pc else '❌'} "
              f"ct={'✅' if ok_ct else '❌'} "
              f"cc={'✅' if ok_cc else '❌'}({len(cc_results)}) "
              f"sd={'✅' if ok_sd else '❌'} "
              f"{verdict}  ({elapsed:.2f}s)")

        results_per_window.append({
            "window": i, "t0": t0, "t1": t1,
            "n_nodes": n_a, "n_edges": len(edges_a),
            "max_shell": max(shell_nk_a) if shell_nk_a else 0,
            "pc_mismatches": pc_mismatches,
            "ct": ct_results,
            "cc": cc_results,
            "sd": sd_results,
            "ok": window_ok,
            "elapsed_sec": round(elapsed, 3),
        })

    summary = {
        "n_windows_evaluated": len(results_per_window),
        "overall_pass": overall_ok,
        "config": vars(args),
    }
    print()
    print(f"=== Block 1 Stage 2 Summary ===")
    print(json.dumps(summary, indent=2))

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "windows": results_per_window}, f, indent=2)
    print(f"Wrote {args.out}")
    return 0 if overall_ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
