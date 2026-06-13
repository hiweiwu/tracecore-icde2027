# RQ2: query throughput vs TCD/OTCD/iPHC on the 1-hour slice (Table VII)
from __future__ import annotations
import argparse, json, random, time, statistics
from pathlib import Path

import numpy as np

from tracecore_mvp import TraceCoreMVP

def stats(name: str, ns: list[int]):
    if not ns:
        return {}
    a = np.array(ns)
    return {
        "n": len(a),
        "mean_us": float(a.mean()) / 1000.0,
        "p50_us": float(np.percentile(a, 50)) / 1000.0,
        "p95_us": float(np.percentile(a, 95)) / 1000.0,
        "p99_us": float(np.percentile(a, 99)) / 1000.0,
        "max_us": float(a.max()) / 1000.0,
        "sum_ms": float(a.sum()) / 1e6,
        "qps_alg": float(len(a) / (a.sum() / 1e9)) if a.sum() else 0.0,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mvp", default="/private/workspace/icde_flow/bench/slice1h/tracecore_mvp.pkl")
    p.add_argument("--queries", default="/private/workspace/icde_flow/bench/slice1h/queries.txt",
                   help="TCD-style queries.txt: ts<TAB>te<TAB>k")
    p.add_argument("--n-point", type=int, default=100, help="number of point_coreness queries")
    p.add_argument("--n-threshold", type=int, default=100, help="number of core_threshold queries")
    p.add_argument("--n-warmup", type=int, default=10, help="warmup queries before timing")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/private/workspace/icde_flow/bench/slice1h/tracecore_mvp_bench.json")
    args = p.parse_args()

    random.seed(args.seed)

    print(f"[load] {args.mvp}")
    t0 = time.perf_counter()
    tc = TraceCoreMVP.load(args.mvp)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"[load] {load_ms:.1f} ms; {len(tc.shells)} epochs; epoch_sec={tc.epoch_sec}")

    all_hosts_per_epoch = {eid: list(shell.keys()) for eid, shell in tc.shells.items() if shell}
    eligible_epochs = list(all_hosts_per_epoch.keys())
    e_min, e_max = min(eligible_epochs), max(eligible_epochs)
    t_min = e_min * tc.epoch_sec
    t_max = (e_max + 1) * tc.epoch_sec - 1
    k_pool = [2, 3, 4, 5, 6]

    point_queries = []
    for _ in range(args.n_point):
        eid = random.choice(eligible_epochs)
        host = random.choice(all_hosts_per_epoch[eid])
        t = eid * tc.epoch_sec + random.randint(0, tc.epoch_sec - 1)
        point_queries.append((host, t))

    print(f"[A] point_coreness: {len(point_queries)} queries, "
          f"warmup={args.n_warmup}")
    for host, t in point_queries[:args.n_warmup]:
        tc.point_coreness(host, t)
    ns_A = []
    for host, t in point_queries:
        s = time.perf_counter_ns()
        tc.point_coreness(host, t)
        ns_A.append(time.perf_counter_ns() - s)
    stat_A = stats("point_coreness", ns_A)
    print(f"     mean={stat_A['mean_us']:.3f} us, p50={stat_A['p50_us']:.3f}, "
          f"p95={stat_A['p95_us']:.3f}, p99={stat_A['p99_us']:.3f}, "
          f"qps_alg={stat_A['qps_alg']:.0f}")

    thresh_queries = []
    for _ in range(args.n_threshold):
        eid = random.choice(eligible_epochs)
        t = eid * tc.epoch_sec + random.randint(0, tc.epoch_sec - 1)
        k = random.choice(k_pool)
        thresh_queries.append((k, t))

    print(f"[B] core_threshold: {len(thresh_queries)} queries")
    for k, t in thresh_queries[:args.n_warmup]:
        tc.core_threshold(k, t)
    ns_B = []
    out_sizes_B = []
    for k, t in thresh_queries:
        s = time.perf_counter_ns()
        result = tc.core_threshold(k, t)
        ns_B.append(time.perf_counter_ns() - s)
        out_sizes_B.append(len(result))
    stat_B = stats("core_threshold", ns_B)
    stat_B["mean_output_size"] = float(np.mean(out_sizes_B))
    print(f"     mean={stat_B['mean_us']:.2f} us, p50={stat_B['p50_us']:.2f}, "
          f"p95={stat_B['p95_us']:.2f}, p99={stat_B['p99_us']:.2f}, "
          f"qps_alg={stat_B['qps_alg']:.0f}, mean_out={stat_B['mean_output_size']:.1f}")

    temp_queries = []
    with open(args.queries) as f:
        for line in f:
            ts, te, k = line.strip().split("\t")
            temp_queries.append((int(ts), int(te), int(k)))
    print(f"[C] temporal_k_core (B2-comparable): {len(temp_queries)} queries from {args.queries}")
    for ts, te, k in temp_queries[:args.n_warmup]:
        tc.temporal_k_core(ts, te, k)
    ns_C = []
    out_sizes_C = []
    for ts, te, k in temp_queries:
        s = time.perf_counter_ns()
        result = tc.temporal_k_core(ts, te, k)
        ns_C.append(time.perf_counter_ns() - s)
        out_sizes_C.append(len(result))
    stat_C = stats("temporal_k_core", ns_C)
    stat_C["mean_output_size"] = float(np.mean(out_sizes_C))
    print(f"     mean={stat_C['mean_us']:.2f} us, p50={stat_C['p50_us']:.2f}, "
          f"p95={stat_C['p95_us']:.2f}, p99={stat_C['p99_us']:.2f}, "
          f"qps_alg={stat_C['qps_alg']:.0f}, mean_out={stat_C['mean_output_size']:.1f}")

    report = {
        "mvp": args.mvp,
        "n_epochs": len(tc.shells),
        "epoch_sec": tc.epoch_sec,
        "load_ms": load_ms,
        "point_coreness": stat_A,
        "core_threshold": stat_B,
        "temporal_k_core": stat_C,
        "config": vars(args),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[save] {args.out}")

    print()
    print("=== TraceCore MVP vs B2 TCD (same 100 queries, alg-only) ===")
    print(f"  TCD:      0.18 us mean, 0.16 p50, 0.43 p95, 0.53 p99,  5.5M qps")
    print(f"  TraceC.C: {stat_C['mean_us']:.2f} us mean, {stat_C['p50_us']:.2f} p50, "
          f"{stat_C['p95_us']:.2f} p95, {stat_C['p99_us']:.2f} p99, "
          f"{stat_C['qps_alg']/1e6:.2f}M qps")
    speedup = 0.18 / stat_C["mean_us"] if stat_C["mean_us"] else 0
    print(f"  speedup (TraceCore over TCD, mean): {speedup:.2f}x")

if __name__ == "__main__":
    main()
