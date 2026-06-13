# RQ2: ingest throughput + query-under-ingest + window sweep (Sec. V-C)
from __future__ import annotations
import argparse, json, pickle, random, threading, time
from pathlib import Path

import numpy as np
import polars as pl
import networkit as nk
import psutil

PARQUET = "/private/workspace/icde_flow/data/lanl_2015/flows_minimal.parquet"

def pct(arr, q):
    return float(np.percentile(np.asarray(arr, dtype=np.float64), q)) if len(arr) else 0.0

class QueryThread(threading.Thread):
    def __init__(self, shells: dict, epoch_sec: int, interval_s: float = 0.01):
        super().__init__(daemon=True)
        self.shells = shells
        self.epoch_sec = epoch_sec
        self.interval = interval_s
        self.latencies_us: list[float] = []
        self.n_queries = 0
        self.stop_flag = False
        self.rng = random.Random(42)

    def run(self):
        while not self.stop_flag:
            eids = list(self.shells.keys())
            if eids:
                eid = self.rng.choice(eids)
                shell = self.shells.get(eid, {})
                if shell:
                    host = next(iter(shell))
                    t = eid * self.epoch_sec
                    tic = time.perf_counter()
                    _ = self.shells.get(t // self.epoch_sec, {}).get(host, 0)
                    self.latencies_us.append((time.perf_counter() - tic) * 1e6)
                    self.n_queries += 1
            time.sleep(self.interval)

def build_instrumented(epoch_sec: int, t0: int, t1: int, with_queries: bool,
                       out_pkl: str | None):
    proc = psutil.Process()
    e_first, e_last = t0 // epoch_sec, (t1 - 1) // epoch_sec
    shells: dict[int, dict[str, int]] = {}
    epoch_ms: list[float] = []
    flow_rows_total = 0
    rss_mb: list[float] = []

    qt = QueryThread(shells, epoch_sec) if with_queries else None
    if qt:
        qt.start()

    t_start = time.perf_counter()
    for eid in range(e_first, e_last + 1):
        a, b = eid * epoch_sec, (eid + 1) * epoch_sec
        tic = time.perf_counter()
        df = (pl.scan_parquet(PARQUET)
                .filter((pl.col("time") >= a) & (pl.col("time") < b))
                .filter(pl.col("src_comp") != pl.col("dst_comp"))
                .select(["src_comp", "dst_comp"])
                .collect())
        flow_rows = df.height
        flow_rows_total += flow_rows
        if flow_rows:
            dfu = df.unique()
            nodes = sorted(set(dfu["src_comp"].to_list()) | set(dfu["dst_comp"].to_list()))
            node2id = {v: i for i, v in enumerate(nodes)}
            edges = set()
            for u, v in zip(dfu["src_comp"].to_list(), dfu["dst_comp"].to_list()):
                x, y = node2id[u], node2id[v]
                if x > y: x, y = y, x
                edges.add((x, y))
            g = nk.Graph(len(nodes), directed=False)
            for x, y in edges:
                g.addEdge(x, y)
            cd = nk.centrality.CoreDecomposition(g); cd.run()
            scores = cd.scores()
            shells[eid] = {nodes[i]: int(scores[i]) for i in range(len(nodes))}
        else:
            shells[eid] = {}
        epoch_ms.append((time.perf_counter() - tic) * 1000)
        if (eid - e_first) % 50 == 0:
            rss_mb.append(proc.memory_info().rss / 1e6)
    wall_sec = time.perf_counter() - t_start
    if qt:
        qt.stop_flag = True
        qt.join(timeout=2)
    rss_mb.append(proc.memory_info().rss / 1e6)

    result = {
        "epoch_sec": epoch_sec, "t0": t0, "t1": t1,
        "n_epochs": len(shells),
        "wall_sec": round(wall_sec, 1),
        "flow_rows_total": flow_rows_total,
        "flows_per_sec_sustained": round(flow_rows_total / wall_sec, 0),
        "epoch_latency_ms": {
            "p50": round(pct(epoch_ms, 50), 1), "p90": round(pct(epoch_ms, 90), 1),
            "p95": round(pct(epoch_ms, 95), 1), "p99": round(pct(epoch_ms, 99), 1),
            "max": round(max(epoch_ms), 1) if epoch_ms else 0,
            "mean": round(float(np.mean(epoch_ms)), 1) if epoch_ms else 0,
        },
        "rss_mb": {"start": round(rss_mb[0], 0), "peak": round(max(rss_mb), 0),
                   "end": round(rss_mb[-1], 0)},
    }
    if qt:
        result["query_under_ingest"] = {
            "n_queries": qt.n_queries,
            "latency_us": {"p50": round(pct(qt.latencies_us, 50), 2),
                           "p95": round(pct(qt.latencies_us, 95), 2),
                           "p99": round(pct(qt.latencies_us, 99), 2)},
        }
    if out_pkl:
        with open(out_pkl, "wb") as f:
            pickle.dump({"epoch_sec": epoch_sec, "shells": shells, "build_stats": {}}, f, protocol=4)
        result["pkl_mb"] = round(Path(out_pkl).stat().st_size / 1e6, 1)
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mvp", default="/private/workspace/icde_flow/bench/lanl_redteam_range/tracecore_mvp.pkl",
                   help="existing 28-day pkl; used only to derive the epoch range")
    p.add_argument("--w-list", default="300,60,900")
    p.add_argument("--out", default="/private/workspace/icde_flow/results/streaming_systems.json")
    p.add_argument("--scratch", default="/private/workspace/icde_flow/bench/streaming_systems")
    args = p.parse_args()

    Path(args.scratch).mkdir(parents=True, exist_ok=True)

    print(f"[range] deriving epoch range from {args.mvp}")
    with open(args.mvp, "rb") as f:
        d = pickle.load(f)
    eids = sorted(d["shells"].keys())
    base_es = d["epoch_sec"]
    t0, t1 = eids[0] * base_es, (eids[-1] + 1) * base_es
    del d
    print(f"[range] t0={t0} t1={t1} ({(t1-t0)/86400:.1f} days)")

    report = {"t0": t0, "t1": t1, "days": round((t1 - t0) / 86400, 1), "runs": {}}
    for i, w in enumerate(int(x) for x in args.w_list.split(",")):
        with_q = (i == 0)
        out_pkl = f"{args.scratch}/shells_w{w}.pkl"
        print(f"\n=== build W={w}s (query_thread={with_q}) ===")
        r = build_instrumented(w, t0, t1, with_q, out_pkl)
        report["runs"][f"W{w}"] = r
        print(json.dumps(r, indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[save] {args.out}")

if __name__ == "__main__":
    main()
