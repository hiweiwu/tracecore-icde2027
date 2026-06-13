# TraceCore engine: sliding-window k-shell maintenance + CoreDelta/ShellIndex/HostTimeline + query primitives (Sec. III)
from __future__ import annotations
import argparse, json, pickle, time
from pathlib import Path

import polars as pl
import networkit as nk

class TraceCoreMVP:
    def __init__(self, parquet: str, epoch_sec: int = 300):
        self.parquet = parquet
        self.epoch_sec = epoch_sec
        self.shells: dict[int, dict[str, int]] = {}
        self.build_stats: dict[int, dict] = {}

    def epoch_of(self, t: int) -> int:
        return t // self.epoch_sec

    def build(self, t_min: int, t_max: int, verbose: bool = True):
        e_first, e_last = self.epoch_of(t_min), self.epoch_of(t_max - 1)
        if verbose:
            print(f"[build] epochs {e_first}..{e_last} ({e_last-e_first+1} total), "
                  f"epoch_sec={self.epoch_sec}")
        t_total = time.perf_counter()

        for eid in range(e_first, e_last + 1):
            e_t0 = eid * self.epoch_sec
            e_t1 = e_t0 + self.epoch_sec
            tic = time.perf_counter()

            df = (pl.scan_parquet(self.parquet)
                    .filter((pl.col("time") >= e_t0) & (pl.col("time") < e_t1))
                    .filter(pl.col("src_comp") != pl.col("dst_comp"))
                    .select(["src_comp", "dst_comp"])
                    .unique()
                    .collect())

            if df.height == 0:
                self.shells[eid] = {}
                self.build_stats[eid] = {"n_nodes": 0, "n_edges": 0,
                                          "build_ms": (time.perf_counter()-tic)*1000}
                continue

            nodes = sorted(set(df["src_comp"].to_list()) | set(df["dst_comp"].to_list()))
            node2id = {v: i for i, v in enumerate(nodes)}

            edges = set()
            for u, v in zip(df["src_comp"].to_list(), df["dst_comp"].to_list()):
                a, b = node2id[u], node2id[v]
                if a > b: a, b = b, a
                edges.add((a, b))

            g = nk.Graph(len(nodes), directed=False)
            for a, b in edges:
                g.addEdge(a, b)
            cd = nk.centrality.CoreDecomposition(g)
            cd.run()
            scores = cd.scores()

            shell = {nodes[i]: int(scores[i]) for i in range(len(nodes))}
            self.shells[eid] = shell
            self.build_stats[eid] = {
                "n_nodes": len(nodes),
                "n_edges": len(edges),
                "max_shell": max(int(s) for s in scores),
                "build_ms": (time.perf_counter() - tic) * 1000,
            }
            if verbose and eid % 12 == 0:
                bs = self.build_stats[eid]
                print(f"  epoch {eid} ({e_t0}..{e_t1}): |V|={bs['n_nodes']:>5} "
                      f"|E|={bs['n_edges']:>6} max_shell={bs['max_shell']:>2} "
                      f"build={bs['build_ms']:.1f}ms")

        total_ms = (time.perf_counter() - t_total) * 1000
        if verbose:
            print(f"[build] done. {len(self.shells)} epochs in {total_ms:.0f} ms "
                  f"({total_ms/max(1,len(self.shells)):.1f} ms/epoch avg)")
        return total_ms

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"epoch_sec": self.epoch_sec,
                         "shells": self.shells,
                         "build_stats": self.build_stats}, f, protocol=4)

    @classmethod
    def load(cls, path: str, parquet: str = ""):
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(parquet, d["epoch_sec"])
        obj.shells = d["shells"]
        obj.build_stats = d["build_stats"]
        return obj

    def point_coreness(self, host: str, t: int) -> int:
        return self.shells.get(self.epoch_of(t), {}).get(host, 0)

    def core_threshold(self, k: int, t: int) -> set:
        return {h for h, s in self.shells.get(self.epoch_of(t), {}).items() if s >= k}

    def temporal_k_core(self, ts: int, te: int, k: int) -> set:
        result = set()
        for eid in range(self.epoch_of(ts), self.epoch_of(te) + 1):
            for h, s in self.shells.get(eid, {}).items():
                if s >= k:
                    result.add(h)
        return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="/private/workspace/icde_flow/data/lanl_2015/flows_minimal.parquet")
    p.add_argument("--t0", type=int, default=100_000)
    p.add_argument("--t1", type=int, default=103_600)
    p.add_argument("--epoch-sec", type=int, default=300)
    p.add_argument("--out", default="/private/workspace/icde_flow/bench/slice1h/tracecore_mvp.pkl")
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tc = TraceCoreMVP(args.parquet, args.epoch_sec)
    build_ms = tc.build(args.t0, args.t1, verbose=True)
    tc.save(args.out)
    pkl_mb = Path(args.out).stat().st_size / 1e6
    print(f"[build] saved {args.out} ({pkl_mb:.2f} MB)")
    print(json.dumps({"build_ms": build_ms, "n_epochs": len(tc.shells),
                      "pkl_mb": pkl_mb, "t0": args.t0, "t1": args.t1,
                      "epoch_sec": args.epoch_sec}, indent=2))

if __name__ == "__main__":
    main()
