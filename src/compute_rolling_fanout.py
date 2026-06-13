# Rolling distinct-neighbor counts at 1h/6h/24h/1wk horizons
from __future__ import annotations
import argparse, time
from collections import Counter, defaultdict, deque
from pathlib import Path

import polars as pl

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="/private/workspace/icde_flow/data/lanl_2015/flows_minimal.parquet")
    p.add_argument("--t0", type=int, default=150_000)
    p.add_argument("--t1", type=int, default=2_557_048)
    p.add_argument("--epoch-sec", type=int, default=300)
    p.add_argument("--windows", default="12,72,288,2016",
                   help="comma-separated epoch-count windows: 12=1h, 72=6h, 288=24h, 2016=1week")
    p.add_argument("--out", default="/private/workspace/icde_flow/bench/lanl_redteam_range/rolling_fanout.parquet")
    args = p.parse_args()

    windows = [int(w) for w in args.windows.split(",")]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] reading edges from {args.parquet} in [{args.t0}, {args.t1})")
    t0 = time.perf_counter()
    edges = (pl.scan_parquet(args.parquet)
               .filter((pl.col("time") >= args.t0) & (pl.col("time") < args.t1))
               .filter(pl.col("src_comp") != pl.col("dst_comp"))
               .with_columns((pl.col("time") // args.epoch_sec).cast(pl.Int64).alias("epoch_id"))
               .select(["epoch_id", "src_comp", "dst_comp"])
               .unique()
               .collect())
    print(f"[load] {edges.height:,} unique (src, dst, epoch) rows in {time.perf_counter()-t0:.1f}s")

    print("[expand] expanding to host→neighbor view (both directions)")
    a = edges.select([pl.col("epoch_id"), pl.col("src_comp").alias("host"), pl.col("dst_comp").alias("neighbor")])
    b = edges.select([pl.col("epoch_id"), pl.col("dst_comp").alias("host"), pl.col("src_comp").alias("neighbor")])
    he = pl.concat([a, b]).unique().sort(["host", "epoch_id"])
    print(f"[expand] {he.height:,} (host, epoch, neighbor) rows")

    print("[group] aggregating neighbors per (host, epoch)")
    g = (he.group_by(["host", "epoch_id"])
           .agg(pl.col("neighbor").alias("neighbors"))
           .sort(["host", "epoch_id"]))
    print(f"[group] {g.height:,} unique (host, epoch) cells")

    max_W = max(windows)
    print(f"[rolling] computing rolling distinct fanout for W = {windows}")
    t0 = time.perf_counter()

    out_rows = []
    cur_host = None
    counts: dict[int, Counter] = {W: Counter() for W in windows}
    window_deques: dict[int, deque] = {W: deque() for W in windows}

    rows = g.iter_rows(named=True)
    n_hosts_done = 0
    for row in rows:
        host = row["host"]
        epoch_id = row["epoch_id"]
        neighbors = set(row["neighbors"])

        if host != cur_host:
            cur_host = host
            for W in windows:
                counts[W].clear()
                window_deques[W].clear()
            n_hosts_done += 1
            if n_hosts_done % 1000 == 0:
                print(f"    progress: {n_hosts_done} hosts processed "
                      f"({(time.perf_counter()-t0):.0f}s, {len(out_rows)} rows)")

        feat = {"host": host, "epoch_id": epoch_id, "fanout_now": len(neighbors)}
        for W in windows:

            window_deques[W].append((epoch_id, neighbors))
            for n in neighbors:
                counts[W][n] += 1

            while window_deques[W] and window_deques[W][0][0] < epoch_id - W + 1:
                old_e, old_neighbors = window_deques[W].popleft()
                for n in old_neighbors:
                    counts[W][n] -= 1
                    if counts[W][n] == 0:
                        del counts[W][n]
            feat[f"rolling_fanout_{W}"] = len(counts[W])
        out_rows.append(feat)

    print(f"[rolling] {len(out_rows):,} rows in {time.perf_counter()-t0:.1f}s ({n_hosts_done} hosts)")

    out_df = pl.DataFrame(out_rows)
    out_df.write_parquet(args.out, compression="zstd")
    sz = Path(args.out).stat().st_size / 1e6
    print(f"[save] {args.out} ({sz:.1f} MB)")

    attackers = ["C17693", "C18025", "C19932", "C22409"]
    print("[stats] attacker rolling-fanout summaries (max over campaign):")
    for h in attackers:
        sub = out_df.filter(pl.col("host") == h)
        if sub.height == 0:
            print(f"    {h}: NOT PRESENT")
            continue
        line = [h, f"n_epochs={sub.height}"]
        for W in windows:
            mx = int(sub[f"rolling_fanout_{W}"].max())
            md = int(sub[f"rolling_fanout_{W}"].median())
            line.append(f"W={W}: median={md}, max={mx}")
        print("    " + " | ".join(line))

if __name__ == "__main__":
    main()
