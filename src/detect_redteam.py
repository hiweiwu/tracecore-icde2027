# RQ3: raw-score detection on the red-team window
from __future__ import annotations
import argparse, json, sys, time, bisect
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "/private/workspace/icde_flow/code")
from tracecore_mvp import TraceCoreMVP

def load_redteam(path: str) -> list[tuple[int, str, str, str]]:
    df = pl.read_csv(path, has_header=False, new_columns=["time", "user", "src", "dst"])
    return list(zip(df["time"].to_list(), df["user"].to_list(),
                    df["src"].to_list(), df["dst"].to_list()))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mvp", required=True, help="TraceCore MVP pkl path")
    p.add_argument("--redteam", default="/private/workspace/icde_flow/data/lanl_2015/redteam.txt.gz")
    p.add_argument("--out-dir", default="/private/workspace/icde_flow/results/detect_redteam")
    p.add_argument("--ascent-windows", default="12,36,144",
                   help="comma-separated epoch counts for ascent features (5min epochs: 12=1h, 36=3h, 144=12h)")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ascent_windows = [int(w) for w in args.ascent_windows.split(",")]

    t_load = time.perf_counter()
    print(f"[load] TraceCore MVP from {args.mvp}")
    tc = TraceCoreMVP.load(args.mvp)
    print(f"[load] {len(tc.shells)} epochs, epoch_sec={tc.epoch_sec}, "
          f"{(time.perf_counter()-t_load)*1000:.0f} ms")

    rt = load_redteam(args.redteam)
    attackers = sorted({src for _, _, src, _ in rt})
    print(f"[redteam] {len(rt)} events, {len(attackers)} unique attacker hosts: {attackers}")

    epoch_ids = sorted(tc.shells.keys())
    eid_min, eid_max = min(epoch_ids), max(epoch_ids)

    print("[1/6] per-attacker trajectories")
    for host in attackers:
        rows = []
        for eid in epoch_ids:
            shell = tc.shells[eid].get(host)
            rows.append({"epoch_id": eid, "time_sec": eid * tc.epoch_sec,
                         "k_shell": shell if shell is not None else 0,
                         "present": shell is not None})
        pl.DataFrame(rows).write_csv(out / f"trajectory_{host}.csv")
        present = sum(1 for r in rows if r["present"])
        max_shell = max((r["k_shell"] for r in rows if r["present"]), default=0)
        print(f"    {host}: present in {present}/{len(rows)} epochs, max_shell={max_shell}")

    print("[2/6] per-event analysis")
    event_rows = []
    for t, user, src, dst in rt:
        eid = tc.epoch_of(t)
        shell_dict = tc.shells.get(eid, {})
        src_shell = shell_dict.get(src, 0)
        dst_shell = shell_dict.get(dst, 0)
        all_vals = sorted(shell_dict.values()) if shell_dict else []
        if all_vals:

            rank = bisect.bisect_left(all_vals, src_shell) / len(all_vals)
            epoch_max = all_vals[-1]
        else:
            rank = None
            epoch_max = 0
        event_rows.append({
            "time": t, "user": user, "src": src, "dst": dst, "epoch_id": eid,
            "src_in_index": src in shell_dict, "src_shell": src_shell,
            "dst_in_index": dst in shell_dict, "dst_shell": dst_shell,
            "src_percentile_rank": rank,
            "epoch_n_hosts": len(all_vals), "epoch_max_shell": epoch_max,
        })
    pl.DataFrame(event_rows).write_csv(out / "event_analysis.csv")

    print("    summary:")
    for host in attackers:
        host_events = [r for r in event_rows if r["src"] == host]
        if not host_events: continue
        in_index = sum(1 for r in host_events if r["src_in_index"])
        avg_shell = np.mean([r["src_shell"] for r in host_events if r["src_in_index"]] or [0])
        avg_rank = np.mean([r["src_percentile_rank"] for r in host_events if r["src_percentile_rank"] is not None] or [0])
        max_shell = max((r["src_shell"] for r in host_events), default=0)
        print(f"      {host}: {len(host_events)} events, "
              f"present_in_epoch={in_index}, avg_src_shell={avg_shell:.2f}, "
              f"avg_percentile={avg_rank:.3f}, max_src_shell={max_shell}")

    print("[3/6] building per-host time series (for ascent features)")
    host_ts: dict[str, dict[int, int]] = defaultdict(dict)
    for eid, shell_dict in tc.shells.items():
        for host, shell in shell_dict.items():
            host_ts[host][eid] = shell
    print(f"    {len(host_ts)} unique hosts across all epochs")

    print("[4/6] scoring every (host, epoch) pair")
    redteam_pos: set[tuple[str, int]] = set()
    for t, _, src, _ in rt:
        redteam_pos.add((src, tc.epoch_of(t)))
    print(f"    {len(redteam_pos)} unique positive (host, epoch) pairs")

    score_rows = []
    for eid in epoch_ids:
        shell_dict = tc.shells[eid]
        for host, shell in shell_dict.items():
            ts = host_ts[host]
            features = {"host": host, "epoch_id": eid, "k_shell": shell}
            for w in ascent_windows:
                prev = ts.get(eid - w, 0)
                features[f"ascent_{w}"] = shell - prev
            features["is_positive"] = (host, eid) in redteam_pos
            score_rows.append(features)
    print(f"    scored {len(score_rows):,} (host, epoch) pairs")

    score_df = pl.DataFrame(score_rows)
    score_df.write_parquet(out / "all_scores.parquet", compression="zstd")
    n_pos = int(score_df["is_positive"].sum())
    n_neg = len(score_df) - n_pos
    print(f"    positives: {n_pos}, negatives: {n_neg:,}")

    print("[5/6] operating-point curves")
    methods = ["k_shell"] + [f"ascent_{w}" for w in ascent_windows]
    fp_targets = [1e-4, 5e-4, 1e-3, 1e-2]
    topK_list = [10, 50, 100, 500, 1000]

    op_results = {}
    for m in methods:

        sorted_df = score_df.sort(m, descending=True)
        scores_arr = sorted_df[m].to_numpy()
        pos_arr = sorted_df["is_positive"].to_numpy()

        cum_tp = np.cumsum(pos_arr).astype(np.int64)
        cum_fp = np.arange(1, len(pos_arr) + 1) - cum_tp

        recalls = {}
        for fp_t in fp_targets:
            max_fp = int(fp_t * n_neg)
            idx = np.searchsorted(cum_fp, max_fp, side="right") - 1
            if idx < 0:
                recalls[f"recall_at_FP_{fp_t}"] = 0.0
            else:
                recalls[f"recall_at_FP_{fp_t}"] = float(cum_tp[idx]) / max(1, n_pos)

        precisions = {}
        for K in topK_list:
            K_actual = min(K, len(pos_arr))
            precisions[f"precision_at_K_{K}"] = float(cum_tp[K_actual - 1]) / K_actual

        op_results[m] = {**recalls, **precisions}
        print(f"    {m}:")
        for fp_t in fp_targets:
            print(f"        recall@FP={fp_t:.0e}: {recalls[f'recall_at_FP_{fp_t}']:.3f}")
        for K in topK_list:
            print(f"        precision@K={K:>4}: {precisions[f'precision_at_K_{K}']:.3f}")

    print("[6/6] writing report")
    report = {
        "mvp": args.mvp,
        "n_epochs": len(tc.shells),
        "epoch_sec": tc.epoch_sec,
        "epoch_range": [eid_min, eid_max],
        "n_redteam_events": len(rt),
        "attackers": attackers,
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "ascent_windows": ascent_windows,
        "operating_points": op_results,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"    wrote {out / 'report.json'}")
    print(f"    wrote {out / 'all_scores.parquet'} ({(out / 'all_scores.parquet').stat().st_size/1e6:.1f} MB)")
    print(f"    wrote {len(attackers)} trajectory CSVs and event_analysis.csv")

if __name__ == "__main__":
    main()
