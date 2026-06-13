# RQ3: detection scoring v2 (raw + residual rankings)
from __future__ import annotations
import argparse, json, sys, time, bisect
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "/private/workspace/icde_flow/code")
from tracecore_mvp import TraceCoreMVP

def load_redteam(path):
    df = pl.read_csv(path, has_header=False, new_columns=["time", "user", "src", "dst"])
    return list(zip(df["time"].to_list(), df["user"].to_list(),
                    df["src"].to_list(), df["dst"].to_list()))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mvp", required=True)
    p.add_argument("--rolling-fanout", required=True,
                   help="parquet from compute_rolling_fanout.py")
    p.add_argument("--redteam", default="/private/workspace/icde_flow/data/lanl_2015/redteam.txt.gz")
    p.add_argument("--out-dir", default="/private/workspace/icde_flow/results/detect_redteam_v2")
    p.add_argument("--ascent-windows", default="12,36,144")
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ascent_windows = [int(w) for w in args.ascent_windows.split(",")]

    print(f"[load] TraceCore MVP from {args.mvp}")
    tc = TraceCoreMVP.load(args.mvp)
    print(f"[load] {len(tc.shells)} epochs, epoch_sec={tc.epoch_sec}")

    rt = load_redteam(args.redteam)
    attackers = sorted({src for _, _, src, _ in rt})
    print(f"[redteam] {len(rt)} events, attackers={attackers}")

    redteam_pos = set()
    for t, _, src, _ in rt:
        redteam_pos.add((src, tc.epoch_of(t)))
    print(f"[redteam] {len(redteam_pos)} unique positive (host, epoch) pairs")

    print("[shells] building per-host k-shell time series")
    host_ts: dict[str, dict[int, int]] = defaultdict(dict)
    for eid, shell_dict in tc.shells.items():
        for host, shell in shell_dict.items():
            host_ts[host][eid] = shell

    epoch_ids = sorted(tc.shells.keys())

    print("[score] building k-shell + ascent feature rows")
    t0 = time.perf_counter()
    rows = []
    for eid in epoch_ids:
        shell_dict = tc.shells[eid]
        for host, shell in shell_dict.items():
            ts = host_ts[host]
            r = {"host": host, "epoch_id": eid, "k_shell": shell}
            for w in ascent_windows:
                r[f"ascent_{w}"] = shell - ts.get(eid - w, 0)
            rows.append(r)
    feat_df = pl.DataFrame(rows)
    print(f"[score] {feat_df.height:,} k-shell feature rows in {time.perf_counter()-t0:.1f}s")

    print(f"[join] loading {args.rolling_fanout}")
    rf = pl.read_parquet(args.rolling_fanout)
    print(f"[join] {rf.height:,} rolling-fanout rows; columns: {rf.columns}")
    joined = feat_df.join(rf, on=["host", "epoch_id"], how="left").fill_null(0)
    print(f"[join] joined: {joined.height:,} rows; cols: {joined.columns}")

    joined = joined.with_columns(
        pl.struct(["host", "epoch_id"]).map_elements(
            lambda s: (s["host"], s["epoch_id"]) in redteam_pos, return_dtype=pl.Boolean
        ).alias("is_positive")
    )
    n_pos = int(joined["is_positive"].sum())
    n_neg = joined.height - n_pos
    print(f"[mark]   positives={n_pos}, negatives={n_neg:,}")

    joined.write_parquet(out / "all_scores_v2.parquet", compression="zstd")

    print("[op]   operating-point curves")
    fp_targets = [1e-4, 5e-4, 1e-3, 1e-2, 5e-2]
    topK_list = [10, 50, 100, 500, 1000, 5000]
    methods = ["k_shell"] + [f"ascent_{w}" for w in ascent_windows] + \
              ["fanout_now", "rolling_fanout_12", "rolling_fanout_72",
               "rolling_fanout_288", "rolling_fanout_2016"]

    op_results = {}
    for m in methods:
        if m not in joined.columns:
            continue
        sorted_df = joined.sort(m, descending=True)
        pos_arr = sorted_df["is_positive"].to_numpy()
        cum_tp = np.cumsum(pos_arr).astype(np.int64)
        cum_fp = np.arange(1, len(pos_arr) + 1) - cum_tp

        recalls = {}
        for fp_t in fp_targets:
            max_fp = int(fp_t * n_neg)
            idx = np.searchsorted(cum_fp, max_fp, side="right") - 1
            recalls[f"recall_at_FP_{fp_t}"] = float(cum_tp[max(0, idx)]) / max(1, n_pos) if idx >= 0 else 0.0

        precisions = {}
        for K in topK_list:
            K_actual = min(K, len(pos_arr))
            precisions[f"precision_at_K_{K}"] = float(cum_tp[K_actual - 1]) / K_actual

        recall_at_each_pos_idx = np.where(pos_arr == True)[0]
        if len(recall_at_each_pos_idx) > 0:
            precs = (cum_tp[recall_at_each_pos_idx] / (recall_at_each_pos_idx + 1)).tolist()
            ap = float(np.mean(precs))
        else:
            ap = 0.0

        op_results[m] = {**recalls, **precisions, "AP": ap}

        print(f"    {m:<22s} "
              f"AP={ap:.3f}  "
              f"R@FP=1e-3={recalls['recall_at_FP_0.001']:.3f}  "
              f"R@FP=1e-2={recalls['recall_at_FP_0.01']:.3f}  "
              f"P@K=100={precisions['precision_at_K_100']:.3f}")

    report = {
        "mvp": args.mvp,
        "rolling_fanout": args.rolling_fanout,
        "n_epochs": len(tc.shells),
        "n_redteam_events": len(rt),
        "attackers": attackers,
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "operating_points": op_results,
    }
    (out / "report_v2.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[save] {out / 'report_v2.json'}")

if __name__ == "__main__":
    main()
