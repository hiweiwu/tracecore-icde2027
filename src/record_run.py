# Ledger bookkeeping: per-run manifest + parsed metrics
from __future__ import annotations
import argparse, json, shutil, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path("/private/workspace/icde_flow/results")
RUNS = ROOT / "runs"
LEDGER_MD = ROOT / "RESULTS_LEDGER.md"
LEDGER_JSON = ROOT / "ledger.json"

def slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("short_tag", help="short identifier, e.g. b2_tcd_slice1h")
    p.add_argument("category", choices=["main", "baseline", "ablation",
                                         "sanity", "smoke", "infra"])
    p.add_argument("--description", default="")
    p.add_argument("--cmd", default="", help="command line that produced the run")
    p.add_argument("--raw-log", help="copy this file into the run dir as raw.log")
    p.add_argument("--inputs", nargs="*", default=[],
                   help="paths to input artifacts (will be referenced, not copied)")
    p.add_argument("--outputs", nargs="*", default=[],
                   help="paths to output artifacts (will be copied into the run dir)")
    p.add_argument("--metric", action="append", default=[],
                   help="key=value metric pair; repeat as needed")
    p.add_argument("--status", default="OK")
    p.add_argument("--note", default="")
    args = p.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = RUNS / f"{ts}_{slug(args.short_tag)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    for kv in args.metric:
        if "=" in kv:
            k, v = kv.split("=", 1)

            try:
                v = float(v) if "." in v else int(v)
            except Exception:
                pass
            metrics[k.strip()] = v

    if args.raw_log and Path(args.raw_log).exists():
        shutil.copy2(args.raw_log, run_dir / "raw.log")

    for out in args.outputs:
        out_p = Path(out)
        if out_p.exists():
            shutil.copy2(out_p, run_dir / out_p.name)

    if args.cmd:
        (run_dir / "cmd.sh").write_text("#!/usr/bin/env bash\n" + args.cmd + "\n")

    manifest = {
        "short_tag": args.short_tag,
        "category": args.category,
        "description": args.description,
        "timestamp": ts,
        "started_iso": datetime.now().isoformat(),
        "cmd": args.cmd,
        "inputs": args.inputs,
        "outputs_copied": [p for p in args.outputs if Path(p).exists()],
        "metrics": metrics,
        "status": args.status,
        "note": args.note,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    ledger = []
    if LEDGER_JSON.exists():
        ledger = json.loads(LEDGER_JSON.read_text())
    ledger.append({
        "ts": ts, "tag": args.short_tag, "category": args.category,
        "status": args.status, "metrics": metrics, "dir": str(run_dir),
        "note": args.note,
    })
    LEDGER_JSON.write_text(json.dumps(ledger, indent=2))

    rows = ["# TraceCore Results Ledger\n",
            "_All experiment runs (main / baseline / ablation / sanity / smoke / infra) are recorded here. "
            "Do NOT delete entries; this is the authoritative experiment journal._\n",
            "| # | Timestamp | Tag | Category | Status | Headline metrics | Note | Dir |",
            "|---|---|---|---|---|---|---|---|"]
    for i, e in enumerate(ledger, 1):
        m_pretty = ", ".join(f"{k}={v}" for k, v in e["metrics"].items())[:120]
        d = e["dir"].split("/")[-1]
        rows.append(f"| {i} | {e['ts']} | {e['tag']} | {e['category']} | {e['status']} | {m_pretty} | {e.get('note','')} | `{d}` |")
    LEDGER_MD.write_text("\n".join(rows) + "\n")

    print(f"Recorded run: {run_dir}")
    print(f"  metrics: {metrics}")
    print(f"  ledger entries: {len(ledger)}")

if __name__ == "__main__":
    main()
