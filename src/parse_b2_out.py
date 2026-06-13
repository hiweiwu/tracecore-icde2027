# RQ2: parser for TCD/OTCD baseline timing output
import re, json, sys, numpy as np
from pathlib import Path

OUT = sys.argv[1] if len(sys.argv) > 1 else "/private/workspace/icde_flow/bench/slice1h/b2_tcd.out"

content = Path(OUT).read_text()

tcd_ns = [int(m.group(1)) for m in re.finditer(r"^TCD Time Clapse\(nanoseconds\):(\d+)", content, re.M)]
otcd_ns = [int(m.group(1)) for m in re.finditer(r"^OTCD Time Clapse\(nanoseconds\):(\d+)", content, re.M)]
queries = re.findall(r"Query:(\S+\s*\S*\s*\S*)", content)

if len(tcd_ns) != len(otcd_ns):
    print(f"WARNING: TCD/OTCD count differ ({len(tcd_ns)} vs {len(otcd_ns)}); truncating to min")
    n = min(len(tcd_ns), len(otcd_ns))
    tcd_ns = tcd_ns[:n]
    otcd_ns = otcd_ns[:n]

def stats(name, arr):
    if not arr:
        print(f"{name}: empty")
        return
    a = np.array(arr)
    print(f"{name}: n={len(a)}, mean={a.mean()/1000:.2f} us, "
          f"p50={np.percentile(a, 50)/1000:.2f} us, "
          f"p95={np.percentile(a, 95)/1000:.2f} us, "
          f"p99={np.percentile(a, 99)/1000:.2f} us, "
          f"max={a.max()/1000:.2f} us, sum={a.sum()/1e6:.3f} ms")

print(f"Parsed {len(tcd_ns)} queries from {OUT}")
stats("TCD ", tcd_ns)
stats("OTCD", otcd_ns)
print(f"Aggregate qps (TCD only)  = {len(tcd_ns)/(sum(tcd_ns)/1e9):,.0f} qps  (algorithm time)")
print(f"Aggregate qps (OTCD only) = {len(otcd_ns)/(sum(otcd_ns)/1e9):,.0f} qps  (algorithm time)")
print(f"Note: the 'algorithm time' excludes I/O and per-query setup overhead.")
