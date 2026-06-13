# RQ2: parser for iPHC baseline timing output
import re, sys, numpy as np, json
from pathlib import Path

OUT = sys.argv[1] if len(sys.argv) > 1 else "/private/workspace/icde_flow/bench/slice1h/b2_iphc.out"
content = Path(OUT).read_text()
ns = [int(m.group(1)) for m in re.finditer(r"^Clapse\(nanoseconds\):(\d+)", content, re.M)]
queries = [m.group(1) for m in re.finditer(r"^Query:(.+)$", content, re.M)]
print(f"Parsed {len(ns)} queries from {OUT}")
if not ns:
    sys.exit(1)
a = np.array(ns) / 1e6
print(f"iPHC per-query latency (ms): n={len(a)}, mean={a.mean():.2f}, "
      f"p50={np.percentile(a,50):.2f}, p95={np.percentile(a,95):.2f}, "
      f"p99={np.percentile(a,99):.2f}, max={a.max():.2f}, sum={a.sum():.2f}")
print(f"Throughput (alg only) = {len(a)/(a.sum()/1000):.2f} qps")
out = {
    "n_queries": len(a),
    "mean_ms": float(a.mean()),
    "p50_ms": float(np.percentile(a, 50)),
    "p95_ms": float(np.percentile(a, 95)),
    "p99_ms": float(np.percentile(a, 99)),
    "max_ms": float(a.max()),
    "sum_ms": float(a.sum()),
    "qps_alg_only": float(len(a) / (a.sum() / 1000)),
}
print(json.dumps(out, indent=2))
