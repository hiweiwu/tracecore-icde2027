# Storage ablation: naive snapshot vs CoreDelta log vs ShellIndex (Table II)
from __future__ import annotations
import json, pickle, time, struct
from pathlib import Path
from collections import defaultdict
import io

ROOT = Path("/private/workspace/icde_flow")
PKL = ROOT / "bench/lanl_redteam_range/tracecore_mvp.pkl"
OUTDIR = ROOT / "bench/storage_block3"
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"[load] {PKL}")
t0 = time.time()
with open(PKL, "rb") as f:
    d = pickle.load(f)
shells = d["shells"]
print(f"[load] {len(shells)} epochs, {sum(len(v) for v in shells.values()):,} (host, epoch) cells, "
      f"{time.time()-t0:.1f}s; pkl size = {PKL.stat().st_size/1e6:.2f} MB")

print("[naive] writing per-epoch JSON snapshots")
naive_path = OUTDIR / "naive_snapshots.jsonl"
total_bytes_uncompressed = 0
import gzip
gz_path = OUTDIR / "naive_snapshots.jsonl.gz"
with open(naive_path, "w") as f, gzip.open(gz_path, "wt") as gf:
    for eid in sorted(shells.keys()):
        line = json.dumps({"epoch": eid, "shells": shells[eid]}) + "\n"
        f.write(line); gf.write(line)
        total_bytes_uncompressed += len(line)
naive_size = naive_path.stat().st_size
naive_gz_size = gz_path.stat().st_size
print(f"  uncompressed: {naive_size/1e6:.2f} MB")
print(f"  gzipped:      {naive_gz_size/1e6:.2f} MB")

print("[delta] computing per-host delta archive")
prev_shell: dict[str, int] = {}
deltas = []
for eid in sorted(shells.keys()):
    cur = shells[eid]

    for host, c_new in cur.items():
        c_old = prev_shell.get(host, -1)
        if c_old != c_new:
            deltas.append((eid, host, c_old, c_new))

    for host in prev_shell:
        if host not in cur:
            deltas.append((eid, host, prev_shell[host], -1))
    prev_shell = dict(cur)
print(f"  total deltas: {len(deltas):,} ({100.0 * len(deltas) / sum(len(v) for v in shells.values()):.2f}% of full snapshots)")

delta_path = OUTDIR / "coredelta.tsv"
delta_gz_path = OUTDIR / "coredelta.tsv.gz"
with open(delta_path, "w") as f, gzip.open(delta_gz_path, "wt") as gf:
    f.write("epoch\thost\tc_old\tc_new\n"); gf.write("epoch\thost\tc_old\tc_new\n")
    for e, h, co, cn in deltas:
        line = f"{e}\t{h}\t{co}\t{cn}\n"
        f.write(line); gf.write(line)
delta_size = delta_path.stat().st_size
delta_gz_size = delta_gz_path.stat().st_size
print(f"  uncompressed: {delta_size/1e6:.2f} MB")
print(f"  gzipped:      {delta_gz_size/1e6:.2f} MB")

print("[shellindex] per-(epoch, k_bucket) sorted host IDs")

host_set = set()
for s in shells.values():
    host_set.update(s.keys())
host2id = {h: i for i, h in enumerate(sorted(host_set))}
print(f"  unique hosts: {len(host2id)}")

shellindex_path = OUTDIR / "shellindex.bin"
total = 0
with open(shellindex_path, "wb") as f:
    for eid in sorted(shells.keys()):

        by_k: dict[int, list[int]] = defaultdict(list)
        for h, k in shells[eid].items():
            by_k[k].append(host2id[h])

        for k, hosts in by_k.items():
            hosts.sort()

            header = struct.pack(">III", eid, k, len(hosts))
            f.write(header)
            prev = 0
            for h in hosts:
                gap = h - prev
                prev = h

                while gap >= 128:
                    f.write(bytes([(gap & 0x7f) | 0x80])); gap >>= 7
                f.write(bytes([gap & 0x7f]))
            total += len(header)
shellindex_size = shellindex_path.stat().st_size
shellindex_gz_path = OUTDIR / "shellindex.bin.gz"
with open(shellindex_path, "rb") as f, gzip.open(shellindex_gz_path, "wb") as gf:
    gf.write(f.read())
shellindex_gz_size = shellindex_gz_path.stat().st_size
print(f"  raw varint:   {shellindex_size/1e6:.2f} MB")
print(f"  gzipped:      {shellindex_gz_size/1e6:.2f} MB")

result = {
    "n_epochs": len(shells),
    "n_unique_hosts": len(host2id),
    "n_host_epoch_cells": sum(len(v) for v in shells.values()),
    "n_deltas": len(deltas),
    "deltas_pct_of_snapshots": round(100 * len(deltas) / sum(len(v) for v in shells.values()), 3),
    "sizes_mb": {
        "tracecore_pkl_in_memory_compressed": round(PKL.stat().st_size / 1e6, 2),
        "naive_snapshot_uncompressed": round(naive_size / 1e6, 2),
        "naive_snapshot_gzipped": round(naive_gz_size / 1e6, 2),
        "coredelta_uncompressed": round(delta_size / 1e6, 2),
        "coredelta_gzipped": round(delta_gz_size / 1e6, 2),
        "shellindex_varint": round(shellindex_size / 1e6, 2),
        "shellindex_varint_gzipped": round(shellindex_gz_size / 1e6, 2),
    },
}
ratios = result["sizes_mb"]
naive = ratios["naive_snapshot_uncompressed"]
result["ratios_vs_naive_uncompressed"] = {
    "naive_gz_vs_naive": round(naive / ratios["naive_snapshot_gzipped"], 2),
    "coredelta_vs_naive": round(naive / ratios["coredelta_uncompressed"], 2),
    "coredelta_gz_vs_naive": round(naive / ratios["coredelta_gzipped"], 2),
    "shellindex_vs_naive": round(naive / ratios["shellindex_varint"], 2),
    "shellindex_gz_vs_naive": round(naive / ratios["shellindex_varint_gzipped"], 2),
}

OUT = ROOT / "results/block3_storage.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2))
print()
print(json.dumps(result, indent=2))
print(f"[save] {OUT}")
