# TraceCore — Artifacts for the ICDE 2027 Submission

Implementation, experiment ledger, and per-run evidence for:

> **TraceCore: A Streaming Coreness Index over NetFlow for Time-Travel Forensics and Intrusion Detection.**
> Wei Wu, Yuan Liu, Hui Lu, Fan Zhang, Zhihong Tian. Submitted to ICDE 2027 (Research Track).

Every result-table number in the paper traces to one of the 40 runs in `evidence/ledger.json` (per-run key metrics and provenance notes); derived quantities are recomputable from `evidence/paper_derived_evidence.json`, and the full ledger table is reproduced in `supplementary.pdf`.

## Layout

| Path | Contents |
|---|---|
| `src/` | All TraceCore code: streaming engine + index, query primitives, TraceCore-R scorer, oracles, benchmarks, detection pipelines, GNN-baseline re-implementations, evidence derivation. |
| `baselines/` | GraphIDS run configs + pointers to the third-party baseline code we do not redistribute (see `baselines/README.md`). |
| `evidence/` | `ledger.json` (40-run ledger with per-run metrics), `paper_derived_evidence.json`. |
| `supplementary.pdf` | The paper's supplementary document: worked index example, extended result tables, and the complete 40-run ledger table. |

## Environment

Python 3.10+, CPU-only for everything except the GNN baselines. Key packages: `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `lightgbm`, `networkit`, `python-igraph` (the two faithfulness oracles), and `torch` (GNN baselines only). All seeds are fixed at **42** throughout.

## Datasets

Both corpora are public and are not redistributed here (size and licensing); download from source and run the preprocessing scripts below.

1. **LANL "Comprehensive, Multi-Source Cyber-Security Events" (Kent 2015)** — request/download from Los Alamos National Laboratory (https://csr.lanl.gov/data/cyber1/). We use only `flows.txt` and `redteam.txt`; host-side auth/proc/DNS logs are unused by design.
2. **NF-UQ-NIDS-v2 sub-datasets** (NF-CSE-CIC-IDS2018-v2, NF-ToN-IoT-v2, NF-UNSW-NB15-v2, NF-BoT-IoT-v2) — University of Queensland NIDS datasets (Sarhan et al.), https://staff.itee.uq.edu.au/marius/NIDS_datasets/ (at time of writing).

`src/prep_baseline_inputs.py` and `src/prep_nfuq_for_gnn.py` convert the raw downloads into the parquet inputs used by every script.

**Path convention:** scripts were run with the repository root at `/private/workspace/icde_flow` (subdirs `data/`, `results/`, `logs/`). The release copy strips comments and docstrings only -- it is AST-verified functionally identical to the code that produced the reported results, and the root constants are unchanged — create the same root (or a symlink), or edit the constant at the top of each script.

## License

Code is released under the MIT License (see `LICENSE`). The LANL and UQ datasets keep their own licenses.
