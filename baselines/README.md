# Baseline assets

We do not redistribute third-party code. This directory holds only our own configuration files plus exact pointers to the upstream code we ran.

## GraphIDS (Guerra et al., official implementation)

- Upstream: the authors' official repository (cited as [GraphIDS] in the paper).
- `graphids_configs/` contains the YAML configs we ran:
  - `NF-CSE-CIC-IDS2018-v2.yaml`, `NF-UNSW-NB15-v2.yaml` — upstream configs, **unchanged** (fraction = 0.2, 100 epochs, default early stopping).
  - `NF-ToN-IoT-v2.yaml` — our config for the sub-dataset the upstream repo does not ship a config for, following the upstream defaults.
- The "parser-patched" NF-ToN-IoT-v2 run noted in the paper patches only the upstream **metrics parser** (output parsing crashed on that dataset); no algorithmic change.

## E-GraphSAGE

- Upstream notebooks: https://github.com/waimorris/E-GraphSAGE (per-flow binary-classification notebooks under `netflow/`; inspected May 2026).
- Note the runtime source-IP randomization step documented in the paper (Sec. V-A); our comparison disables it by construction.
- Our pure-PyTorch re-implementation (no DGL/PyG): `../src/egraphsage_pure_torch.py`.

## Anomal-E

- Re-implemented per the paper's recipe on the same encoder: 30 epochs Deep Graph Infomax + 200-tree Isolation Forest on benign-only training-edge embeddings.
- Our implementation: `../src/anomale_pure_torch.py`.

## Yang et al. temporal k-core baselines (TCD / OTCD / iPHC-TCQ)

- Built from the authors' released C++ source (`TCD.cpp`, `i-PHC.cpp`; see [Yang et al., PVLDB 2023] in the paper).
- Graph/query input formats follow the authors' `graph-format.txt` / `query-format.txt`.
- Our query sampling + timing parsers: `../src/parse_b2_out.py`, `../src/parse_b2_iphc.py`.
