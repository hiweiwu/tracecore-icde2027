#!/usr/bin/env bash
eval "$(/opt/conda/condabin/conda shell.bash hook)"
conda activate tracecore_cpu
cd /private/workspace/icde_flow/code
mkdir -p /private/workspace/icde_flow/logs

case "$1" in
  role)
    exec python role_residual_r1.py > /private/workspace/icde_flow/logs/role_residual_r1.log 2>&1 ;;
  stage3)
    exec python block1_stage3_oracle.py > /private/workspace/icde_flow/logs/block1_stage3.log 2>&1 ;;
  protocol)
    { python protocol_variants.py --sub NF-UNSW-NB15-v2
      python protocol_variants.py --sub NF-CSE-CIC-IDS2018-v2
      python protocol_variants.py --sub NF-ToN-IoT-v2
    } > /private/workspace/icde_flow/logs/protocol_variants.log 2>&1 ;;
  bench)
    exec python bench_streaming_systems.py > /private/workspace/icde_flow/logs/streaming_systems.log 2>&1 ;;
esac
