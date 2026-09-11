#!/bin/bash
# memmap -> RLDS conversion. LOGIN NODE. Detach it so it survives your SSH session:
#
#   setsid nohup bash jsc/convert_rlds_login.sh plant_flower_2scoops pottimer \
#       > ../../logs/memvla-convert-login.log 2>&1 < /dev/null &
#
# Why not sbatch: under `srun` this script deadlocks. The process parks in futex_do_wait with
# 4 threads, ~2 MB read, nothing written and 0.24 s of CPU consumed, and never emits an example.
# It reproduces with and without a GPU allocated, so it is not GPU probing. The identical script
# on the login node runs at ~8 s/episode. Unresolved -- and not worth chasing for ~15 min of
# single-core work per task. jsc/convert_rlds.sbatch is kept for whoever wants to retry.
#
# Cost: one core, ~15 min per task, no GPU, no network.
set -uo pipefail
source /e/project1/m3/vanjani1/ssmpolicy/env/setup_env_memvla.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=""
cd /e/project1/m3/vanjani1/ssmpolicy/baselines/MemoryVLA
for T in "$@"; do
  echo "=== $T  $(date) ==="
  python -u jsc/memmap_to_rlds.py \
      --memmap "/e/project1/m3/vanjani1/ssmpolicy/artifacts/memmap/$T" \
      --out    "/e/project1/m3/vanjani1/ssmpolicy/artifacts/rlds"
  echo "=== $T exit=$? $(date) ==="
done
