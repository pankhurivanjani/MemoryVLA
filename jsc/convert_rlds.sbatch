#!/bin/bash
#SBATCH --account=m3
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --job-name=memvla-rlds
#SBATCH --output=/e/project1/m3/vanjani1/ssmpolicy/logs/%x-%j.out
#
# *** DOES NOT WORK -- use jsc/convert_rlds_login.sh instead. ***
# Kept only so the failure is on record and someone can retry it. Under srun this deadlocks in
# futex_do_wait (4 threads, ~2 MB read, nothing written, 0.24 s CPU) before emitting a single
# example, with or without a GPU allocated. The same script on the login node runs fine.
#
# memmap -> RLDS conversion for the MemoryVLA baseline. Pure CPU work (PNG encoding).
#
# NO GPU is requested, and CUDA_VISIBLE_DEVICES is forced empty below, on purpose. With a GH200
# visible, this job deadlocks: the process parks in futex_do_wait with 4 threads, 2 MB read and
# nothing written, and never emits a single example. Calling
# tf.config.set_visible_devices([], "GPU") is too late -- `import tensorflow` has already probed
# the device by then. On the login node, where there is no GPU to probe, the identical script
# runs at ~8 s/episode. Hiding the GPU from the process reproduces those conditions.
#
#   sbatch --export=ALL,TASK=plant_flower_2scoops jsc/convert_rlds.sbatch
set -uo pipefail
source /e/project1/m3/vanjani1/ssmpolicy/env/setup_env_memvla.sh
cd /e/project1/m3/vanjani1/ssmpolicy/baselines/MemoryVLA
TASK="${TASK:?set TASK=<memmap dir name>, e.g. plant_flower_2scoops}"
# Must come AFTER setup_env_memvla.sh, which defaults it to 0.
export CUDA_VISIBLE_DEVICES=""
echo "converting $TASK on $(hostname) at $(date)"
# --ntasks-per-node=1 above is load-bearing: without it srun fans out one task per CPU chunk,
# and every one of them runs the whole conversion into its own TFDS .incomplete* dir. The
# symptom is ~18 idle processes and an output tree that never grows. -u so the progress bar
# reaches the log while the job is still running.
srun python -u jsc/memmap_to_rlds.py \
    --memmap "/e/project1/m3/vanjani1/ssmpolicy/artifacts/memmap/${TASK}" \
    --out    "/e/project1/m3/vanjani1/ssmpolicy/artifacts/rlds"
echo "exit=$? at $(date)"
