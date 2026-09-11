#!/bin/bash
#SBATCH --account=m3
#SBATCH --partition=booster
#SBATCH --nodes=1          # override at submit: sbatch --nodes=N
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=12:00:00
# 12 h is the ceiling on the `normal` QOS (`sacctmgr show qos format=Name,MaxWall`); 24 h is
# rejected outright with QOSMaxWallDurationPerJobLimit. If 20k steps do not fit, resume from the
# last checkpoint: train.py takes --is_resume True --resume_step N --resume_epoch N with
# --pretrained_checkpoint <run>/checkpoints/step-<N>-epoch-<E>-loss=<L>.pt.
#SBATCH --job-name=memvla-train
#SBATCH --output=/e/project1/m3/vanjani1/ssmpolicy/logs/%x-%j.out
#
# Fine-tune MemoryVLA on a real-robot Franka task. This mirrors upstream's OWN real-world recipe
# (script/train/real_world/train_real.sh) and changes only what our data forces:
#
#   * --action_dim 8            upstream is 7 (EEF delta xyz+rpy+gripper); ours is 7 absolute
#                               joint targets + gripper width, matching what the SSM policy
#                               predicts, so the two MAEs are measured on the same quantity.
#   * --vla.data_mix            franka_lerobot, registered in vla/datasets/rlds/oxe/*.
#   * world size 4, not 8       JUPITER is 4x GH200 per node.
#   * per_device_batch_size 8,   upstream's 32 gets a rank OOM-killed here (see "Memory on GH200"
#     global_batch_size 128      in jsc/README.md). The EFFECTIVE batch is held at 128 by gradient
#                                accumulation, which base_strategy.py computes as
#                                global // per_device // world_size = 128//8//4 = 4.
#                                This is a throughput knob, NOT a change to the method: the
#                                memory bank is an instance dict on CogMemBank that persists
#                                across forward calls and is cleared per EPISODE, not per batch
#                                (vla/memory_vla.py `self.bank`), so the batch size does not
#                                bound the temporal window. mem_length=256 does.
#
# --mem=0 --exclusive are load-bearing, not tidiness. Without an explicit request the srun step
# gets a share of the node's 878 GB rather than the whole thing, and four ranks each holding an
# mmap'd 30 GB checkpoint plus the fp32 model exceed it -- the kernel OOM killer then takes rank 0
# out with SIGKILL, which torchrun reports only as `exitcode: -9` with no traceback. It is NOT a
# GPU OOM: it reproduces with PYTORCH_CUDA_ALLOC_CONF unset, which would have made a real HBM
# exhaustion raise "CUDA out of memory" instead.
#
# Everything else (lr 2e-5, DiT-L, mem_length 256, repeated_diffusion_steps 4,
# future_action_window_size 15, image_aug) is upstream's value, untouched.
#
#   sbatch --export=ALL,TASK=plant_flower_2scoops jsc/train_memvla.sbatch
#   SMOKE=1 -> 20 steps, to prove the pipeline before spending a day of GPU.
set -uo pipefail
source /e/project1/m3/vanjani1/ssmpolicy/env/setup_env_memvla.sh
cd /e/project1/m3/vanjani1/ssmpolicy/baselines/MemoryVLA

TASK="${TASK:?set TASK=<rlds dir name>, e.g. plant_flower_2scoops}"
SMOKE="${SMOKE:-0}"
SSM=/e/project1/m3/vanjani1/ssmpolicy
DATA_ROOT="$SSM/artifacts/rlds/${TASK}"
RUN_ROOT="$SSM/artifacts/memvla_runs"
RUN_ID="memvla_${TASK}"

N_GPU=4                                  # GPUs per node (gres above)
# [JSC] multi-node: torchrun needs --nnodes and each rank needs the SAME rendezvous. Derive
# both from SLURM so `sbatch --nodes=4` is the only thing that has to change. world size =
# N_NODES * N_GPU, and expected_world_size must match or FSDP hangs at init.
N_NODES="${SLURM_JOB_NUM_NODES:-1}"
WORLD=$((N_NODES * N_GPU))
# Capture what the caller actually exported BEFORE applying any default, so an explicit
# `--export=...,BS=1` survives the SMOKE branch. (Defaulting first and then reassigning inside
# the branch silently ignored the caller's value and made a batch-size sweep impossible.)
BS_REQ="${BS:-}"
SHUFFLE_REQ="${SHUFFLE:-}"
GLOBAL_BS_REQ="${GLOBAL_BS:-}"
MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"
if [ "$SMOKE" = "1" ]; then
  MAX_STEPS=20; SAVE_INTERVAL=10; RUN_ID="${RUN_ID}_smoke"
  BS="${BS_REQ:-4}"
  # Upstream does not scale the shuffle buffer with the step count, so a 20-step run would still
  # decode 32k frames into host RAM before step 0 -- minutes of wall clock, gigabytes per rank,
  # and no benefit at all for a pipeline check.
  SHUFFLE="${SHUFFLE_REQ:-1000}"
else
  BS="${BS_REQ:-8}"
  # In the EPISODIC path the shuffle buffer holds whole episodes, not frames -- frame transforms
  # run after the shuffle (dataset.py: "Apply Frame Transforms" follows dataset.shuffle). Our
  # tasks have 45 training episodes; upstream's 32_000 therefore asks tf.data to buffer 32,000
  # episodes drawn repeatedly from those 45, at ~73 MB of encoded PNGs each. That is the memory
  # blowup that OOM-killed every large-batch run at the shuffle-fill stage, and it is why
  # upstream's value is safe for them (their datasets have tens of thousands of episodes) and
  # not for us. 1000 still shuffles all 45 episodes many times over, and is the value verified
  # end-to-end at BS=8 with gradient accumulation x4.
  SHUFFLE="${SHUFFLE_REQ:-1000}"
fi
# Effective batch, held at upstream-comparable 128 by gradient accumulation rather than by a
# per-device batch this machine cannot fit.
GLOBAL_BS="${GLOBAL_BS:-$((WORLD * BS))}"
if [ "$SMOKE" != "1" ]; then GLOBAL_BS="${GLOBAL_BS_REQ:-128}"; fi

# load_vla() wants `<run_dir>/checkpoints/<name>.pt` with config.json + dataset_statistics.json
# one level up -- exactly the layout of the CogACT-Large snapshot, so point at the file in place.
CKPT=$(python - <<'PY'
from huggingface_hub import snapshot_download
import os, glob
d = snapshot_download("CogACT/CogACT-Large", local_files_only=True,
                      allow_patterns=["config.json", "dataset_statistics.json", "checkpoints/*.pt"])
print(glob.glob(os.path.join(d, "checkpoints", "*.pt"))[0])
PY
)
[ -f "$CKPT" ] || { echo "FATAL: CogACT-Large checkpoint not cached. Run jsc/fetch_models.sh on the LOGIN node."; exit 1; }

# --- resume ------------------------------------------------------------------------------
# 20k steps do not fit in the 12 h QOS cap (~6.9k steps/slice observed), so a run needs
# several slices. RESUME=1 picks this run's highest step-NNNNNN-epoch-EE-loss=L.pt and derives
# --is_resume/--resume_step/--resume_epoch from the FILENAME, which is the only place the
# epoch is recorded. Default RESUME=0 reproduces the original fresh-start behaviour exactly.
IS_RESUME=False; RESUME_STEP=0; RESUME_EPOCH=0
if [ "${RESUME:-0}" = "1" ]; then
  CKDIR="$RUN_ROOT/${RUN_ID}--image_aug/checkpoints"
  LATEST=$(ls -1 "$CKDIR"/step-*.pt 2>/dev/null | sed -E 's/.*step-0*([0-9]+)-epoch-0*([0-9]+)-.*/\1 \2 &/' \
           | sort -k1,1n | tail -1)
  [ -n "$LATEST" ] || { echo "FATAL: RESUME=1 but no step-*.pt under $CKDIR"; exit 1; }
  RESUME_STEP=$(awk '{print $1}' <<<"$LATEST")
  RESUME_EPOCH=$(awk '{print $2}' <<<"$LATEST")
  CKPT=$(awk '{print $3}' <<<"$LATEST")
  IS_RESUME=True
  if [ "$RESUME_STEP" -ge "$MAX_STEPS" ]; then
    echo "DONE: $RUN_ID already at step $RESUME_STEP >= MAX_STEPS $MAX_STEPS"; exit 0
  fi
  echo "[resume] step=$RESUME_STEP epoch=$RESUME_EPOCH from $CKPT"
fi
[ -d "$DATA_ROOT/franka_lerobot" ] || { echo "FATAL: no RLDS at $DATA_ROOT. Run jsc/convert_rlds.sbatch first."; exit 1; }

echo "task=$TASK  nodes=$N_NODES gpus/node=$N_GPU world=$WORLD  bs=$BS  global_bs=$GLOBAL_BS  max_steps=$MAX_STEPS  smoke=$SMOKE"
echo "ckpt=$CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Compute nodes are offline; every weight must already be in $HF_HOME.
export HF_HUB_OFFLINE=1

# setup_env_memvla.sh sets expandable_segments:True, which is right for the SSM policy and wrong
# here. On a GH200 the CUDA allocator can back segments with Grace host memory, so exhausting the
# 97.8 GB of HBM does not raise "CUDA out of memory" -- it quietly spills to the host until the
# kernel OOM killer takes a rank out with SIGKILL (exitcode -9), which reports nothing useful.
# Turning it off makes a real GPU OOM fail loudly, with the requested/free byte counts.
# MEMVLA_EXPANDABLE=1 restores it once the memory envelope is known.
if [ "${MEMVLA_EXPANDABLE:-0}" != "1" ]; then
  unset PYTORCH_CUDA_ALLOC_CONF
  echo "PYTORCH_CUDA_ALLOC_CONF unset (GPU OOM will raise instead of spilling to host)"
fi
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29511}

# --trackers must be a JSON list. draccus splits any other form into CHARACTERS and raises
# nothing: '("jsonl",)' parses to ('(', '"', 'j', 's', ...) and a bare 'jsonl' to ('j','s',...).
# jsonl only -- W&B would need network from a compute node that has none.

srun torchrun --nnodes=$N_NODES --nproc_per_node=$N_GPU \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  train.py \
    --pretrained_checkpoint "$CKPT" \
    --vla.type 'prism-dinosiglip-224px+oxe+diffusion' \
    --vla.data_mix 'franka_lerobot' \
    --vla.expected_world_size $WORLD \
    --vla.per_device_batch_size $BS \
    --vla.global_batch_size $GLOBAL_BS \
    --vla.learning_rate 2e-5 \
    --vla.max_steps $MAX_STEPS \
    --vla.shuffle_buffer_size $SHUFFLE \
    --data_root_dir "$DATA_ROOT" \
    --run_root_dir "$RUN_ROOT" \
    --run_id "$RUN_ID" \
    --image_aug True \
    --save_interval $SAVE_INTERVAL \
    --repeated_diffusion_steps 4 \
    --future_action_window_size 15 \
    --action_dim 8 \
    --action_model_type 'DiT-L' \
    --dataloader_type 'stream' \
    --mem_length 256 \
    --is_resume $IS_RESUME \
    --resume_step $RESUME_STEP \
    --resume_epoch $RESUME_EPOCH \
    --trackers '["jsonl"]' \
    --hf_token "$HF_HOME/token"
rc=$?
echo "exit=$rc at $(date)"

# --- self-chain ---------------------------------------------------------------------------
# 20k steps need ~3 slices of the 12 h QOS cap. CHAIN=1 resubmits this same script with
# RESUME=1 so the run continues unattended. autoresume.sh only globs the SSM cells
# (plant_*|pottimer_*|plate_sponge_*) and never covered these `memvla-train` jobs, which is
# why both runs sat idle for 14 h after their first timeout.
#
# Guard: only resubmit if the newest checkpoint is strictly HIGHER than the one we resumed
# from. A slice that crashes before its first save would otherwise resubmit itself forever.
# A manual `scancel` sends SIGTERM, srun exits, and this block still runs -- so a deliberate
# cancel used to resubmit the very run you just killed. A sentinel file is the kill switch.
if [ -f "$RUN_ROOT/.chain_disabled" ]; then
  echo "[chain] $RUN_ROOT/.chain_disabled present -- not resubmitting"
elif [ "${CHAIN:-0}" = "1" ]; then
  CKDIR="$RUN_ROOT/${RUN_ID}--image_aug/checkpoints"
  NOW=$(ls -1 "$CKDIR"/step-*.pt 2>/dev/null | sed -E 's/.*step-0*([0-9]+)-epoch.*/\1/' | sort -n | tail -1)
  NOW="${NOW:-0}"
  if [ "$NOW" -ge "$MAX_STEPS" ]; then
    echo "[chain] $RUN_ID reached step $NOW >= $MAX_STEPS -- done, not resubmitting"
  elif [ "$NOW" -le "$RESUME_STEP" ]; then
    echo "[chain] NO PROGRESS (newest step $NOW <= resumed-from $RESUME_STEP) -- refusing to resubmit"
  else
    jid=$(sbatch --parsable --export=ALL,TASK="$TASK",RESUME=1,CHAIN=1,MAX_STEPS="$MAX_STEPS" \
          /e/project1/m3/vanjani1/ssmpolicy/baselines/MemoryVLA/jsc/train_memvla.sbatch)
    echo "[chain] $RUN_ID at step $NOW -> resubmitted as $jid"
  fi
fi
