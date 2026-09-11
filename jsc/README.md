# MemoryVLA as a baseline on JUPITER

MemoryVLA (arXiv 2508.19236, ICLR 2026) fine-tuned on our real-robot Franka datasets and scored
**open-loop on the same held-out episodes** as the SSM policy, so the two MAEs sit in one table.

Upstream clone is `baselines/MemoryVLA` at commit `d732ea9` (branch `main` = the OpenVLA-codebase
version, which is the one the published RMBench/LIBERO numbers come from). Everything we added
lives in `jsc/`; everything we *changed* upstream is four small diffs, listed under
[Patches](#patches) and visible in full with `git -C baselines/MemoryVLA diff`.

Why open-loop and not a success rate: there is no simulator for these tasks, and closed-loop
means robot time. Open-loop MAE against the teleoperator is what `wt-main/scripts/openloop_memmap.py`
already reports for our own policy (0.116 frozen / 0.026 LoRA), so it is the metric that already
has a baseline column to join.

## Status

| Piece | State |
|---|---|
| `env/venv_memvla` | built and verified |
| upstream patches | applied (6 files) |
| weights cached | CogACT-Large 30 GB, Llama-2 (mirror) 13 GB, DINOv2 2.3 GB, SigLIP 3.3 GB |
| `plant_flower_2scoops` RLDS | built (4.2 GB), **all checks pass** |
| `pottimer` RLDS | built (3.7 GB), **all checks pass** |
| training job | **running**: plant + pottimer, 20k steps, 1.55 s/step (~8.6 h), 0 OOM kills |
| open-loop eval | written, untested (needs a trained checkpoint) |

First green run: `SMOKE=1 BS=1`, 7 min wall clock on 4x GH200, `exit=0`. `BS=4` had failed at the
same point, so the binding constraint is activation memory, not the model/optimizer footprint --
see [Memory on GH200](#memory-on-gh200).

`jsc/check_rlds.py --task plant_flower_2scoops` passes every check: JOINT_POS masks all-True over
8 dims, chunk `(T, 16, 8)`, 8-dim proprio, instruction carried through, normalized actions inside
[-1, 1], q01/q99 within 0.015 rad of the raw memmap quantiles, and `val` holding exactly the five
held-out episodes (lengths 711/795/862/917/1003) with no train leakage.

## Llama-2 provenance (read this before quoting the number)

`prismatic` rebuilds the LLM from a HF repo and takes the tokenizer from it before the
CogACT/MemoryVLA checkpoint overwrites every LLM weight. The official
`meta-llama/Llama-2-7b-hf` is gated and this account gets **403** on it, so runs are currently
made against **`NousResearch/Llama-2-7b-hf`**, a verbatim re-upload, selected by
`MEMVLA_LLAMA2_REPO` in `env/setup_env_memvla.sh`.

The weights are identical and are overwritten at load regardless, so a run started on the mirror
stays valid. Once Meta grants access, comment that export out and re-run `jsc/fetch_models.sh` --
no code change, and no need to retrain.

## The gate itself

`meta-llama/Llama-2-7b-hf` is gated by Meta. The trap worth remembering: `HfApi().model_info()`
returns success for a gated repo you cannot actually read, so it is useless as an access check --
only an `hf_hub_download` of a real file returns the 403.

To switch to the official repo once access is granted: comment out `MEMVLA_LLAMA2_REPO` in
`env/setup_env_memvla.sh` and re-run `jsc/fetch_models.sh`.

## Environment

`env/venv_memvla`, built by `env/build_memvla_env.sh` (login node; needs network).
Separate from `env/venv` and `env/venv311` because MemoryVLA pins `transformers==4.40.1`, which
cannot coexist with venv311's transformers 5.13.

Python 3.11, not upstream's 3.10 — 3.11 is proven on this machine and every MemoryVLA pin has a
cp311 aarch64 wheel. Two pins we could not honour:

| Pin | Used instead | Why |
|---|---|---|
| `torch==2.2.0` | `torch==2.10.0+cu128` | PyTorch shipped no linux-aarch64 CUDA wheel before 2.5. 2.10 is what `env/venv311` already runs on this GH200. |
| `flash_attn==2.5.5` | not installed; `MEMVLA_NO_FLASH=1` turns the flag off | No aarch64 wheel, and torch's own flash SDPA backend is unavailable on this GH200 regardless. **It is not optional for training.** Eval escapes because `inference_mode=True` forces the flag `False` (`base_llm.py:111`), but `LLaMa2LLMBackbone` defaults `use_flash_attention_2=True` and `prismatic/models/materialize.py` never passes it, so training dies in transformers' `_check_and_enable_flash_attn_2`. With the flag off, transformers 4.40 selects SDPA -- the same attention mathematically, and prismatic's sequences are only ~290 tokens, so the missing fused kernel costs little. |

Three build traps worth remembering, all now pinned in `build_memvla_env.sh`:

* **`tensorflow-metadata`** resolves to 1.21.0, which imports `google.protobuf.runtime_version`
  (protobuf ≥ 5.27) while tensorflow 2.15 caps protobuf at < 5. Unpinned, `import tensorflow_datasets`
  dies. Pinned to 1.14.0.
* **`protobuf`** unpinned resolved to 6.x. Pinned `<5`.
* **`tensorflow-graphics`** is imported unconditionally by `oxe/utils/droid_utils.py` even though we
  never load DROID, and it declares `tensorflow-addons`, which has no aarch64 wheel. Installed
  `--no-deps`; the submodule we need does not touch addons.

## Data

`jsc/memmap_to_rlds.py` builds the RLDS/TFDS shards MemoryVLA's loader wants **from
`artifacts/memmap/<task>`, not from the LeRobot source**. That is the point: the memmap is what the
SSM policy trains and is scored on, so both models get the same pixels, the same actions, and the
same train/test episode split. Rebuilding from LeRobot would re-derive the split
(`random.Random(42).shuffle`) and re-decode the AV1 video, with two chances to silently disagree.

    setsid nohup bash jsc/convert_rlds_login.sh plant_flower_2scoops pottimer \
        > ../../logs/memvla-convert-login.log 2>&1 < /dev/null &

**Login node, not sbatch.** Under `srun` the converter deadlocks in `futex_do_wait` — 4 threads,
~2 MB read, nothing written, 0.24 s of CPU — and never emits an example. It reproduces with and
without a GPU allocated, so it is not GPU probing; the same script on the login node runs at
~8 s/episode. Unresolved, and not worth chasing for ~15 min of single-core work per task.
`jsc/convert_rlds.sbatch` is kept with the failure documented in its header.

Writes `artifacts/rlds/<task>/franka_lerobot/1.0.0/`. memmap `train` → RLDS `train`, memmap `test`
→ RLDS `val` (writing `val` explicitly matters: `dataset.py` otherwise silently slices
`train[:95%]` and the held-out episodes stop being held out).

Available splits, for reference — `plate_sponge` has **no** held-out episodes and cannot be scored
open-loop without a rebuild:

| task | train | test |
|---|---|---|
| `plant_flower_2scoops` | 45 ep / 36,207 fr | 5 ep / 4,288 fr |
| `pottimer` | 45 ep / 39,962 fr | 5 ep / 4,533 fr |
| `sponge_plate_mem` | 43 ep / 15,836 fr | 5 ep / 1,928 fr |
| `plate_sponge` | 34 ep / 17,092 fr | — |

MemoryVLA takes a **single** camera, so only the external `right`/front view is wired to
`observation/image`. The wrist stream is written into the shards but left unloaded, so a
two-camera variant later does not need a rebuild.

## Training

    sbatch --export=ALL,TASK=plant_flower_2scoops jsc/train_memvla.sbatch
    # SMOKE=1 -> 20 steps, to prove the pipeline before spending a day of GPU

Upstream's own real-world recipe (`script/train/real_world/train_real.sh`), changed only where
this machine or this dataset forces it:

| Setting | Ours | Upstream | Why |
|---|---|---|---|
| `action_dim` | 8 | 7 | absolute joint targets + gripper width, not EEF deltas |
| `data_mix` | `franka_lerobot` | `custom_finetuning` | our OXE registration |
| world size | 4 | 8 | JUPITER is 4xGH200/node |
| `per_device_batch_size` | 8 | 32 | 32 and 16 get a rank OOM-killed here |
| `global_batch_size` | 128 | 256 | effective batch held by gradient accumulation x4 |
| `shuffle_buffer_size` | 1000 | 32_000 | the buffer holds **episodes**, and we have 45 |

lr 2e-5, DiT-L, `mem_length 256`, `repeated_diffusion_steps 4`, `future_action_window_size 15`
(= chunk of 16, matching `future_steps: 16`) are upstream's, untouched.

**None of this changes the method.** The memory bank is an instance dict on `CogMemBank` that
persists across forward calls and is cleared per *episode*, not per batch, so `per_device_batch_size`
does not bound the temporal window -- `mem_length=256` does. (An earlier version of this file
claimed the opposite; it was wrong.)

Measured: **1.55 s/step**, so 20k steps is ~8.6 h, inside the 12 h QOS ceiling.

## Evaluation

    python jsc/openloop_memvla.py \
        --checkpoint ../../artifacts/memvla_runs/memvla_plant_flower_2scoops/checkpoints/step-020000.pt \
        --memmap ../../artifacts/memmap/plant_flower_2scoops \
        --out ../../artifacts/openloop_memvla/plant.json

Same metric as `wt-main/scripts/openloop_memmap.py`: mean |predicted − teleoperated| over every
frame and all 8 dims in **raw** units, scoring only the first action of each predicted chunk,
with the memory bank carried across the whole episode and reset only at t=0. The pre/wait/post
phase split is **lifted out of that script by AST** rather than reimplemented, so the boundaries
cannot drift between the two rows — this matters most for `pottimer`, where the 30 s hold means a
policy that simply freezes scores well on a frame-averaged MAE.

Defaults to DDIM-10 rather than the 100-step DDPM loop; `--ddpm` checks what that shortcut costs.

## Patches

Eight changes across six files against `d732ea9`. `git -C baselines/MemoryVLA diff` shows them
in full. Numbers 1, 2 and 6 fix things that corrupt results rather than crash.

0. **`prismatic/models/backbones/llm/llama2.py`** — two changes. (a) The 7B repo id now reads
   `MEMVLA_LLAMA2_REPO` (default: the official gated repo), so the mirror is an env setting
   rather than an edit. (b) `use_flash_attention_2` now defaults from `MEMVLA_NO_FLASH` instead
   of hardcoded `True`, because flash-attn is not built here and training — unlike eval — does
   not otherwise turn it off.
1. **`vla/memory_vla.py`** — the one that would have silently ruined the comparison.
   `predict_action` ran `normalized_actions[:, 6] = where(... < 0.5, 0, 1)` unconditionally. That
   is correct for upstream's 7-dim EEF action space, where index 6 *is* a binary gripper, and
   destructive for our 8-dim space, where index 6 is `joint_7` and the real gripper is a
   continuous width at index 7 — it would have snapped an elbow angle to 0 or 1 rad and left the
   gripper alone. Now guarded on `shape[-1] == 7`, so every upstream path is byte-identical.
2. **`vla/memory_vla.py` (`from_pretrained`)** — the released DiT is built for a 7-dim action
   space, so three tensors cannot transfer to our 8-dim head: `net.x_embedder.linear.weight`,
   `net.final_layer.linear.weight`, `net.final_layer.linear.bias`. `strict=False` does **not**
   cover this (it forgives missing/unexpected keys but still raises on size mismatch), so they are
   dropped explicitly and the rest of the DiT loads pretrained. **Only the action head's input and
   output projections train from scratch** — unavoidable when the action space changes, and worth
   stating whenever this baseline's number is reported.
3. **`vla/datasets/rlds/oxe/materialize.py`** — allow `ActionEncoding.JOINT_POS` (the enum already
   existed, only the mask branch was missing) with `absolute_action_mask=[True]*8`. All 8 dims are
   absolute, so past-the-end padding repeats the last valid action instead of zeroing it; and the
   gripper is normalized like the rest because it is a width, not a flag.
4. **`vla/datasets/rlds/oxe/configs.py`** — register `franka_lerobot` (`StateEncoding.JOINT`,
   `ActionEncoding.JOINT_POS`, single camera).
5. **`transforms.py` / `mixtures.py`** — identity standardization (the converter already writes the
   standardized layout) and a single-dataset mixture.
6. **`vla/memory_vla.py` (`from_pretrained`, checkpoint load)** — `torch.load(..., map_location=
   "cuda")` → `"cpu"`. Upstream deserializes the whole 30 GB CogACT checkpoint straight into HBM
   on **every** rank before FSDP has wrapped anything: on 4x GH200 that pinned 94.5 GB of the
   97.8 GB per GPU, and on Grace-Hopper's coherent memory the same pressure hit the host and got
   a rank SIGKILLed (`exitcode -9`, MaxRSS 402 GB). The model is still on CPU at that point, so
   the checkpoint has no reason to touch the GPU; `del model_state_dict` frees it right after.
   **Anyone judging batch-size headroom from a run without this fix will read it far too low.**

## Memory on GH200

Getting a 7B to train here took five fixes, four of them upstream behaviours that are harmless on
the 8x A100 x86 node MemoryVLA was written for. The failure mode throughout was
`torchrun ... exitcode: -9` -- a **SIGKILL from the kernel OOM killer, with no Python traceback
anywhere**. It is not a CUDA OOM: it reproduces with `PYTORCH_CUDA_ALLOC_CONF` unset, which would
have made real HBM exhaustion raise `CUDA out of memory` instead.

| Fix | Effect |
|---|---|
| `torch.load(map_location="cpu")`, was `"cuda"` | 94.5 GB -> 1.5 GB of HBM per rank at load |
| `torch.load(..., mmap=True)` | stops each rank materialising the 30 GB state dict in RAM |
| `PYTORCH_CUDA_ALLOC_CONF` unset | a GH200 allocator can back segments with Grace host memory, so HBM exhaustion silently spills to the host and gets OOM-killed instead of raising |
| `--mem=0 --exclusive` | the srun step otherwise gets a share of the node's 878 GB, not all of it |
| `per_device_batch_size` 4 -> 1, shuffle buffer 32k -> 1k for smoke | the one that actually closed it |

Diagnosing this: when a distributed job dies with `exitcode: -9`, read `/proc/<pid>/wchan` and
`/proc/<pid>/io` and suspect host memory. And do **not** size batches off an `nvidia-smi` reading
taken before the first training step -- 15.5 GB idle became 96 GB once the step ran.

## Things that will bite

* **Compute nodes have no network.** Every weight must be in `$HF_HOME` before the job starts;
  `jsc/fetch_models.sh` is login-node only.
* **The converter deadlocks under `srun`.** Run it on the login node (`jsc/convert_rlds_login.sh`).
* **Nothing started from an SSH session survives it.** Background jobs are children of that shell
  and die with it. Use `sbatch`, or `setsid nohup … &` for the downloads that must run on the login
  node.
* **timm model names are not HF repo ids.** `vit_large_patch14_reg4_dinov2.lvd142m` happens to
  live at `timm/<same name>`, but `vit_so400m_patch14_siglip_224` lives at
  `timm/ViT-SO400M-14-SigLIP`; guessing gives a 404. Let timm resolve it (`create_model(...,
  pretrained=True)`), which is what `jsc/fetch_models.sh` now does.
* **The RLDS pipeline repeats forever.** Counting episodes off `as_numpy_iterator()` never
  terminates and tells you nothing -- compare a bounded sample against known episode lengths.
* **`mem_length=256`, not the default 16.** Upstream's real-world config, and worth knowing when
  describing MemoryVLA as a "growing" memory: the bank is bounded, and past 256 entries it merges
  the two most similar adjacent frames (ToMe) rather than growing. Our episodes run ~500–900
  frames, so consolidation does kick in.
