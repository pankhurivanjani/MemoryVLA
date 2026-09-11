#!/bin/bash
# Pre-cache every weight the MemoryVLA baseline needs. LOGIN NODE ONLY -- JUPITER compute nodes
# have no network, so anything not in $HF_HOME before the job starts is a hard failure inside it.
#
# Survives a dropped SSH session only if you detach it:
#   setsid nohup bash jsc/fetch_models.sh > ../../logs/memvla-fetch.log 2>&1 < /dev/null &
#
# Three things are needed, and they come from three places:
#   1. CogACT-Large.pt  -- the pretrained VLA checkpoint train.py fine-tunes from (~31 GB).
#   2. dinov2 + siglip  -- the two vision towers, pulled by timm through the HF hub.
#   3. Llama-2-7b       -- prismatic rebuilds the LLM from the HF repo before loading (1) over
#      it, so the repo must be reachable even though its weights are immediately overwritten.
#      This one is GATED; see the note where it fails.
set -uo pipefail
source /e/project1/m3/vanjani1/ssmpolicy/env/setup_env_memvla.sh
# This script is the one place that must reach the network.
unset HF_HUB_OFFLINE
export HF_HUB_ENABLE_HF_TRANSFER=0

python - <<'PY'
import os, sys, traceback
from huggingface_hub import snapshot_download, hf_hub_download

tok_path = os.path.join(os.environ["HF_HOME"], "token")
tok = open(tok_path).read().strip() if os.path.exists(tok_path) else None
ok, failed = [], []

def step(label, fn):
    print(f"\n=== {label} ===", flush=True)
    try:
        out = fn()
        print(f"  OK -> {out}", flush=True)
        ok.append(label)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        failed.append((label, type(e).__name__))

# 1. The VLA checkpoint. allow_patterns keeps this to the three files load_vla() actually opens
#    (config.json, dataset_statistics.json, checkpoints/*.pt) rather than the whole repo.
step("CogACT-Large (pretrained VLA checkpoint)", lambda: snapshot_download(
    "CogACT/CogACT-Large", token=tok,
    allow_patterns=["config.json", "dataset_statistics.json", "checkpoints/*.pt"]))

# 2. Vision towers. Do NOT guess the repo id from the timm model name -- they diverge:
#    vit_large_patch14_reg4_dinov2.lvd142m does live at timm/<same name>, but
#    vit_so400m_patch14_siglip_224 lives at timm/ViT-SO400M-14-SigLIP and guessing gives a 404.
#    Asking timm to build the model with pretrained=True makes timm resolve and cache exactly
#    what prismatic will ask for later, offline.
import timm

def _pull_timm(name):
    timm.create_model(name, pretrained=True, num_classes=0)
    return timm.get_pretrained_cfg(name).hf_hub_id

for tid in ("vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"):
    step(f"timm {tid}", lambda tid=tid: _pull_timm(tid))

# 3. The LLM, from whichever repo MEMVLA_LLAMA2_REPO selects (see env/setup_env_memvla.sh).
#    Gated by Meta on the official repo: model_info() succeeds without access but resolve/
#    returns 403, so this must be tested with a real file download, not with model_info.
LLAMA_REPO = os.environ.get("MEMVLA_LLAMA2_REPO", "meta-llama/Llama-2-7b-hf")
step(f"{LLAMA_REPO} (LLM base + tokenizer)", lambda: snapshot_download(
    LLAMA_REPO, token=tok,
    allow_patterns=["*.json", "*.model", "*.safetensors", "tokenizer*"]))

print("\n" + "=" * 70)
print("cached OK :", ", ".join(ok) if ok else "(none)")
for label, kind in failed:
    print(f"FAILED    : {label}  [{kind}]")
if any("Llama-2" in label for label, _ in failed):
    print("""
Llama-2 is gated. Two ways forward:
  (a) Request access at https://huggingface.co/meta-llama/Llama-2-7b-hf while signed in as
      pankhurivanjani (Meta's form; usually granted within minutes to a few hours), then re-run
      this script. Nothing else needs to change.
  (b) Point the backbone at a mirror instead -- see jsc/README.md. The checkpoint in step (1)
      overwrites every LLM weight at load time, so only the config and tokenizer are actually
      taken from this repo. This sidesteps Meta's gate, so it is your call, not one this script
      makes for you.""")
sys.exit(1 if failed else 0)
PY
