"""Open-loop rollout of a fine-tuned MemoryVLA on an ssmpolicy memmap split.

This is the MemoryVLA counterpart of wt-main/scripts/openloop_memmap.py, and it is written to
produce a number that can sit in the same table as that script's: same held-out episodes, same
ground truth, same metric (mean |predicted - teleoperated| over every frame and every action
dimension, in RAW units -- joint radians and gripper width, not normalized ones), same
pre/wait/post phase split.

"Open-loop" means the policy sees the RECORDED observation at every step and its own predictions
are never fed back. The memory bank IS carried across the whole episode (reset only at t=0, via
`episode_first_frame='True'`), so memory still has to work over the full horizon -- this is not a
per-frame independent evaluation, exactly as in the SSM script.

Two deliberate choices, both of which affect comparability:
  * Only the FIRST action of the predicted chunk is scored (`actions[0]`), because that is the
    one the robot would execute at time t and it is what openloop_memmap.py scores.
  * DDIM with 10 steps, not the 100-step DDPM default. MemoryVLA is a diffusion policy, so a
    full DDPM loop per frame would put a 4k-frame split into the multi-hour range. Set
    --ddpm to check that the shortcut is not costing accuracy.

    python jsc/openloop_memvla.py \
        --checkpoint ../../artifacts/memvla_runs/memvla_plant_flower_2scoops/checkpoints/step-020000.pt \
        --memmap ../../artifacts/memmap/plant_flower_2scoops \
        --out ../../artifacts/openloop_memvla/plant.json
"""
import argparse
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vla.load import load_vla  # noqa: E402

IMG = 224
# The SSM open-loop script. We lift `_phases` out of it rather than reimplementing, so the phase
# boundaries cannot drift between the two rows of the table.
SSM_OPENLOOP = "/e/project1/m3/vanjani1/ssmpolicy/wt-main/scripts/openloop_memmap.py"


def _load_phases():
    """Compile just `_phases` out of the SSM script.

    A plain import will not do: that module imports eval_common at module scope, which needs the
    SSM policy env (mamba-ssm, hydra, lightning) that this venv deliberately does not have.
    `_phases` itself is pure numpy, so lifting the one function is both sufficient and exact.
    """
    try:
        tree = ast.parse(open(SSM_OPENLOOP).read())
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_phases")
        ns = {"np": np}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), SSM_OPENLOOP, "exec"), ns)
        return ns["_phases"], "lifted from openloop_memmap.py"
    except Exception as e:
        print(f"WARNING: could not lift _phases ({type(e).__name__}: {e}); "
              f"phase MAE will be omitted", flush=True)
        return (lambda gt, **kw: (None, None)), f"unavailable ({type(e).__name__})"


def _load_split(root: Path, split: str):
    d = root / split
    meta = json.load(open(d / "meta.json"))
    n = meta["total_frames"]
    if n == 0:
        raise SystemExit(f"{d} has 0 frames -- this task was built without a held-out split")
    obs = np.memmap(d / "observations.dat", dtype="float32", mode="r",
                    shape=(n, meta["observation_dim"]))
    act = np.memmap(d / "actions.dat", dtype="float32", mode="r",
                    shape=(n, meta["action_dim"]))
    img = np.memmap(d / "images_front_rgb.dat", dtype="uint8", mode="r", shape=(n, IMG, IMG, 3))
    return meta, obs, act, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to .pt under <run>/checkpoints/")
    ap.add_argument("--memmap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--unnorm_key", default="franka_lerobot")
    ap.add_argument("--cfg_scale", type=float, default=1.5)
    ap.add_argument("--ddpm", action="store_true", help="full 100-step DDPM instead of DDIM-10")
    ap.add_argument("--num_ddim_steps", type=int, default=10)
    ap.add_argument("--episodes", default=None, help="comma-separated indices within the split")
    a = ap.parse_args()

    phases, phases_src = _load_phases()
    root = Path(a.memmap).resolve()
    meta, obs, act, img = _load_split(root, a.split)
    want = None if a.episodes is None else {int(x) for x in a.episodes.split(",") if x.strip()}

    # Rebuild the model with the SAME architecture kwargs training used. This is not optional:
    # `MemoryVLA.from_pretrained` defaults action_dim to 7 and loads the action head with
    # strict=False, so an 8-dim checkpoint under a 7-dim default would load the DiT body, leave
    # the in/out projections randomly initialised, and predict confident garbage with no error
    # anywhere. Reading them back out of the run's own config.json is what keeps eval from
    # drifting away from training.
    run_dir = Path(a.checkpoint).resolve().parents[1]
    cfg_json = run_dir / "config.json"
    if not cfg_json.exists():
        raise SystemExit(f"missing {cfg_json}; cannot confirm the architecture this checkpoint "
                         f"was trained with")
    tcfg = json.load(open(cfg_json))
    arch = {k: tcfg[k] for k in (
        "action_dim", "future_action_window_size", "action_model_type", "use_ema",
        "dataloader_type", "group_size", "per_token_size", "mem_length",
        "retrieval_layers", "use_timestep_pe", "fusion_type", "consolidate_type",
        "update_fused") if k in tcfg}
    print(f"loading {a.checkpoint}", flush=True)
    print(f"  arch from config.json: {arch}", flush=True)
    if arch.get("action_dim") != meta["action_dim"]:
        raise SystemExit(f"trained action_dim={arch.get('action_dim')} but the memmap has "
                         f"{meta['action_dim']}-dim actions")

    vla = load_vla(a.checkpoint, load_for_training=False, **arch)
    vla = vla.to("cuda").eval()
    act_dim = vla.get_action_dim(a.unnorm_key)
    if act_dim != meta["action_dim"]:
        raise SystemExit(f"norm stats for `{a.unnorm_key}` are {act_dim}-dim but the memmap has "
                         f"{meta['action_dim']}; wrong unnorm_key")
    if vla.action_model.in_channels != meta["action_dim"]:
        raise SystemExit(f"action head takes {vla.action_model.in_channels} channels, memmap has "
                         f"{meta['action_dim']}")

    episodes = []
    for ep_i, tr in enumerate(meta["trajectories"]):
        if want is not None and ep_i not in want:
            continue
        s, L = tr["start_idx"], tr["length"]
        instruction = tr.get("instruction", "")
        preds, gts = [], []
        for t in range(L):
            r = s + t
            frame = Image.fromarray(np.asarray(img[r]))
            with torch.no_grad():
                actions, _ = vla.predict_action(
                    image=frame,
                    instruction=instruction,
                    unnorm_key=a.unnorm_key,
                    cfg_scale=a.cfg_scale,
                    use_ddim=not a.ddpm,
                    num_ddim_steps=a.num_ddim_steps,
                    # resets both memory banks and cur_timestep; everything after t=0
                    # accumulates into the bank exactly as it would on the robot.
                    episode_first_frame="True" if t == 0 else "False",
                )
            preds.append(np.asarray(actions[0], dtype=np.float64))
            gts.append(np.asarray(act[r], dtype=np.float64))
            if t % 200 == 0:
                print(f"  ep{ep_i} {t}/{L}", flush=True)

        P, G = np.stack(preds), np.stack(gts)
        err = np.abs(P - G)
        ph, resume = phases(G)
        phase_mae = None
        if ph is not None:
            phase_mae = {k: float(err[v].mean()) for k, v in ph.items() if err[v].size}
            phase_mae["wait_frames"] = int(ph["wait"].stop - ph["wait"].start)
            phase_mae["resume_frame"] = int(resume)
            dP = np.abs(np.diff(P[ph["wait"]], axis=0)).sum(1)
            dG = np.abs(np.diff(G[ph["wait"]], axis=0)).sum(1)
            phase_mae["wait_motion_pred"] = float(dP.mean()) if dP.size else 0.0
            phase_mae["wait_motion_gt"] = float(dG.mean()) if dG.size else 0.0
        episodes.append({
            "phase_mae": phase_mae,
            "episode": ep_i, "length": L, "instruction": instruction,
            "mae": float(err.mean()),
            "mae_per_dim": err.mean(0).round(6).tolist(),
            "rmse": float(np.sqrt(((P - G) ** 2).mean())),
            "pred": P.round(5).tolist(), "gt": G.round(5).tolist(),
        })
        print(f"  ep{ep_i}: mae={episodes[-1]['mae']:.4f}", flush=True)

    overall = float(np.mean([e["mae"] for e in episodes])) if episodes else float("nan")
    out = {
        "model": "MemoryVLA",
        "checkpoint": str(a.checkpoint),
        "memmap": str(root), "split": a.split,
        "sampler": "ddpm" if a.ddpm else f"ddim-{a.num_ddim_steps}",
        "cfg_scale": a.cfg_scale,
        "phases_source": phases_src,
        "mae": overall,
        "episodes": episodes,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nMemoryVLA open-loop MAE ({root.name}/{a.split}): {overall:.4f}"
          f"  over {len(episodes)} episodes -> {a.out}")


if __name__ == "__main__":
    main()
