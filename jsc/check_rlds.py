"""Validate the converted RLDS against the memmap it came from, through MemoryVLA's own loader.

This exercises everything except the model: the `franka_lerobot` OXE registration, the JOINT_POS
mask branch, the converter's layout, action chunking, and BOUNDS_Q99 normalization. It needs no
LLM weights, so it can run while Llama-2 access is still pending.

What it checks, in order of how badly each would corrupt the baseline if wrong:

  1. Action statistics round-trip. The RLDS pipeline normalizes to [q01, q99] -> [-1, 1] and
     `predict_action` inverts that at eval time. If the q01/q99 the loader computed do not match
     the memmap's own train-split quantiles, every reported action is silently rescaled.
  2. `absolute_action_mask` is all-True over 8 dims. If it were the EEF default the last action
     of every chunk near an episode end would be zeroed instead of held.
  3. Chunk shape is (16, 8) -- 1 current + 15 future, matching the SSM policy's future_steps: 16.
  4. Proprio is 8-dim, i.e. joint_state and gripper_state concatenated in that order.
  5. The val split is non-empty and disjoint in length-profile from train (a cheap check that
     the held-out episodes really were written as `val` and not sliced out of `train`).

    python jsc/check_rlds.py --task plant_flower_2scoops
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402

tf.config.set_visible_devices([], "GPU")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vla.datasets.rlds.dataset import make_interleaved_episodic_dataset  # noqa: E402
from vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights  # noqa: E402
from vla.datasets.rlds.utils.data_utils import NormalizationType  # noqa: E402

FUTURE = 15  # future_action_window_size; chunk is FUTURE + 1


def build(data_root: Path, train: bool):
    per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
        data_root, [("franka_lerobot", 1.0)],
        load_camera_views=("primary",), load_depth=False, load_proprio=True,
        load_language=True, action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
    )
    cfg = dict(
        traj_transform_kwargs=dict(window_size=1, future_action_window_size=FUTURE,
                                   skip_unlabeled=True),
        frame_transform_kwargs=dict(resize_size=(224, 224), num_parallel_calls=8),
        dataset_kwargs_list=per_dataset_kwargs, shuffle_buffer_size=1000,
        sample_weights=weights, balance_weights=True,
        traj_transform_threads=1, traj_read_threads=1, train=train,
        load_all_data_for_training=False,
    )
    # Returns (dataset, dataset_len, all_dataset_statistics) -- not a bare dataset.
    ds, ds_len, stats = make_interleaved_episodic_dataset(**cfg, use_optim_group_sample=False)
    return ds, per_dataset_kwargs[0], ds_len, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--rlds", default="/e/project1/m3/vanjani1/ssmpolicy/artifacts/rlds")
    ap.add_argument("--memmap", default="/e/project1/m3/vanjani1/ssmpolicy/artifacts/memmap")
    a = ap.parse_args()

    data_root = Path(a.rlds) / a.task
    mm_root = Path(a.memmap) / a.task
    fails = []

    def check(label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
        if not ok:
            fails.append(label)

    print(f"\n=== {a.task} ===")
    ds, kwargs, ds_len, stats = build(data_root, train=True)

    print("\n-- dataset kwargs (the JOINT_POS mask branch) --")
    check("absolute_action_mask is all-True over 8 dims",
          kwargs.get("absolute_action_mask") == [True] * 8,
          str(kwargs.get("absolute_action_mask")))
    check("action_normalization_mask is all-True over 8 dims",
          kwargs.get("action_normalization_mask") == [True] * 8,
          str(kwargs.get("action_normalization_mask")))

    print("\n-- one episode through the loader --")
    ep = next(iter(ds.as_numpy_iterator()))
    act, obs = ep["action"], ep["observation"]
    check("action chunk is (T, 16, 8)",
          act.ndim == 3 and act.shape[1:] == (FUTURE + 1, 8), str(act.shape))
    check("proprio is 8-dim", obs["proprio"].shape[-1] == 8, str(obs["proprio"].shape))
    check("image is 224x224x3 uint8",
          obs["image_primary"].shape[-3:] == (224, 224, 3), str(obs["image_primary"].shape))
    instr = ep["task"]["language_instruction"][0]
    instr = instr.decode() if isinstance(instr, bytes) else str(instr)
    check("language instruction is non-empty", len(instr) > 0, instr[:60] + "...")
    check("normalized actions lie in [-1, 1]",
          float(np.abs(act).max()) <= 1.0 + 1e-5, f"max|a|={np.abs(act).max():.4f}")

    print("\n-- action stats vs the memmap's own train quantiles --")
    st = stats.get("franka_lerobot", stats) if isinstance(stats, dict) else {}
    st = st.get("action", st)
    if not (isinstance(st, dict) and "q01" in st):
        check("action statistics available", False, f"got keys {list(st)[:6]}")
    else:
        q01, q99 = np.array(st["q01"], float), np.array(st["q99"], float)
        meta = json.load(open(mm_root / "train" / "meta.json"))
        raw = np.memmap(mm_root / "train" / "actions.dat", dtype="float32", mode="r",
                        shape=(meta["total_frames"], meta["action_dim"]))
        r01, r99 = np.quantile(np.asarray(raw), 0.01, axis=0), np.quantile(np.asarray(raw), 0.99, axis=0)
        d01, d99 = np.abs(q01 - r01).max(), np.abs(q99 - r99).max()
        # Loose tolerance on purpose: the loader's quantiles are over chunked, padded actions,
        # so they are close to but not identical with the raw per-frame ones. An order-of-
        # magnitude agreement is what rules out a units or column-order mistake.
        check("q01/q99 agree with the raw memmap actions", max(d01, d99) < 0.05,
              f"max|dq01|={d01:.4f} max|dq99|={d99:.4f}")

    print("\n-- val split --")
    try:
        vds, _, vlen, _ = build(data_root, train=False)
        mm_test = json.load(open(mm_root / "test" / "meta.json"))
        want = {t["length"] for t in mm_test["trajectories"]}
        mm_train = json.load(open(mm_root / "train" / "meta.json"))
        train_lens = {t["length"] for t in mm_train["trajectories"]}
        # The pipeline repeats forever, so counting episodes is meaningless -- take a few times
        # more than the split holds and compare the SET of lengths. Episode length is effectively
        # a fingerprint here (they range over hundreds of frames and rarely collide).
        seen = set()
        for i, e in enumerate(vds.as_numpy_iterator()):
            seen.add(e["action"].shape[0])
            if i >= 4 * len(want):
                break
        check("val holds exactly the memmap's held-out episodes",
              seen == want, f"rlds={sorted(seen)} memmap={sorted(want)}")
        leaked = seen & (train_lens - want)
        check("no train episode leaked into val", not leaked, f"overlap={sorted(leaked)}")
    except Exception as e:
        check("val split loads", False, f"{type(e).__name__}: {str(e)[:120]}")

    print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILED: {', '.join(fails)}"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
