"""Convert an ssmpolicy memmap dataset to the RLDS/TFDS shards MemoryVLA's loader reads.

Source is `artifacts/memmap/<task>/`, NOT the LeRobot v3 dataset it was built from. That is
deliberate: the memmap is what the SSM policy trains and is scored on, so building the baseline's
input from the same bytes makes the two runs share their images, their actions, AND their
train/test episode split exactly. Going back to the LeRobot source would mean re-deriving the
split (`random.Random(42).shuffle(...)` in scripts/build_lerobot_franka_memmap.py) and re-decoding
the AV1 video, with two chances to silently disagree.

Layout written:  <out>/<task>/franka_lerobot/1.0.0/...
`franka_lerobot` is the OXE registry key added in vla/datasets/rlds/oxe/configs.py; one dataset
name per task keeps a single registry entry, and the task lives in the directory above it, which
is what `--data_root_dir` points at.

Splits: memmap `train` -> RLDS `train`, memmap `test` -> RLDS `val`. dataset.py falls back to
slicing `train[:95%]` when no `val` split exists, so writing `val` explicitly is what keeps the
held-out episodes actually held out.

    python jsc/memmap_to_rlds.py \
        --memmap ../../artifacts/memmap/sponge_plate_mem \
        --out    ../../artifacts/rlds
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

# tfds pulls in tensorflow, which grabs the whole 96 GB of HBM at import unless told otherwise.
# Nothing here needs a GPU.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402

tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds  # noqa: E402

IMG = 224
N_JOINTS = 7

# Set by main() before the builder is instantiated. tfds builders take no constructor args
# beyond `data_dir`, and subclassing per task would put the task name in the dataset name.
_SRC: Path = None
_CAMERAS: dict = {}


def _load_split(root: Path, split: str):
    """Open one memmap split read-only. Returns None for a split with no frames."""
    d = root / split
    meta = json.load(open(d / "meta.json"))
    n = meta["total_frames"]
    if n == 0:
        return None
    arrs = {
        "obs": np.memmap(d / "observations.dat", dtype="float32", mode="r",
                         shape=(n, meta["observation_dim"])),
        "act": np.memmap(d / "actions.dat", dtype="float32", mode="r",
                         shape=(n, meta["action_dim"])),
    }
    for slot, fname in _CAMERAS.items():
        p = d / fname
        if p.exists():
            arrs[slot] = np.memmap(p, dtype="uint8", mode="r", shape=(n, IMG, IMG, 3))
    return meta, arrs


class FrankaLerobot(tfds.core.GeneratorBasedBuilder):
    """Single-arm Franka Panda, 8-dim absolute joint targets, from a real-robot teleop dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Built from an ssmpolicy memmap split."}

    @classmethod
    def _get_pkg_dir_path(cls):
        # tfds looks for README.md / CITATIONS.bib next to the module that defines the builder.
        # For a builder defined in a script (module `__main__`) that path is the .py file itself,
        # and metadata loading dies in iterdir() with NotADirectoryError. Point it at the
        # containing directory; _read_files ignores every name it does not recognise.
        from etils import epath
        return epath.Path(__file__).parent

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        # PNG, not JPEG: the memmap frames are already one generation of AV1 loss
                        # away from the camera, and a second lossy pass would put the baseline on
                        # different pixels than the policy it is being compared against.
                        "image": tfds.features.Image(
                            shape=(IMG, IMG, 3), dtype=np.uint8, encoding_format="png"),
                        "wrist_image": tfds.features.Image(
                            shape=(IMG, IMG, 3), dtype=np.uint8, encoding_format="png"),
                        # Split 7+1 rather than one 8-vector because dataset.py builds `proprio`
                        # by concatenating `state_obs_keys` along axis 1.
                        "joint_state": tfds.features.Tensor(shape=(N_JOINTS,), dtype=np.float32),
                        "gripper_state": tfds.features.Tensor(shape=(1,), dtype=np.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(N_JOINTS + 1,), dtype=np.float32),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                    # start_idx ties an RLDS episode back to its rows in the memmap, so an
                    # open-loop run can be checked against the exact frames it trained beside.
                    "memmap_split": tfds.features.Text(),
                    "memmap_start_idx": tfds.features.Scalar(dtype=np.int64),
                }),
            }),
        )

    def _split_generators(self, dl_manager):
        out = {}
        for mm_split, rlds_split in (("train", "train"), ("test", "val")):
            if _load_split(_SRC, mm_split) is not None:
                out[rlds_split] = self._generate_examples(mm_split)
        return out

    def _generate_examples(self, mm_split):
        loaded = _load_split(_SRC, mm_split)
        if loaded is None:
            return
        meta, arrs = loaded
        for ep_i, tr in enumerate(meta["trajectories"]):
            a, n = tr["start_idx"], tr["length"]
            obs, act = np.asarray(arrs["obs"][a:a + n]), np.asarray(arrs["act"][a:a + n])
            imgs = {s: np.asarray(arrs[s][a:a + n]) for s in _CAMERAS if s in arrs}
            steps = []
            for i in range(n):
                steps.append({
                    "observation": {
                        "image": imgs["front"][i],
                        # `wrist_image` is written but never loaded: MemoryVLA takes a single
                        # camera (image_obs_keys.wrist is None in the OXE config). It is kept so
                        # a two-camera variant does not need the dataset rebuilt.
                        "wrist_image": imgs["wrist"][i] if "wrist" in imgs
                        else np.zeros((IMG, IMG, 3), np.uint8),
                        "joint_state": obs[i, :N_JOINTS].astype(np.float32),
                        "gripper_state": obs[i, N_JOINTS:N_JOINTS + 1].astype(np.float32),
                    },
                    "action": act[i].astype(np.float32),
                    "discount": np.float32(1.0),
                    "reward": np.float32(float(i == n - 1)),
                    "is_first": i == 0,
                    "is_last": i == n - 1,
                    "is_terminal": i == n - 1,
                    "language_instruction": tr["instruction"],
                })
            key = f"{mm_split}_{ep_i:04d}"
            yield key, {
                "steps": steps,
                "episode_metadata": {
                    "file_path": f"{_SRC.name}/{mm_split}/{ep_i}",
                    "memmap_split": mm_split,
                    "memmap_start_idx": np.int64(a),
                },
            }


def main():
    global _SRC, _CAMERAS
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap", required=True, help="artifacts/memmap/<task>")
    ap.add_argument("--out", required=True, help="RLDS root; shards land in <out>/<task>/")
    ap.add_argument("--front", default="images_front_rgb.dat",
                    help="memmap file wired to observation/image (the camera the policy sees)")
    ap.add_argument("--wrist", default="images_wrist_rgb.dat")
    a = ap.parse_args()

    _SRC = Path(a.memmap).resolve()
    _CAMERAS = {"front": a.front, "wrist": a.wrist}
    task = _SRC.name
    data_dir = Path(a.out).resolve() / task
    data_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        loaded = _load_split(_SRC, split)
        if loaded is None:
            print(f"  {split}: empty, skipped")
            continue
        meta, _ = loaded
        print(f"  {split}: {len(meta['trajectories'])} episodes, {meta['total_frames']} frames, "
              f"obs_dim={meta['observation_dim']} act_dim={meta['action_dim']}")
        if (meta["observation_dim"], meta["action_dim"]) != (N_JOINTS + 1, N_JOINTS + 1):
            raise SystemExit("this converter is single-arm 8-dim only; "
                             f"got obs={meta['observation_dim']} act={meta['action_dim']}")

    builder = FrankaLerobot(data_dir=str(data_dir))
    builder.download_and_prepare()
    print(f"wrote {data_dir}/franka_lerobot/1.0.0")
    print(f"  --data_root_dir {data_dir}   --data_mix franka_lerobot")


if __name__ == "__main__":
    main()
