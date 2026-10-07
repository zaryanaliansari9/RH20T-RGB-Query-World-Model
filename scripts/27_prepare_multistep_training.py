from pathlib import Path
import numpy as np
import json

import argparse

parser = argparse.ArgumentParser(
    description="Prepare train/validation/test splits and normalization statistics."
)

parser.add_argument(
    "--data-root",
    type=Path,
    required=True,
    help="Root directory containing multistep_index/.",
)

args = parser.parse_args()

DATA_ROOT = args.data_root.resolve()

INDEX = (
    DATA_ROOT
    / "multistep_index"
    / "samples.npy"
)

MULTISTEP = (
    DATA_ROOT
    / "multistep_world_model"
)

OUT = (
    DATA_ROOT
    / "multistep_training"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42

# ------------------------------------------------------------
# LOAD INDEX
# ------------------------------------------------------------

data = np.load(INDEX, allow_pickle=True)

print("Total samples:", len(data))

# ------------------------------------------------------------
# UNIQUE SCENES
# ------------------------------------------------------------

scenes = sorted(set(x["scene"] for x in data))

print("Total scenes :", len(scenes))

rng = np.random.default_rng(SEED)
rng.shuffle(scenes)

n = len(scenes)

n_train = int(0.80 * n)
n_val = int(0.10 * n)

train_scenes = scenes[:n_train]
val_scenes = scenes[n_train:n_train + n_val]
test_scenes = scenes[n_train + n_val:]

train_set = set(train_scenes)
val_set = set(val_scenes)
test_set = set(test_scenes)

print("\nScene split:")
print("Train:", len(train_scenes))
print("Val  :", len(val_scenes))
print("Test :", len(test_scenes))

# ------------------------------------------------------------
# SAMPLE INDICES
# ------------------------------------------------------------

train_idx = []
val_idx = []
test_idx = []

for i, sample in enumerate(data):

    scene = sample["scene"]

    if scene in train_set:
        train_idx.append(i)

    elif scene in val_set:
        val_idx.append(i)

    elif scene in test_set:
        test_idx.append(i)

np.save(
    OUT / "train_indices.npy",
    np.asarray(train_idx, dtype=np.int64)
)

np.save(
    OUT / "val_indices.npy",
    np.asarray(val_idx, dtype=np.int64)
)

np.save(
    OUT / "test_indices.npy",
    np.asarray(test_idx, dtype=np.int64)
)

print("\nSample split:")
print("Train:", len(train_idx))
print("Val  :", len(val_idx))
print("Test :", len(test_idx))

# ------------------------------------------------------------
# NORMALIZATION STATISTICS
# ------------------------------------------------------------
#
# IMPORTANT:
# Statistics are computed ONLY from the training scenes.
#
# This prevents validation/test information leaking into
# training.
# ------------------------------------------------------------

print("\nComputing normalization statistics...")

action_sum = np.zeros(6, dtype=np.float64)
action_sq_sum = np.zeros(6, dtype=np.float64)

tcp_sum = np.zeros(7, dtype=np.float64)
tcp_sq_sum = np.zeros(7, dtype=np.float64)

count_action = 0
count_tcp = 0

for idx in train_idx:

    sample = data[idx]

    actions = np.asarray(
        sample["actions"],
        dtype=np.float64
    )

    tcp = np.asarray(
        sample["tcp_history"],
        dtype=np.float64
    )

    # actions: (10, 6)
    action_sum += actions.sum(axis=0)
    action_sq_sum += (actions ** 2).sum(axis=0)
    count_action += actions.shape[0]

    # tcp: (16, 7)
    tcp_sum += tcp.sum(axis=0)
    tcp_sq_sum += (tcp ** 2).sum(axis=0)
    count_tcp += tcp.shape[0]

action_mean = action_sum / count_action
action_var = action_sq_sum / count_action - action_mean ** 2
action_std = np.sqrt(np.maximum(action_var, 1e-12))

tcp_mean = tcp_sum / count_tcp
tcp_var = tcp_sq_sum / count_tcp - tcp_mean ** 2
tcp_std = np.sqrt(np.maximum(tcp_var, 1e-12))

np.save(OUT / "action_mean.npy", action_mean.astype(np.float32))
np.save(OUT / "action_std.npy", action_std.astype(np.float32))

np.save(OUT / "tcp_mean.npy", tcp_mean.astype(np.float32))
np.save(OUT / "tcp_std.npy", tcp_std.astype(np.float32))

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

summary = {
    "total_scenes": len(scenes),

    "train_scenes": len(train_scenes),
    "val_scenes": len(val_scenes),
    "test_scenes": len(test_scenes),

    "total_samples": len(data),
    "train_samples": len(train_idx),
    "val_samples": len(val_idx),
    "test_samples": len(test_idx),

    "history_length": 16,
    "prediction_horizon": 10,
    "action_dim": 6,
    "tcp_dim": 7,

    "seed": SEED,

    "action_mean": action_mean.tolist(),
    "action_std": action_std.tolist(),

    "tcp_mean": tcp_mean.tolist(),
    "tcp_std": tcp_std.tolist(),
}

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nNormalization:")
print("Action mean:", action_mean)
print("Action std :", action_std)

print("\nTCP mean:", tcp_mean)
print("TCP std :", tcp_std)

print("\nDONE")
print("Output:", OUT)