from pathlib import Path
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(
    description="Build the global multistep sample index."
)

parser.add_argument(
    "--data-root",
    type=Path,
    required=True,
    help="Root directory containing multistep_world_model/.",
)

args = parser.parse_args()

DATA_ROOT = args.data_root.resolve()

MULTISTEP = DATA_ROOT / "multistep_world_model"
OUT = DATA_ROOT / "multistep_index"
OUT.mkdir(parents=True, exist_ok=True)

scene_dirs = sorted([p for p in MULTISTEP.iterdir() if p.is_dir()])

all_samples = []
scene_counts = {}

for i, scene in enumerate(scene_dirs, 1):

    actions = np.load(scene / "actions.npy")
    tcp_hist = np.load(scene / "tcp_histories.npy")
    rgb_hist = np.load(scene / "rgb_histories.npy", allow_pickle=True)

    target_tcp = np.load(scene / "target_tcp.npy")
    target_rgb = np.load(scene / "target_rgb.npy", allow_pickle=True)

    n = len(actions)

    assert len(tcp_hist) == n
    assert len(rgb_hist) == n
    assert len(target_tcp) == n
    assert len(target_rgb) == n

    scene_counts[scene.name] = n

    for j in range(n):

        all_samples.append({
            "scene": scene.name,
            "scene_index": i - 1,
            "sample_index": j,

            "rgb_history": rgb_hist[j].tolist(),
            "target_rgb": target_rgb[j].tolist(),

            "tcp_history": tcp_hist[j].astype(np.float32),
            "actions": actions[j].astype(np.float32),
            "target_tcp": target_tcp[j].astype(np.float32),
        })

    if i % 50 == 0:
        print(f"Processed {i}/{len(scene_dirs)} scenes")

out = OUT / "samples.npy"

np.save(
    out,
    np.array(all_samples, dtype=object),
    allow_pickle=True
)

summary = {
    "scenes": len(scene_dirs),
    "samples": len(all_samples),
    "history_length": 16,
    "prediction_horizon": 10,
    "action_dim": 6,
    "tcp_dim": 7,
}

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nDONE")
print("Scenes :", len(scene_dirs))
print("Samples:", len(all_samples))
print("Output :", out)