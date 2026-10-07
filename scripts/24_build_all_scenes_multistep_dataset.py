import os
import json
import numpy as np
from pathlib import Path
import argparse



# ============================================================
# CONFIGURATION
# ============================================================
parser = argparse.ArgumentParser(
    description="Build the multistep world-model dataset."
)

parser.add_argument(
    "--data-root",
    type=Path,
    required=True,
    help="Root directory containing clean_aligned/.",
)

args = parser.parse_args()

DATA_ROOT = args.data_root.resolve()

CLEAN_ALIGNED_DIR = (
    DATA_ROOT / "clean_aligned"
)

OUTPUT_ROOT = (
    DATA_ROOT / "multistep_world_model"
)

HISTORY_LENGTH = 16
PREDICTION_HORIZON = 10


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print("BUILDING MULTI-STEP DATASET FOR ALL SCENES")
print("=" * 70)

print()
print("Input:")
print(CLEAN_ALIGNED_DIR)

print()
print("Output:")
print(OUTPUT_ROOT)

print()
print("History length     :", HISTORY_LENGTH)
print("Prediction horizon :", PREDICTION_HORIZON)


# ============================================================
# FIND SCENES
# ============================================================

if not CLEAN_ALIGNED_DIR.exists():

    raise FileNotFoundError(
        f"Clean aligned directory does not exist:\n"
        f"{CLEAN_ALIGNED_DIR}"
    )


scene_dirs = sorted(
    [
        p
        for p in CLEAN_ALIGNED_DIR.iterdir()
        if p.is_dir()
    ]
)


print()
print("Scenes found :", len(scene_dirs))


if len(scene_dirs) == 0:

    raise RuntimeError(
        "No scene directories found."
    )


# ============================================================
# GLOBAL SUMMARY
# ============================================================

scene_summaries = []

total_training_samples = 0
valid_scene_count = 0
skipped_scene_count = 0


# ============================================================
# PROCESS EACH SCENE
# ============================================================

for scene_index, scene_dir in enumerate(scene_dirs):

    scene_name = scene_dir.name

    print()
    print("=" * 70)
    print(
        f"SCENE {scene_index + 1}/{len(scene_dirs)}"
    )
    print("=" * 70)

    print()
    print("Scene :", scene_name)


    # --------------------------------------------------------
    # INPUT FILES
    # --------------------------------------------------------

    required_files = [

        "rgb_paths.npy",

        "rgb_timestamps.npy",

        "tcp.npy",

        "tcp_timestamps.npy",

        "actions.npy",

        "action_timestamps.npy",

    ]


    missing_files = [

        name
        for name in required_files
        if not (scene_dir / name).exists()
    ]


    if missing_files:

        print()
        print("SKIPPED")

        print(
            "Missing files:",
            missing_files
        )

        skipped_scene_count += 1

        scene_summaries.append({

            "scene": scene_name,

            "status": "skipped",

            "reason": "missing_files",

            "missing_files": missing_files

        })

        continue


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    try:

        rgb_paths = np.load(
            scene_dir / "rgb_paths.npy",
            allow_pickle=True
        )

        rgb_timestamps = np.load(
            scene_dir / "rgb_timestamps.npy"
        )

        tcp = np.load(
            scene_dir / "tcp.npy"
        ).astype(
            np.float32
        )

        tcp_timestamps = np.load(
            scene_dir / "tcp_timestamps.npy"
        )

        actions = np.load(
            scene_dir / "actions.npy"
        ).astype(
            np.float32
        )

        action_timestamps = np.load(
            scene_dir / "action_timestamps.npy"
        )


    except Exception as e:

        print()
        print("SKIPPED")

        print(
            "Could not load scene:",
            e
        )

        skipped_scene_count += 1

        scene_summaries.append({

            "scene": scene_name,

            "status": "skipped",

            "reason": str(e)

        })

        continue


    # --------------------------------------------------------
    # BASIC SHAPE CHECK
    # --------------------------------------------------------

    rgb_count = len(
        rgb_paths
    )

    tcp_count = len(
        tcp
    )

    action_count = len(
        actions
    )


    print()
    print("RGB samples    :", rgb_count)
    print("TCP samples    :", tcp_count)
    print("Action samples :", action_count)


    # --------------------------------------------------------
    # ALIGNMENT CHECK
    # --------------------------------------------------------

    if rgb_count != tcp_count:

        print()
        print("SKIPPED")

        print(
            "RGB/TCP sample count mismatch."
        )

        skipped_scene_count += 1

        scene_summaries.append({

            "scene": scene_name,

            "status": "skipped",

            "reason": "rgb_tcp_count_mismatch",

            "rgb_count": rgb_count,

            "tcp_count": tcp_count

        })

        continue


    # --------------------------------------------------------
    # DETERMINE AVAILABLE SAMPLES
    # --------------------------------------------------------

    total_samples = min(
        rgb_count,
        tcp_count,
        action_count
    )


    # --------------------------------------------------------
    # REQUIRED NUMBER OF SAMPLES
    #
    # History:
    #
    #   t-15 ... t
    #
    # Actions:
    #
    #   a_t ... a_(t+9)
    #
    # Targets:
    #
    #   o_(t+1) ... o_(t+10)
    #
    # Therefore:
    #
    # N - HISTORY_LENGTH - HORIZON + 1
    # --------------------------------------------------------

    num_training_samples = (
        total_samples
        - HISTORY_LENGTH
        - PREDICTION_HORIZON
        + 1
    )


    print(
        "Training samples:",
        max(
            0,
            num_training_samples
        )
    )


    if num_training_samples <= 0:

        print()
        print("SKIPPED")

        print(
            "Scene is too short."
        )

        skipped_scene_count += 1

        scene_summaries.append({

            "scene": scene_name,

            "status": "skipped",

            "reason": "scene_too_short",

            "total_samples": total_samples

        })

        continue


    # ========================================================
    # BUILD ARRAYS
    # ========================================================

    rgb_histories = []

    tcp_histories = []

    action_sequences = []

    target_rgb_sequences = []

    target_tcp_sequences = []

    observation_timestamp_sequences = []

    action_timestamp_sequences = []

    target_timestamp_sequences = []


    # --------------------------------------------------------
    # SLIDING WINDOW
    # --------------------------------------------------------

    for i in range(
        num_training_samples
    ):

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        history_start = i

        history_end = (
            i
            + HISTORY_LENGTH
        )


        # ----------------------------------------------------
        # ACTIONS
        #
        # Action sequence starts at the LAST
        # observation in the history.
        #
        # For:
        #
        # history = frames 34 ... 49
        #
        # actions = action at 49 ... action at 58
        # ----------------------------------------------------

        action_start = (
            i
            + HISTORY_LENGTH
            - 1
        )

        action_end = (
            action_start
            + PREDICTION_HORIZON
        )


        # ----------------------------------------------------
        # TARGETS
        #
        # First target is the observation immediately
        # following the last history frame.
        #
        # history = 34 ... 49
        #
        # targets = 50 ... 59
        # ----------------------------------------------------

        target_start = (
            i
            + HISTORY_LENGTH
        )

        target_end = (
            target_start
            + PREDICTION_HORIZON
        )


        # ----------------------------------------------------
        # STORE RGB HISTORY
        # ----------------------------------------------------

        rgb_histories.append(
            np.asarray(
                rgb_paths[
                    history_start:history_end
                ]
            )
        )


        # ----------------------------------------------------
        # STORE TCP HISTORY
        # ----------------------------------------------------

        tcp_histories.append(
            tcp[
                history_start:history_end
            ]
        )


        # ----------------------------------------------------
        # STORE ACTIONS
        # ----------------------------------------------------

        action_sequences.append(
            actions[
                action_start:action_end
            ]
        )


        # ----------------------------------------------------
        # STORE TARGET RGB
        # ----------------------------------------------------

        target_rgb_sequences.append(
            np.asarray(
                rgb_paths[
                    target_start:target_end
                ]
            )
        )


        # ----------------------------------------------------
        # STORE TARGET TCP
        # ----------------------------------------------------

        target_tcp_sequences.append(
            tcp[
                target_start:target_end
            ]
        )


        # ----------------------------------------------------
        # TIMESTAMPS
        # ----------------------------------------------------

        observation_timestamp_sequences.append(
            rgb_timestamps[
                history_start:history_end
            ]
        )


        action_timestamp_sequences.append(
            action_timestamps[
                action_start:action_end
            ]
        )


        target_timestamp_sequences.append(
            rgb_timestamps[
                target_start:target_end
            ]
        )


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    rgb_histories = np.asarray(
        rgb_histories,
        dtype=object
    )

    tcp_histories = np.asarray(
        tcp_histories,
        dtype=np.float32
    )

    action_sequences = np.asarray(
        action_sequences,
        dtype=np.float32
    )

    target_rgb_sequences = np.asarray(
        target_rgb_sequences,
        dtype=object
    )

    target_tcp_sequences = np.asarray(
        target_tcp_sequences,
        dtype=np.float32
    )

    observation_timestamp_sequences = np.asarray(
        observation_timestamp_sequences
    )

    action_timestamp_sequences = np.asarray(
        action_timestamp_sequences
    )

    target_timestamp_sequences = np.asarray(
        target_timestamp_sequences
    )


    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = (
        OUTPUT_ROOT
        / scene_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # SAVE DATA
    # ========================================================

    np.save(
        output_dir / "rgb_histories.npy",
        rgb_histories
    )

    np.save(
        output_dir / "tcp_histories.npy",
        tcp_histories
    )

    np.save(
        output_dir / "actions.npy",
        action_sequences
    )

    np.save(
        output_dir / "target_rgb.npy",
        target_rgb_sequences
    )

    np.save(
        output_dir / "target_tcp.npy",
        target_tcp_sequences
    )

    np.save(
        output_dir / "observation_timestamps.npy",
        observation_timestamp_sequences
    )

    np.save(
        output_dir / "action_timestamps.npy",
        action_timestamp_sequences
    )

    np.save(
        output_dir / "target_timestamps.npy",
        target_timestamp_sequences
    )


    # ========================================================
    # TEMPORAL CHECK
    # ========================================================

    first_observation_time = (
        observation_timestamp_sequences[0, -1]
    )

    first_target_time = (
        target_timestamp_sequences[0, 0]
    )

    prediction_interval = (
        first_target_time
        - first_observation_time
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "scene": scene_name,

        "status": "success",

        "total_aligned_samples": int(
            total_samples
        ),

        "history_length": int(
            HISTORY_LENGTH
        ),

        "prediction_horizon": int(
            PREDICTION_HORIZON
        ),

        "num_training_samples": int(
            num_training_samples
        ),

        "rgb_history_shape":
            list(
                rgb_histories.shape
            ),

        "tcp_history_shape":
            list(
                tcp_histories.shape
            ),

        "action_shape":
            list(
                action_sequences.shape
            ),

        "target_rgb_shape":
            list(
                target_rgb_sequences.shape
            ),

        "target_tcp_shape":
            list(
                target_tcp_sequences.shape
            ),

        "first_observation":
            str(
                rgb_histories[0, 0]
            ),

        "last_observation":
            str(
                rgb_histories[0, -1]
            ),

        "first_target":
            str(
                target_rgb_sequences[0, 0]
            ),

        "last_target":
            str(
                target_rgb_sequences[0, -1]
            ),

        "first_observation_timestamp":
            int(
                observation_timestamp_sequences[0, -1]
            ),

        "first_target_timestamp":
            int(
                target_timestamp_sequences[0, 0]
            ),

        "first_prediction_interval_ms":
            int(
                prediction_interval
            )

    }


    with open(
        output_dir / "summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print()
    print("SHAPES")

    print(
        "RGB history :",
        rgb_histories.shape
    )

    print(
        "TCP history :",
        tcp_histories.shape
    )

    print(
        "Actions     :",
        action_sequences.shape
    )

    print(
        "Target RGB  :",
        target_rgb_sequences.shape
    )

    print(
        "Target TCP  :",
        target_tcp_sequences.shape
    )

    print()
    print(
        "First prediction interval:",
        prediction_interval,
        "ms"
    )

    print()
    print(
        "Saved to:"
    )

    print(
        output_dir
    )


    # ========================================================
    # GLOBAL RECORD
    # ========================================================

    valid_scene_count += 1

    total_training_samples += (
        num_training_samples
    )

    scene_summaries.append({

        "scene": scene_name,

        "status": "success",

        "total_aligned_samples":
            int(total_samples),

        "training_samples":
            int(num_training_samples),

        "history_length":
            int(HISTORY_LENGTH),

        "prediction_horizon":
            int(PREDICTION_HORIZON)

    })


# ============================================================
# GLOBAL SUMMARY
# ============================================================

global_summary = {

    "total_scenes_found":
        len(scene_dirs),

    "valid_scenes":
        valid_scene_count,

    "skipped_scenes":
        skipped_scene_count,

    "total_training_samples":
        total_training_samples,

    "history_length":
        HISTORY_LENGTH,

    "prediction_horizon":
        PREDICTION_HORIZON,

    "scenes":
        scene_summaries

}


with open(
    OUTPUT_ROOT / "all_scenes_summary.json",
    "w"
) as f:

    json.dump(
        global_summary,
        f,
        indent=4
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("ALL-SCENE MULTI-STEP DATASET COMPLETE")
print("=" * 70)

print()
print(
    "Scenes found      :",
    len(scene_dirs)
)

print(
    "Valid scenes      :",
    valid_scene_count
)

print(
    "Skipped scenes    :",
    skipped_scene_count
)

print(
    "Training samples  :",
    total_training_samples
)

print()
print("Global summary:")
print(
    OUTPUT_ROOT
    / "all_scenes_summary.json"
)

print()
print("DONE")