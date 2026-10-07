#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import json
import argparse


# ============================================================
# CONFIGURATION
# ============================================================
parser = argparse.ArgumentParser(
    description="Build clean aligned RH20T scenes."
)

parser.add_argument(
    "--rh20t-project",
    type=Path,
    required=True,
    help="Path to the RH20T_project directory.",
)

parser.add_argument(
    "--rgb-dir",
    type=Path,
    required=True,
    help="Path to processed RGB frames.",
)

parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="Output directory for clean_aligned scenes.",
)

args = parser.parse_args()

RH20T_PROJECT = args.rh20t_project.resolve()

ALIGNED_DIR = (
    RH20T_PROJECT
    / "dataset"
    / "aligned"
)

TRANSITIONS_DIR = (
    RH20T_PROJECT
    / "dataset"
    / "transitions"
)

RGB_DIR = args.rgb_dir.resolve()

OUTPUT_DIR = args.output_dir.resolve()

HISTORY_LENGTH = 16


# ============================================================
# HELPERS
# ============================================================

def scene_name_from_aligned(path):
    """
    task_xxx_user_xxx_scene_xxx_cfg_xxx_aligned.npy
    ->
    task_xxx_user_xxx_scene_xxx_cfg_xxx
    """

    return path.name.replace("_aligned.npy", "")


def scene_name_from_transition(path):
    """
    task_xxx_user_xxx_scene_xxx_cfg_xxx_transitions.npy
    ->
    task_xxx_user_xxx_scene_xxx_cfg_xxx
    """

    return path.name.replace("_transitions.npy", "")


def load_object_array(path):
    return np.load(path, allow_pickle=True)


def make_absolute_rgb_path(scene, image_path):
    """
    aligned.npy contains image_path.

    Make sure the resulting path points to the RGB frame
    inside dataset/processed/rgb_frames.
    """

    p = Path(str(image_path))

    # Already absolute
    if p.is_absolute() and p.exists():
        return str(p)

    # Directly relative to RGB scene directory
    candidate = RGB_DIR / scene / p

    if candidate.exists():
        return str(candidate)

    # If image_path already contains the scene directory
    candidate = RGB_DIR / p

    if candidate.exists():
        return str(candidate)

    # Search by filename as a final fallback
    matches = list((RGB_DIR / scene).rglob(p.name))

    if matches:
        return str(matches[0])

    return ""


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BUILDING CLEAN ALIGNED DATASET FOR ALL RH20T SCENES")
    print("=" * 70)
    print()

    print("Aligned input     :", ALIGNED_DIR)
    print("Transitions input :", TRANSITIONS_DIR)
    print("RGB input         :", RGB_DIR)
    print("Output            :", OUTPUT_DIR)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # FIND FILES
    # --------------------------------------------------------

    aligned_files = {
        scene_name_from_aligned(p): p
        for p in ALIGNED_DIR.glob("*_aligned.npy")
    }

    transition_files = {
        scene_name_from_transition(p): p
        for p in TRANSITIONS_DIR.glob("*_transitions.npy")
    }

    rgb_scenes = {
        p.name
        for p in RGB_DIR.iterdir()
        if p.is_dir()
    }

    print("Aligned scenes    :", len(aligned_files))
    print("Transition scenes :", len(transition_files))
    print("RGB scenes        :", len(rgb_scenes))
    print()

    # --------------------------------------------------------
    # ONLY USE SCENES WITH ALL THREE COMPONENTS
    # --------------------------------------------------------

    scenes = sorted(
        set(aligned_files)
        & set(transition_files)
        & rgb_scenes
    )

    print("Complete scenes   :", len(scenes))
    print()

    if not scenes:
        print("ERROR: No complete scenes found.")
        return

    # --------------------------------------------------------
    # GLOBAL STATISTICS
    # --------------------------------------------------------

    valid_scenes = 0
    skipped_scenes = 0

    total_samples = 0
    total_aligned = 0
    total_transitions = 0

    skipped_reasons = {}

    # --------------------------------------------------------
    # PROCESS EACH SCENE
    # --------------------------------------------------------

    for scene_idx, scene in enumerate(scenes, start=1):

        print(
            f"[{scene_idx:04d}/{len(scenes):04d}] "
            f"{scene}",
            end=" ... ",
            flush=True
        )

        try:

            aligned_path = aligned_files[scene]
            transition_path = transition_files[scene]

            aligned = load_object_array(aligned_path)
            transitions = load_object_array(transition_path)

            n_aligned = len(aligned)
            n_transitions = len(transitions)

            total_aligned += n_aligned
            total_transitions += n_transitions

            # ------------------------------------------------
            # BASIC VALIDATION
            # ------------------------------------------------

            if n_aligned < HISTORY_LENGTH + 1:
                skipped_scenes += 1
                reason = "too_few_aligned_samples"
                skipped_reasons[reason] = (
                    skipped_reasons.get(reason, 0) + 1
                )

                print(f"SKIP ({reason})")
                continue

            if n_transitions < HISTORY_LENGTH:
                skipped_scenes += 1
                reason = "too_few_transitions"
                skipped_reasons[reason] = (
                    skipped_reasons.get(reason, 0) + 1
                )

                print(f"SKIP ({reason})")
                continue

            # ------------------------------------------------
            # EXTRACT TCP
            # ------------------------------------------------

            tcp = []
            rgb_paths = []
            timestamps = []

            valid = True

            for record in aligned:

                # robot_window should be (8, 13)
                robot_window = np.asarray(
                    record["robot_window"],
                    dtype=np.float32
                )

                if robot_window.ndim != 2:
                    valid = False
                    break

                if robot_window.shape[1] < 7:
                    valid = False
                    break

                # ------------------------------------------------
                # TCP = XYZ + QUATERNION
                # ------------------------------------------------
                #
                # robot_window:
                #
                # [x y z qx qy qz qw ...]
                #
                # We use the LAST robot state in the window.
                #
                tcp_value = robot_window[-1, :7]

                tcp.append(tcp_value)

                # ------------------------------------------------
                # RGB
                # ------------------------------------------------

                image_path = make_absolute_rgb_path(
                    scene,
                    record["image_path"]
                )

                rgb_paths.append(image_path)

                # ------------------------------------------------
                # TIMESTAMP
                # ------------------------------------------------

                timestamps.append(
                    float(record["timestamp"])
                )

            if not valid:
                skipped_scenes += 1
                reason = "invalid_robot_window"

                skipped_reasons[reason] = (
                    skipped_reasons.get(reason, 0) + 1
                )

                print(f"SKIP ({reason})")
                continue

            tcp = np.asarray(tcp, dtype=np.float32)
            rgb_paths = np.asarray(rgb_paths)
            timestamps = np.asarray(timestamps, dtype=np.float64)

            # ------------------------------------------------
            # EXTRACT ACTIONS
            # ------------------------------------------------

            actions = []

            for record in transitions:

                action = np.asarray(
                    record["action"],
                    dtype=np.float32
                ).reshape(-1)

                if action.shape[0] < 6:
                    valid = False
                    break

                # World model currently expects 6-DoF action:
                #
                # dx dy dz
                # d_rx d_ry d_rz
                #
                # Gripper is deliberately excluded.
                actions.append(action[:6])

            if not valid:
                skipped_scenes += 1
                reason = "invalid_action"

                skipped_reasons[reason] = (
                    skipped_reasons.get(reason, 0) + 1
                )

                print(f"SKIP ({reason})")
                continue

            actions = np.asarray(
                actions,
                dtype=np.float32
            )

            # ------------------------------------------------
            # ALIGN LENGTHS
            # ------------------------------------------------
            #
            # Normally:
            #
            # aligned      = 164
            # transitions  = 163
            #
            # because transition t describes:
            #
            # state[t] -> state[t+1]
            #
            # Therefore actions have one fewer sample.
            # ------------------------------------------------

            usable = min(
                len(tcp),
                len(rgb_paths),
                len(actions) + 1
            )

            if usable <= HISTORY_LENGTH:
                skipped_scenes += 1
                reason = "insufficient_usable_samples"

                skipped_reasons[reason] = (
                    skipped_reasons.get(reason, 0) + 1
                )

                print(f"SKIP ({reason})")
                continue

            tcp = tcp[:usable]
            rgb_paths = rgb_paths[:usable]
            timestamps = timestamps[:usable]

            actions = actions[:usable - 1]

            # ------------------------------------------------
            # CHECK RGB FILES
            # ------------------------------------------------

            existing_rgb = np.array(
                [
                    bool(p) and Path(p).exists()
                    for p in rgb_paths
                ],
                dtype=bool
            )

            rgb_count = int(existing_rgb.sum())

            if rgb_count == 0:
                skipped_scenes += 1
                reason = "no_rgb_files_found"

                skipped_reasons[reason] = (
                    skipped_reasons.get(reason, 0) + 1
                )

                print(f"SKIP ({reason})")
                continue

            # ------------------------------------------------
            # CREATE OUTPUT
            # ------------------------------------------------

            scene_output = OUTPUT_DIR / scene
            scene_output.mkdir(
                parents=True,
                exist_ok=True
            )

            np.save(
                scene_output / "rgb_paths.npy",
                rgb_paths
            )

            np.save(
                scene_output / "tcp.npy",
                tcp
            )

            np.save(
                scene_output / "actions.npy",
                actions
            )

            np.save(
                scene_output / "rgb_timestamps.npy",
                timestamps
            )

            np.save(
                scene_output / "action_timestamps.npy",
                timestamps[:-1]
            )

            # Action time differences
            if len(timestamps) >= 2:

                action_dt = np.diff(timestamps)

                np.save(
                    scene_output / "action_time_differences.npy",
                    action_dt
                )

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            summary = {
                "scene": scene,
                "aligned_samples": int(n_aligned),
                "transition_samples": int(n_transitions),
                "usable_samples": int(usable),
                "rgb_samples": int(len(rgb_paths)),
                "rgb_files_found": int(rgb_count),
                "tcp_shape": list(tcp.shape),
                "actions_shape": list(actions.shape),
                "history_length": HISTORY_LENGTH,
            }

            with open(
                scene_output / "summary.json",
                "w"
            ) as f:

                json.dump(
                    summary,
                    f,
                    indent=2
                )

            valid_scenes += 1

            # For history-based models:
            possible_samples = usable - HISTORY_LENGTH

            total_samples += max(
                0,
                possible_samples
            )

            print(
                f"OK "
                f"(aligned={n_aligned}, "
                f"transitions={n_transitions}, "
                f"usable={usable}, "
                f"samples≈{possible_samples})"
            )

        except Exception as e:

            skipped_scenes += 1

            reason = type(e).__name__

            skipped_reasons[reason] = (
                skipped_reasons.get(reason, 0) + 1
            )

            print(
                f"ERROR: {type(e).__name__}: {e}"
            )

    # --------------------------------------------------------
    # GLOBAL SUMMARY
    # --------------------------------------------------------

    summary = {
        "aligned_scenes_found": len(aligned_files),
        "transition_scenes_found": len(transition_files),
        "rgb_scenes_found": len(rgb_scenes),
        "complete_scenes": len(scenes),
        "valid_scenes": valid_scenes,
        "skipped_scenes": skipped_scenes,
        "total_aligned_samples": total_aligned,
        "total_transition_samples": total_transitions,
        "approx_history_samples": total_samples,
        "history_length": HISTORY_LENGTH,
        "skipped_reasons": skipped_reasons,
    }

    summary_path = OUTPUT_DIR / "all_scenes_summary.json"

    with open(summary_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print()
    print("Complete scenes :", len(scenes))
    print("Valid scenes    :", valid_scenes)
    print("Skipped scenes  :", skipped_scenes)
    print("Aligned samples :", total_aligned)
    print("Transitions     :", total_transitions)
    print()
    print("Summary:")
    print(summary_path)
    print()


if __name__ == "__main__":
    main()