from pathlib import Path
import numpy as np
from PIL import Image
import json
import time
import sys
import multiprocessing as mp


# ============================================================
# PATHS
# ============================================================

import argparse

parser = argparse.ArgumentParser(
    description="Preprocess RGB images into a memory-mapped cache."
)

parser.add_argument(
    "--data-root",
    type=Path,
    required=True,
    help="Root directory containing multistep_index/.",
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=2,
    help="Number of image-processing workers.",
)

args = parser.parse_args()

DATA_ROOT = args.data_root.resolve()

SAMPLES_PATH = (
    DATA_ROOT
    / "multistep_index"
    / "samples.npy"
)

OUT_DIR = (
    DATA_ROOT
    / "preprocessed_images"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128

IMAGE_FILE = OUT_DIR / "images.dat"
INDEX_FILE = OUT_DIR / "index.json"
PROGRESS_FILE = OUT_DIR / "progress.npy"


# ============================================================
# SETTINGS
# ============================================================

# Start with 4.
# Do NOT immediately use 8 or 12 because your source appears
# to be the slower disk.
NUM_WORKERS = args.num_workers

# Number of images assigned to a worker at a time.
CHUNK_SIZE = 64

IMAGE_BYTES = (
    3
    * IMAGE_SIZE
    * IMAGE_SIZE
)


# ============================================================
# TERMINAL HELPERS
# ============================================================

SPINNER = ["|", "/", "-", "\\"]


def format_time(seconds):

    if seconds is None or seconds == float("inf"):
        return "--:--:--"

    seconds = int(seconds)

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    return f"{h:02d}:{m:02d}:{s:02d}"


def progress_bar(current, total, width=35):

    fraction = current / total

    filled = int(width * fraction)

    return (
        "#"
        * filled
        + "-"
        * (width - filled)
    )


def show_progress(
    current,
    total,
    start_time,
    current_path,
):

    elapsed = (
        time.perf_counter()
        - start_time
    )

    rate = (
        current / elapsed
        if elapsed > 0
        else 0
    )

    remaining = (
        (total - current) / rate
        if rate > 0
        else float("inf")
    )

    percent = (
        100.0
        * current
        / total
    )

    spinner = SPINNER[
        current % len(SPINNER)
    ]

    bar = progress_bar(
        current,
        total,
    )

    path = str(current_path)

    if len(path) > 55:
        path = "..." + path[-52:]

    line = (
        f"\r"
        f"{spinner} "
        f"{bar} "
        f"{percent:6.2f}% "
        f"| {current:,}/{total:,} "
        f"| {rate:7.1f} img/s "
        f"| ETA {format_time(remaining)} "
        f"| {path}"
    )

    sys.stdout.write(line)
    sys.stdout.flush()


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_path(path):

    p = Path(path)

    if p.is_absolute() and p.exists():
        return p

    candidates = [
        p,
        ROOT / p,
        ROOT / "dataset/processed/rgb_frames" / p,
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"\nCould not find image:\n{path}"
    )


# ============================================================
# WORKER
# ============================================================

def process_image(args):

    index, path = args

    p = resolve_path(path)

    with Image.open(p) as img:

        img = img.convert("RGB")

        if img.size != (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):

            img = img.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                Image.Resampling.BILINEAR,
            )

        arr = np.asarray(
            img,
            dtype=np.uint8,
        )

    # HWC -> CHW

    arr = np.transpose(
        arr,
        (2, 0, 1),
    )

    # Make contiguous because the transpose
    # produces a view.

    arr = np.ascontiguousarray(arr)

    return index, arr


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    # Required for multiprocessing safety.

    mp.set_start_method(
        "fork",
        force=True,
    )

    print("=" * 70)
    print("IMAGE PREPROCESSING")
    print("=" * 70)

    print("\nLoading samples...")

    samples = np.load(
        SAMPLES_PATH,
        allow_pickle=True,
    )

    print(
        "Samples:",
        len(samples),
    )


    # ========================================================
    # COLLECT UNIQUE PATHS
    # ========================================================

    print(
        "\nCollecting unique image paths..."
    )

    unique_paths = set()

    collect_start = (
        time.perf_counter()
    )

    for i, sample in enumerate(samples):

        for p in sample["rgb_history"]:
            unique_paths.add(str(p))

        for p in sample["target_rgb"]:
            unique_paths.add(str(p))

        if (
            i + 1
        ) % 50000 == 0:

            elapsed = (
                time.perf_counter()
                - collect_start
            )

            rate = (
                (i + 1)
                / elapsed
            )

            print(
                f"  processed samples: "
                f"{i + 1:,}/"
                f"{len(samples):,} "
                f"({rate:.1f} "
                f"samples/s)"
            )

    unique_paths = sorted(
        unique_paths
    )

    total_images = len(
        unique_paths
    )

    print(
        f"\nUnique images: "
        f"{total_images:,}"
    )


    # ========================================================
    # STORAGE
    # ========================================================

    total_bytes = (
        total_images
        * IMAGE_BYTES
    )

    total_gib = (
        total_bytes
        / (1024 ** 3)
    )

    print(
        f"Output size: "
        f"{total_gib:.2f} GiB"
    )

    print(
        f"Workers: "
        f"{NUM_WORKERS}"
    )

    # ========================================================
    # RESUME SUPPORT
    # ========================================================

    completed = np.zeros(
        total_images,
        dtype=np.bool_,
    )

    if PROGRESS_FILE.exists():

        print(
            "\nExisting progress file found."
        )

        old_progress = np.load(
            PROGRESS_FILE
        )

        if len(old_progress) == total_images:

            completed[:] = old_progress

            print(
                "Already completed:",
                f"{completed.sum():,}",
                "/",
                f"{total_images:,}",
            )

        else:

            print(
                "Progress file does not "
                "match current dataset."
            )


    # ========================================================
    # CREATE / OPEN MEMMAP
    # ========================================================

    if IMAGE_FILE.exists():

        print(
            "\nOpening existing "
            "memory-mapped image file..."
        )

        images = np.memmap(
            IMAGE_FILE,
            dtype=np.uint8,
            mode="r+",
            shape=(
                total_images,
                3,
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
        )

    else:

        print(
            "\nCreating memory-mapped "
            "image file..."
        )

        images = np.memmap(
            IMAGE_FILE,
            dtype=np.uint8,
            mode="w+",
            shape=(
                total_images,
                3,
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
        )


    # ========================================================
    # IMAGE LIST
    # ========================================================

    jobs = [
        (i, unique_paths[i])
        for i in range(total_images)
        if not completed[i]
    ]

    remaining_images = len(jobs)

    print(
        "\nImages remaining:",
        f"{remaining_images:,}",
    )

    if remaining_images == 0:

        print(
            "\nAll images already "
            "processed."
        )

    else:

        print("\n")
        print("=" * 70)
        print(
            "LOADING / RESIZING / "
            "WRITING IMAGES"
        )
        print("=" * 70)
        print()


        # ====================================================
        # MULTIPROCESSING
        # ====================================================

        start_time = (
            time.perf_counter()
        )

        last_print = start_time

        processed = 0

        last_rate_time = start_time
        last_rate_processed = 0

        with mp.Pool(
            processes=NUM_WORKERS
        ) as pool:

            iterator = pool.imap_unordered(
                process_image,
                jobs,
                chunksize=CHUNK_SIZE,
            )

            for index, arr in iterator:

                images[index] = arr

                completed[index] = True

                processed += 1

                now = time.perf_counter()

                if now - last_print >= 1.0:

                    elapsed = now - start_time

                    avg_rate = (
                        processed / elapsed
                        if elapsed > 0
                        else 0
                    )

                    recent_elapsed = now - last_rate_time

                    recent_processed = (
                        processed - last_rate_processed
                    )

                    recent_rate = (
                        recent_processed / recent_elapsed
                        if recent_elapsed > 0
                        else 0
                    )

                    remaining = (
                        remaining_images - processed
                    )

                    eta = (
                        remaining / recent_rate
                        if recent_rate > 0
                        else float("inf")
                    )

                    percent = (
                        100.0
                        * processed
                        / remaining_images
                    )

                    spinner = SPINNER[
                        processed % len(SPINNER)
                    ]

                    bar = progress_bar(
                        processed,
                        remaining_images,
                    )

                    path = str(
                        unique_paths[index]
                    )

                    if len(path) > 45:
                        path = "..." + path[-42:]

                    line = (
                        f"\r"
                        f"{spinner} "
                        f"{bar} "
                        f"{percent:6.2f}% "
                        f"| {processed:,}/{remaining_images:,} "
                        f"| recent {recent_rate:6.1f} img/s "
                        f"| avg {avg_rate:6.1f} img/s "
                        f"| ETA {format_time(eta)} "
                        f"| {path}"
                    )

                    sys.stdout.write(line)
                    sys.stdout.flush()

                    last_print = now
                    last_rate_time = now
                    last_rate_processed = processed

                # Checkpoint every 20000 images.

                if processed % 20000 == 0:
                    images.flush()
                    np.save(PROGRESS_FILE, completed,)

        print()

        # ====================================================
        # FINAL FLUSH
        # ====================================================

        print(
            "\nFlushing image cache "
            "to disk..."
        )

        images.flush()

        np.save(
            PROGRESS_FILE,
            completed,
        )


    # ========================================================
    # SAVE INDEX
    # ========================================================

    print(
        "Saving image index..."
    )

    path_to_id = {
        path: i
        for i, path
        in enumerate(unique_paths)
    }

    with open(
        INDEX_FILE,
        "w",
    ) as f:

        json.dump(
            {
                "num_images":
                    total_images,

                "image_shape": [
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ],

                "dtype": "uint8",

                "image_file":
                    str(
                        IMAGE_FILE.relative_to(
                            ROOT
                        )
                    ),

                "paths":
                    path_to_id,
            },
            f,
        )


    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print(
        "PREPROCESSING COMPLETE"
    )
    print("=" * 70)

    print(
        f"Images processed : "
        f"{total_images:,}"
    )

    print(
        f"Dataset size     : "
        f"{total_gib:.2f} GiB"
    )

    print(
        f"\nImage file       : "
        f"{IMAGE_FILE}"
    )

    print(
        f"Index file       : "
        f"{INDEX_FILE}"
    )

    print(
        "\nCache ready for training."
    )
