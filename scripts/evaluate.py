import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pytorch_msssim import ssim

from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("dataset")

INDEX_PATH = ROOT / "multistep_index" / "samples.npy"
TRAIN_DIR = ROOT / "multistep_training"

CACHE_DIR = ROOT / "preprocessed_images"
CACHE_FILE = CACHE_DIR / "images.dat"
CACHE_INDEX_FILE = CACHE_DIR / "index.json"

MODEL_DIR = ROOT / "world_model_spatial_transformer"
CHECKPOINT = MODEL_DIR / "best.pt"

OUTPUT_DIR = MODEL_DIR / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Dataset settings
# ------------------------------------------------------------

HISTORY = 16
HORIZON = 10

IMAGE_SIZE = 128

TRANSFORMER_DIM = 256
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_FF_DIM = 512
TRANSFORMER_DROPOUT = 0.1

SPATIAL_GRID = 4
NUM_VISUAL_TOKENS = SPATIAL_GRID * SPATIAL_GRID


# ------------------------------------------------------------
# Evaluation settings
# ------------------------------------------------------------

# Number of test samples to evaluate.
#
# Start with 100.
#
# Increase later to 500 or the complete test set.
NUM_EVAL_SAMPLES = None

BATCH_SIZE = 16

NUM_WORKERS = 2

SEED = 42


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print(
    "MULTISTEP WORLD MODEL EVALUATION "
    "(RGB ONLY + L1 SSIM)"
)
print("=" * 70)

print("Device     :", DEVICE)

if DEVICE == "cuda":
    print(
        "GPU        :",
        torch.cuda.get_device_name(0),
    )

print("Checkpoint :", CHECKPOINT)
print("Output     :", OUTPUT_DIR)

print("Dynamics        : Spatial Transformer")
print("Dim             :", TRANSFORMER_DIM)
print("Heads           :", TRANSFORMER_HEADS)
print("Layers          :", TRANSFORMER_LAYERS)
print("Spatial grid    :", SPATIAL_GRID, "x", SPATIAL_GRID)

# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    INDEX_PATH,
    TRAIN_DIR / "test_indices.npy",
    TRAIN_DIR / "action_mean.npy",
    TRAIN_DIR / "action_std.npy",
    TRAIN_DIR / "tcp_mean.npy",
    TRAIN_DIR / "tcp_std.npy",
    CACHE_FILE,
    CACHE_INDEX_FILE,
    CHECKPOINT,
]

for file in required_files:

    if not file.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file}"
        )


# ============================================================
# LOAD SAMPLE INDEX
# ============================================================

print("\nLoading sample index...")

samples = np.load(
    INDEX_PATH,
    allow_pickle=True,
)

print(
    "Total samples:",
    len(samples),
)


# ============================================================
# LOAD TEST INDICES
# ============================================================

print("\nLoading test indices")

eval_indices_all = np.load(
    TRAIN_DIR / "test_indices.npy"
)

print(
    "Total test samples:",
    len(eval_indices_all),
)


# ============================================================
# SELECT EVALUATION SAMPLES
# ============================================================

rng = np.random.default_rng(SEED)

if NUM_EVAL_SAMPLES is None:
    eval_indices = eval_indices_all
elif len(eval_indices_all) > NUM_EVAL_SAMPLES:
    eval_indices = rng.choice(
        eval_indices_all,
        size=NUM_EVAL_SAMPLES,
        replace=False,
    )
else:
    eval_indices = eval_indices_all

eval_indices = np.asarray(
    eval_indices,
    dtype=np.int64,
)

print(
    "Evaluation samples:",
    len(eval_indices),
)


# ============================================================
# LOAD NORMALIZATION
# ============================================================

action_mean = torch.tensor(
    np.load(
        TRAIN_DIR / "action_mean.npy"
    ),
    dtype=torch.float32,
)

action_std = torch.tensor(
    np.load(
        TRAIN_DIR / "action_std.npy"
    ),
    dtype=torch.float32,
)

tcp_mean = torch.tensor(
    np.load(
        TRAIN_DIR / "tcp_mean.npy"
    ),
    dtype=torch.float32,
)

tcp_std = torch.tensor(
    np.load(
        TRAIN_DIR / "tcp_std.npy"
    ),
    dtype=torch.float32,
)


print("\nNormalization loaded.")


# ============================================================
# LOAD IMAGE CACHE
# ============================================================

print("\nLoading preprocessed image cache...")

with open(
    CACHE_INDEX_FILE,
    "r",
) as f:

    cache_index = json.load(f)


NUM_IMAGES = cache_index["num_images"]

CACHE_SHAPE = (
    NUM_IMAGES,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
)

images = np.memmap(
    CACHE_FILE,
    dtype=np.uint8,
    mode="r",
    shape=CACHE_SHAPE,
)

print(
    "Cached images:",
    NUM_IMAGES,
)

print(
    "Cache shape  :",
    images.shape,
)


# ============================================================
# PATH -> CACHE ID
# ============================================================

print("\nLoading image path index...")

path_to_id = cache_index["paths"]

print(
    "Indexed paths:",
    len(path_to_id),
)


# ============================================================
# CACHE IMAGE LOADER
# ============================================================

def load_cached_image(path):
    """
    Load one image from the preprocessed memmap.

    Returns:

        (3, 128, 128)

        float32

        [0, 1]
    """

    path = str(path)

    if path not in path_to_id:

        raise KeyError(
            f"\nImage path not found in cache:\n{path}"
        )

    image_id = path_to_id[path]

    arr = images[image_id]

    # Copy before converting to torch.
    #
    # This prevents problems with the read-only memmap.

    arr = np.array(
        arr,
        dtype=np.uint8,
        copy=True,
    )

    return (
        torch.from_numpy(arr)
        .float()
        .div_(255.0)
    )


# ============================================================
# DATASET
# ============================================================

class EvaluationDataset(Dataset):

    def __init__(
        self,
        samples,
        indices,
        action_mean,
        action_std,
        tcp_mean,
        tcp_std,
    ):

        self.samples = samples
        self.indices = indices

        self.action_mean = action_mean
        self.action_std = action_std

        self.tcp_mean = tcp_mean
        self.tcp_std = tcp_std


    def __len__(self):

        return len(self.indices)


    def __getitem__(self, i):

        idx = int(
            self.indices[i]
        )

        sample = self.samples[idx]


        # ----------------------------------------------------
        # RGB HISTORY
        # ----------------------------------------------------

        rgb_paths = sample[
            "rgb_history"
        ]

        rgb_history = torch.stack(
            [
                load_cached_image(p)
                for p in rgb_paths
            ]
        )

        # (16, 3, 128, 128)


        # ----------------------------------------------------
        # TARGET RGB
        # ----------------------------------------------------

        target_paths = sample[
            "target_rgb"
        ]

        target_rgb = torch.stack(
            [
                load_cached_image(p)
                for p in target_paths
            ]
        )

        # (10, 3, 128, 128)


        # ----------------------------------------------------
        # TCP
        # ----------------------------------------------------

        tcp_history = torch.tensor(
            sample["tcp_history"],
            dtype=torch.float32,
        )

        # ----------------------------------------------------
        # ACTIONS
        # ----------------------------------------------------

        actions = torch.tensor(
            sample["actions"],
            dtype=torch.float32,
        )


        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        tcp_history = (
            tcp_history
            - self.tcp_mean
        ) / (
            self.tcp_std + 1e-8
        )

        actions = (
            actions
            - self.action_mean
        ) / (
            self.action_std + 1e-8
        )


        return {
            "rgb_history": rgb_history,
            "tcp_history": tcp_history,
            "actions": actions,
            "target_rgb": target_rgb,
        }


# ============================================================
# MODEL
# ============================================================

class ImageEncoder(nn.Module):

    def __init__(
        self,
        feature_dim=256,
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                128,
                feature_dim,
                4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (
                    SPATIAL_GRID,
                    SPATIAL_GRID,
                )
            ),
        )

    def forward(self, x):

        return self.net(x)


# ============================================================
# IMAGE DECODER
# ============================================================

class ImageDecoder(nn.Module):

    def __init__(
        self,
        latent_dim=128,
    ):

        super().__init__()

        self.net = nn.Sequential(

            # 4 -> 8
            nn.ConvTranspose2d(
                latent_dim,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            # 8 -> 16
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            # 16 -> 32
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            # 32 -> 64
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            # 64 -> 128
            nn.ConvTranspose2d(
                16,
                3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, z):

        return torch.sigmoid(
            self.net(z)
        )

# ============================================================
# WORLD MODEL — TRANSFORMER DYNAMICS
# ============================================================

class WorldModel(nn.Module):

    def __init__(
        self,
        image_latent=128,
        feature_dim=256,
        d_model=TRANSFORMER_DIM,
        nhead=TRANSFORMER_HEADS,
        num_layers=TRANSFORMER_LAYERS,
        dim_feedforward=TRANSFORMER_FF_DIM,
        dropout=TRANSFORMER_DROPOUT,
    ):

        super().__init__()

        self.image_latent = image_latent
        self.feature_dim = feature_dim
        self.d_model = d_model

        # ====================================================
        # IMAGE ENCODER
        # ====================================================

        self.image_encoder = ImageEncoder(
            feature_dim=feature_dim
        )

        # ====================================================
        # VISUAL TOKEN PROJECTION
        # ====================================================

        self.visual_projection = nn.Linear(
            feature_dim,
            d_model,
        )

        # ====================================================
        # TCP ENCODER
        # ====================================================

        self.tcp_encoder = nn.Sequential(
            nn.Linear(
                7,
                64,
            ),
            nn.ReLU(),
        )

        self.tcp_projection = nn.Linear(
            64,
            d_model,
        )

        # ====================================================
        # ACTION ENCODER
        # ====================================================

        self.action_encoder = nn.Sequential(
            nn.Linear(
                6,
                64,
            ),
            nn.ReLU(),
        )

        self.action_projection = nn.Linear(
            64,
            d_model,
        )

        # ====================================================
        # POSITIONAL EMBEDDINGS
        # ====================================================

        self.temporal_position = nn.Parameter(
            torch.zeros(
                1,
                HISTORY,
                1,
                d_model,
            )
        )

        self.spatial_position = nn.Parameter(
            torch.zeros(
                1,
                1,
                NUM_VISUAL_TOKENS,
                d_model,
            )
        )

        # ====================================================
        # TOKEN TYPES
        #
        # 0 = visual
        # 1 = TCP
        # 2 = action
        # ====================================================

        self.token_type_embedding = nn.Embedding(
            3,
            d_model,
        )

        # ====================================================
        # TRANSFORMER
        # ====================================================

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.history_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.transformer_norm = nn.LayerNorm(
            d_model
        )

        # ====================================================
        # RGB LATENT HEAD
        # ====================================================

        self.rgb_latent_head = nn.Sequential(

            nn.Linear(
                d_model,
                image_latent,
            ),

            nn.ReLU(),
        )

        # ====================================================
        # TCP HEAD
        # ====================================================

        self.tcp_head = nn.Sequential(

            nn.Linear(
                d_model,
                d_model,
            ),

            nn.ReLU(),

            nn.Linear(
                d_model,
                7,
            ),
        )

        # ====================================================
        # IMAGE DECODER
        # ====================================================

        self.decoder = ImageDecoder(
            latent_dim=image_latent
        )

    # ========================================================
    # ENCODE HISTORY
    # ========================================================

    def encode_history(
        self,
        rgb_history,
        tcp_history,
    ):

        B, T, C, H, W = rgb_history.shape

        # ----------------------------------------------------
        # RGB -> SPATIAL FEATURES
        # ----------------------------------------------------

        rgb = rgb_history.reshape(
            B * T,
            C,
            H,
            W,
        )

        visual_features = self.image_encoder(
            rgb
        )

        # [B*T, 256, 4, 4]

        # ----------------------------------------------------
        # SPATIAL FEATURE MAP -> TOKENS
        # ----------------------------------------------------

        visual_features = (
            visual_features
            .flatten(2)
            .transpose(1, 2)
        )

        # [B*T, 16, 256]

        visual_features = (
            visual_features
            .reshape(
                B,
                T,
                NUM_VISUAL_TOKENS,
                self.feature_dim,
            )
        )

        # ----------------------------------------------------
        # PROJECT VISUAL TOKENS
        # ----------------------------------------------------

        visual_tokens = self.visual_projection(
            visual_features
        )

        # [B, 16, 16, 256]

        # ----------------------------------------------------
        # TEMPORAL + SPATIAL POSITION
        # ----------------------------------------------------

        visual_tokens = (
            visual_tokens
            + self.temporal_position[:, :T]
            + self.spatial_position[
                :,
                :,
                :NUM_VISUAL_TOKENS
            ]
        )

        # ----------------------------------------------------
        # VISUAL TOKEN TYPE
        # ----------------------------------------------------

        visual_type = self.token_type_embedding(
            torch.tensor(
                0,
                device=rgb_history.device,
            )
        )

        visual_tokens = (
            visual_tokens
            + visual_type
        )

        # ====================================================
        # TCP TOKENS
        # ====================================================

        tcp_z = self.tcp_encoder(
            tcp_history
        )

        tcp_tokens = self.tcp_projection(
            tcp_z
        )

        # [B, 16, 256]

        tcp_tokens = (
            tcp_tokens
            + self.temporal_position[
                :,
                :T,
                0,
                :
            ]
        )

        tcp_type = self.token_type_embedding(
            torch.tensor(
                1,
                device=rgb_history.device,
            )
        )

        tcp_tokens = (
            tcp_tokens
            + tcp_type
        )

        # ====================================================
        # INTERLEAVE VISUAL + TCP
        # ====================================================

        frame_tokens = torch.cat(
            [
                visual_tokens,
                tcp_tokens.unsqueeze(2),
            ],
            dim=2,
        )

        # [B, 16, 17, 256]

        frame_tokens = frame_tokens.reshape(
            B,
            T * (NUM_VISUAL_TOKENS + 1),
            self.d_model,
        )

        # [B, 272, 256]

        return frame_tokens

    # ========================================================
    # ONE-STEP SPATIAL TRANSFORMER PREDICTION
    # ========================================================

    def predict_one_step(
        self,
        rgb_history,
        tcp_history,
        action,
    ):

        B = rgb_history.shape[0]

        # --------------------------------------------------------
        # HISTORY TOKENS
        # --------------------------------------------------------

        history_tokens = self.encode_history(
            rgb_history,
            tcp_history,
        )

        # --------------------------------------------------------
        # ACTION TOKEN
        # --------------------------------------------------------

        action_z = self.action_encoder(
            action
        )

        action_token = self.action_projection(
            action_z
        ).unsqueeze(1)

        action_type = self.token_type_embedding(
            torch.tensor(
                2,
                device=rgb_history.device,
            )
        )

        action_token = (
            action_token
            + action_type
        )

        # --------------------------------------------------------
        # HISTORY + ACTION
        # --------------------------------------------------------

        tokens = torch.cat(
            [
                history_tokens,
                action_token,
            ],
            dim=1,
        )

        # --------------------------------------------------------
        # TRANSFORMER
        # --------------------------------------------------------

        transformed = self.history_transformer(
            tokens
        )

        transformed = self.transformer_norm(
            transformed
        )

        # --------------------------------------------------------
        # CURRENT FRAME VISUAL TOKENS
        # --------------------------------------------------------

        history_transformed = (
            transformed[:, :-1]
            .reshape(
                B,
                HISTORY,
                NUM_VISUAL_TOKENS + 1,
                self.d_model,
            )
        )

        current_visual_tokens = (
            history_transformed[
                :,
                -1,
                :NUM_VISUAL_TOKENS,
                :
            ]
        )

        # --------------------------------------------------------
        # NEXT SPATIAL LATENT
        # --------------------------------------------------------

        predicted_visual_tokens = (
            self.rgb_latent_head(
                current_visual_tokens
            )
        )

        rgb_latent = (
            predicted_visual_tokens
            .reshape(
                B,
                SPATIAL_GRID,
                SPATIAL_GRID,
                self.image_latent,
            )
            .permute(
                0,
                3,
                1,
                2,
            )
            .contiguous()
        )

        # --------------------------------------------------------
        # RGB PREDICTION
        # --------------------------------------------------------

        rgb = self.decoder(
            rgb_latent
        )

        return rgb

    # ========================================================
    # FULL AUTOREGRESSIVE FORWARD
    # ========================================================

    def forward(
        self,
        rgb_history,
        tcp_history,
        actions,
    ):

        current_rgb_history = rgb_history
        current_tcp_history = tcp_history

        predicted_rgb = []

        for t in range(
            actions.shape[1]
        ):

            rgb = self.predict_one_step(
                current_rgb_history,
                current_tcp_history,
                actions[:, t],
            )

            predicted_rgb.append(
                rgb
            )

            # ----------------------------------------------------
            # RGB FEEDBACK
            # ----------------------------------------------------

            current_rgb_history = torch.cat(
                [
                    current_rgb_history[:, 1:],
                    rgb.detach().unsqueeze(1),
                ],
                dim=1,
            )

        predicted_rgb = torch.stack(
            predicted_rgb,
            dim=1,
        )

        return predicted_rgb


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = WorldModel().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Checkpoint epoch:",
    checkpoint.get(
        "epoch",
        "unknown",
    ),
)


print(
    "Checkpoint val loss:",
    checkpoint.get(
        "val_loss",
        "unknown",
    ),
)

# ============================================================
# DATA LOADER
# ============================================================

print("\nCreating evaluation DataLoader...")

dataset = EvaluationDataset(
    samples,
    eval_indices,
    action_mean,
    action_std,
    tcp_mean,
    tcp_std,
)


def worker_init_fn(worker_id):

    import os

    os.environ[
        "OMP_NUM_THREADS"
    ] = "1"

    os.environ[
        "MKL_NUM_THREADS"
    ] = "1"


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(
        NUM_WORKERS > 0
    ),
    prefetch_factor=2,
    worker_init_fn=worker_init_fn,
)


print(
    "Evaluation batches:",
    len(loader),
)

print("\nTesting spatial Transformer forward pass...")

test_batch = next(iter(loader))

test_rgb = test_batch["rgb_history"].to(
    DEVICE,
    non_blocking=True,
)

test_tcp = test_batch["tcp_history"].to(
    DEVICE,
    non_blocking=True,
)

test_action = test_batch["actions"][:, 0].to(
    DEVICE,
    non_blocking=True,
)

with torch.no_grad():

    test_pred_rgb = model.predict_one_step(
        test_rgb,
        test_tcp,
        test_action,
    )

print(
    "Test RGB output:",
    tuple(test_pred_rgb.shape),
)

del test_rgb
del test_tcp
del test_action
del test_pred_rgb
del test_batch

if DEVICE == "cuda":
    torch.cuda.empty_cache()

# ============================================================
# EVALUATION
# ============================================================

print("\nRunning evaluation...")

# ------------------------------------------------------------
# MODEL METRICS
# ------------------------------------------------------------

# ------------------------------------------------------------
# RGB METRICS
# ------------------------------------------------------------

total_rgb_mse = 0.0
total_rgb_mae = 0.0
total_ssim = 0.0

# ------------------------------------------------------------
# PER-STEP RGB METRICS
# ------------------------------------------------------------

rgb_mse_per_step = np.zeros(
    HORIZON,
    dtype=np.float64,
)

rgb_mae_per_step = np.zeros(
    HORIZON,
    dtype=np.float64,
)

ssim_per_step = np.zeros(
    HORIZON,
    dtype=np.float64,
)

rgb_loss_per_step = np.zeros(
    HORIZON,
    dtype=np.float64,
)

psnr_per_step = np.zeros(
    HORIZON,
    dtype=np.float64,
)

# ------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------

visual_examples = []

MAX_VISUAL_EXAMPLES = 5

# ------------------------------------------------------------
# TIMER
# ------------------------------------------------------------

start_time = time.perf_counter()

# ============================================================
# RUN EVALUATION
# ============================================================

with torch.no_grad():

    for batch_idx, batch in enumerate(loader):

        # ----------------------------------------------------
        # MOVE DATA TO DEVICE
        # ----------------------------------------------------

        rgb_history = batch[
            "rgb_history"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        tcp_history = batch[
            "tcp_history"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        actions = batch[
            "actions"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        target_rgb = batch[
            "target_rgb"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # AUTOREGRESSIVE MODEL ROLLOUT
        # ----------------------------------------------------

        current_rgb_history = rgb_history.clone()
        current_tcp_history = tcp_history

        pred_rgb_steps = []

        for t in range(HORIZON):

            if DEVICE == "cuda":

                with torch.amp.autocast(
                    "cuda",
                    dtype=torch.float16,
                ):

                    pred_rgb_t = (
                        model.predict_one_step(
                            current_rgb_history,
                            current_tcp_history,
                            actions[:, t],
                        )
                    )

            else:

                pred_rgb_t = (
                    model.predict_one_step(
                        current_rgb_history,
                        current_tcp_history,
                        actions[:, t],
                    )
                )

            pred_rgb_steps.append(
                pred_rgb_t
            )

            # ------------------------------------------------
            # Feed predictions into next history
            # ------------------------------------------------

            current_rgb_history = torch.cat(
                [
                    current_rgb_history[:, 1:],
                    pred_rgb_t.unsqueeze(1),
                ],
                dim=1,
            )

        pred_rgb = torch.stack(
            pred_rgb_steps,
            dim=1,
        )

        # --------------------------------------------------------
        # RGB MSE
        # --------------------------------------------------------

        rgb_mse = torch.mean(
            (
                pred_rgb
                - target_rgb
            ) ** 2
        )

        # --------------------------------------------------------
        # RGB MAE
        # --------------------------------------------------------

        rgb_mae = torch.mean(
            torch.abs(
                pred_rgb
                - target_rgb
            )
        )

        # --------------------------------------------------------
        # SSIM
        # --------------------------------------------------------

        pred_rgb_ssim = (
            pred_rgb
            .reshape(
                -1,
                3,
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
            .float()
        )

        target_rgb_ssim = (
            target_rgb
            .reshape(
                -1,
                3,
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
            .float()
        )

        with torch.autocast(
            device_type=DEVICE,
            enabled=False,
        ):

            ssim_value = ssim(
                pred_rgb_ssim,
                target_rgb_ssim,
                data_range=1.0,
                size_average=True,
            )

        total_rgb_mse += rgb_mse.item()
        total_rgb_mae += rgb_mae.item()
        total_ssim += ssim_value.item()

        # ----------------------------------------------------
        # PER-STEP METRICS
        # ----------------------------------------------------

        for t in range(HORIZON):

            # ----------------------------------------------------
            # MSE
            # ----------------------------------------------------

            mse_t = torch.mean(
                (
                    pred_rgb[:, t]
                    - target_rgb[:, t]
                ) ** 2
            )

            # ----------------------------------------------------
            # MAE
            # ----------------------------------------------------

            mae_t = torch.mean(
                torch.abs(
                    pred_rgb[:, t]
                    - target_rgb[:, t]
                )
            )

            # ----------------------------------------------------
            # SSIM
            # ----------------------------------------------------

            with torch.autocast(
                device_type=DEVICE,
                enabled=False,
            ):

                ssim_t = ssim(
                    pred_rgb[:, t].float(),
                    target_rgb[:, t].float(),
                    data_range=1.0,
                    size_average=True,
                )

            rgb_mse_per_step[t] += mse_t.item()
            rgb_mae_per_step[t] += mae_t.item()
            ssim_per_step[t] += ssim_t.item()

            rgb_loss_per_step[t] += (
                0.7 * mae_t.item()
                + 0.3 * (1.0 - ssim_t.item())
            )

            psnr_per_step[t] += (
                10.0
                * np.log10(
                    1.0 / max(
                        mse_t.item(),
                        1e-12,
                    )
                )
            )
            
        # ----------------------------------------------------
        # SAVE VISUALIZATION EXAMPLES
        # ----------------------------------------------------

        if len(visual_examples) < MAX_VISUAL_EXAMPLES:

            n = min(
                MAX_VISUAL_EXAMPLES
                - len(visual_examples),

                rgb_history.shape[0],
            )

            for j in range(n):

                visual_examples.append(
                    {
                        "history":
                            rgb_history[j]
                            .float()
                            .cpu(),

                        "target_rgb":
                            target_rgb[j]
                            .float()
                            .cpu(),

                        "pred_rgb":
                            pred_rgb[j]
                            .float()
                            .cpu(),
                    }
                )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            (batch_idx + 1) % 10
            == 0
        ):

            print(
                f"  batch "
                f"{batch_idx + 1}/"
                f"{len(loader)}"
            )

        # ----------------------------------------------------
        # BATCH COUNT
        # ----------------------------------------------------

        num_batches = (
            batch_idx + 1
        )


# ============================================================
# FINAL METRICS
# ============================================================

if DEVICE == "cuda":
    torch.cuda.synchronize()

elapsed = (
    time.perf_counter()
    - start_time
)

# ------------------------------------------------------------
# AVERAGE MODEL LOSSES
# ------------------------------------------------------------
total_rgb_mse /= num_batches
total_rgb_mae /= num_batches
total_ssim /= num_batches

psnr = 10.0 * np.log10(
    1.0 / max(total_rgb_mse, 1e-12)
)

total_rgb_loss = (
    0.7 * total_rgb_mae
    + 0.3 * (1.0 - total_ssim)
)

rgb_mse_per_step /= num_batches
rgb_mae_per_step /= num_batches
ssim_per_step /= num_batches
rgb_loss_per_step /= num_batches
psnr_per_step /= num_batches

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RGB EVALUATION RESULTS")
print("=" * 70)

print(
    f"Samples evaluated : "
    f"{len(eval_indices):,}"
)

print(
    f"RGB MSE           : "
    f"{total_rgb_mse:.8f}"
)

print(
    f"RGB MAE           : "
    f"{total_rgb_mae:.8f}"
)

print(
    f"SSIM              : "
    f"{total_ssim:.8f}"
)

print(
    f"Evaluation time    : "
    f"{elapsed:.2f} seconds"
)

print(
    f"RGB L1+SSIM loss : "
    f"{total_rgb_loss:.8f}"
)

print(
    f"PSNR              : "
    f"{psnr:.4f} dB"
)


# ============================================================
# PER-STEP METRICS
# ============================================================

print("\nPer-step RGB metrics:")

for t in range(HORIZON):

    print(
        f"  t+{t + 1:02d} "
        f"| MSE = {rgb_mse_per_step[t]:.8f} "
        f"| MAE = {rgb_mae_per_step[t]:.8f} "
        f"| SSIM = {ssim_per_step[t]:.6f} "
        f"| Loss = {rgb_loss_per_step[t]:.8f}"
    )

# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "checkpoint":
        str(CHECKPOINT),

    "checkpoint_epoch":
        checkpoint.get(
            "epoch",
            None,
        ),

    "checkpoint_val_loss":
        checkpoint.get(
            "val_loss",
            None,
        ),

    "num_samples":
        int(len(eval_indices)),

    "rgb_loss":
        float(total_rgb_loss),

    "rgb_mse":
        float(total_rgb_mse),

    "rgb_mae":
        float(total_rgb_mae),

    "ssim":
        float(total_ssim),

    "psnr_db":
        float(psnr),

    "rgb_loss_per_step":
        rgb_loss_per_step.tolist(),

    "rgb_mse_per_step":
        rgb_mse_per_step.tolist(),

    "rgb_mae_per_step":
        rgb_mae_per_step.tolist(),

    "ssim_per_step":
        ssim_per_step.tolist(),

    "psnr_per_step":
        psnr_per_step.tolist(),

    "evaluation_time_seconds":
        float(elapsed),

    "history":
        HISTORY,

    "horizon":
        HORIZON,

    "spatial_grid":
        SPATIAL_GRID,

    "transformer_dim":
        TRANSFORMER_DIM,

    "transformer_heads":
        TRANSFORMER_HEADS,

    "transformer_layers":
        TRANSFORMER_LAYERS,

    "batch_size":
        BATCH_SIZE,

    "seed":
        SEED,
}

with open(
    OUTPUT_DIR / "metrics.json",
    "w",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )

# ============================================================
# RGB VISUALIZATION
# ============================================================

print("\nSaving RGB prediction visualizations...")


for example_id, example in enumerate(
    visual_examples
):

    target = (
        example["target_rgb"]
        .numpy()
    )

    prediction = (
        example["pred_rgb"]
        .numpy()
    )

    # Convert uint8 images to [0, 1]
    if target.max() > 1.0:
        target = target / 255.0

    if prediction.max() > 1.0:
        prediction = prediction / 255.0


    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        HORIZON,
        figsize=(20, 5),
    )


    # --------------------------------------------------------
    # Actual
    # --------------------------------------------------------

    for t in range(HORIZON):

        image = np.transpose(
            target[t],
            (1, 2, 0),
        )

        axes[0, t].imshow(
            np.clip(
                image,
                0,
                1,
            )
        )

        axes[0, t].set_title(
            f"t+{t + 1}"
        )

        axes[0, t].axis(
            "off"
        )


    # --------------------------------------------------------
    # Predicted
    # --------------------------------------------------------

    for t in range(HORIZON):

        image = np.transpose(
            prediction[t],
            (1, 2, 0),
        )

        axes[1, t].imshow(
            np.clip(
                image,
                0,
                1,
            )
        )

        axes[1, t].set_title(
            f"t+{t + 1}"
        )

        axes[1, t].axis(
            "off"
        )


    axes[0, 0].set_ylabel(
        "Actual",
        fontsize=12,
    )

    axes[1, 0].set_ylabel(
        "Predicted",
        fontsize=12,
    )


    fig.suptitle(
        f"World Model RGB Prediction "
        f"— Example {example_id}",
        fontsize=14,
    )


    plt.tight_layout()


    filename = (
        OUTPUT_DIR
        / f"rgb_prediction_{example_id:02d}.png"
    )


    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

# ============================================================
# PER-STEP ERROR PLOT
# ============================================================

steps = np.arange(
    1,
    HORIZON + 1,
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    steps,
    rgb_mse_per_step,
    marker="o",
    label="RGB MSE",
)

plt.plot(
    steps,
    rgb_mae_per_step,
    marker="s",
    label="RGB MAE",
)

plt.xlabel(
    "Prediction step"
)

plt.ylabel(
    "Error"
)

plt.title(
    "RGB Prediction Error vs Prediction Horizon"
)

plt.xticks(
    steps
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "rgb_error_per_step.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# PER-STEP SSIM PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    steps,
    ssim_per_step,
    marker="o",
)

plt.xlabel(
    "Prediction step"
)

plt.ylabel(
    "SSIM"
)

plt.title(
    "RGB SSIM vs Prediction Horizon"
)

plt.xticks(
    steps
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ssim_per_step.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nFiles:"
)

print(
    "  metrics.json"
)

print(
    "  rgb_error_per_step.png"
)

print(
    "  ssim_per_step.png"
)

print(
    "  rgb_prediction_00.png ..."
)

print(
    "  ssim_per_step.png"
)

print(
    "\nEvaluated checkpoint:"
)

print(
    CHECKPOINT
)