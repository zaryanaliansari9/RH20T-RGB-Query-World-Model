import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from pytorch_msssim import ssim

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================
# CONFIG
# ============================================================

import argparse

# ============================================================
# COMMAND-LINE CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser(
    description="Train the RH20T RGB Query Token world model."
)

parser.add_argument(
    "--data-root",
    type=Path,
    required=True,
    help="Prepared RH20T data root.",
)

parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="Directory for model checkpoints and training history.",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=16,
)

parser.add_argument(
    "--epochs",
    type=int,
    default=2,
)

parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-4,
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=2,
)

args = parser.parse_args()

ROOT = args.data_root.resolve()

INDEX_PATH = (
    ROOT
    / "multistep_index"
    / "samples.npy"
)

TRAIN_DIR = (
    ROOT
    / "multistep_training"
)

CACHE_DIR = (
    ROOT
    / "preprocessed_images"
)

CACHE_IMAGE_FILE = (
    CACHE_DIR / "images.dat"
)

CACHE_INDEX_FILE = (
    CACHE_DIR / "index.json"
)

OUTPUT_DIR = args.output_dir.resolve()
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

HISTORY = 16
HORIZON = 10

# ============================================================
# SPATIAL TRANSFORMER
# ============================================================

TRANSFORMER_DIM = 256
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_FF_DIM = 512
TRANSFORMER_DROPOUT = 0.1

SPATIAL_GRID = 4
NUM_VISUAL_TOKENS = SPATIAL_GRID * SPATIAL_GRID

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.learning_rate
NUM_WORKERS = args.num_workers

MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

IMAGE_SIZE = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 60)
print("MULTISTEP WORLD MODEL TRAINING — TRANSFORMER")
print("=" * 60)

print("Device :", DEVICE)

if DEVICE == "cuda":
    print("GPU    :", torch.cuda.get_device_name(0))

print("Dynamics       : Transformer")
print("Transformer dim:", TRANSFORMER_DIM)
print("Transformer heads:", TRANSFORMER_HEADS)
print("Transformer layers:", TRANSFORMER_LAYERS)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading sample index...")

samples = np.load(INDEX_PATH, allow_pickle=True)

print("Total samples:", len(samples))


train_indices = np.load(
    TRAIN_DIR / "train_indices.npy"
)

if MAX_TRAIN_SAMPLES is not None:
    train_indices = train_indices[:MAX_TRAIN_SAMPLES]

val_indices = np.load(
    TRAIN_DIR / "val_indices.npy"
)

if MAX_VAL_SAMPLES is not None:
    val_indices = val_indices[:MAX_VAL_SAMPLES]

print("Train:", len(train_indices))
print("Val  :", len(val_indices))

# ============================================================
# NORMALIZATION
# ============================================================

action_mean = torch.tensor(
    np.load(TRAIN_DIR / "action_mean.npy"),
    dtype=torch.float32,
)

action_std = torch.tensor(
    np.load(TRAIN_DIR / "action_std.npy"),
    dtype=torch.float32,
)

tcp_mean = torch.tensor(
    np.load(TRAIN_DIR / "tcp_mean.npy"),
    dtype=torch.float32,
)

tcp_std = torch.tensor(
    np.load(TRAIN_DIR / "tcp_std.npy"),
    dtype=torch.float32,
)


print("\nNormalization:")
print("Action mean:", action_mean.numpy())
print("Action std :", action_std.numpy())
print("TCP mean   :", tcp_mean.numpy())
print("TCP std    :", tcp_std.numpy())


# ============================================================
# PREPROCESSED IMAGE CACHE
# ============================================================

print("\nLoading preprocessed image cache...")

with open(CACHE_INDEX_FILE, "r") as f:
    cache_index = json.load(f)

cache_paths = cache_index["paths"]
cache_num_images = cache_index["num_images"]

image_cache = np.memmap(
    CACHE_IMAGE_FILE,
    dtype=np.uint8,
    mode="r",
    shape=(
        cache_num_images,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ),
)

print(
    f"Cached images: {cache_num_images:,}"
)

print(
    f"Cache shape  : {image_cache.shape}"
)

def load_cached_image(path):
    """
    Load a preprocessed image directly from images.dat.

    Returns:
        torch.FloatTensor
        Shape: (3, 128, 128)
        Range: [0, 1]
    """

    path = str(path)

    image_id = cache_paths.get(path)

    if image_id is None:
        raise KeyError(
            f"Image path not found in cache:\n{path}"
        )

    # Copy the memmap slice so the tensor does not depend
    # directly on the underlying mmap buffer.
    arr = np.array(
        image_cache[image_id],
        dtype=np.uint8,
        copy=True,
    )

    return torch.from_numpy(arr)

# ============================================================
# DATASET
# ============================================================

class MultistepDataset(Dataset):

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

        idx = int(self.indices[i])

        sample = self.samples[idx]

        # ----------------------------------------------------
        # RGB history
        # ----------------------------------------------------

        rgb_paths = sample["rgb_history"]

        rgb_history = torch.stack(
            [load_cached_image(p) for p in rgb_paths]
        )

        # Shape:
        # (16, 3, 128, 128)

        # ----------------------------------------------------
        # Target RGB
        # ----------------------------------------------------

        target_paths = sample["target_rgb"]

        target_rgb = torch.stack(
            [load_cached_image(p) for p in target_paths]
        )

        # Shape:
        # (10, 3, 128, 128)

        # ----------------------------------------------------
        # TCP
        # ----------------------------------------------------

        tcp_history = torch.tensor(
            sample["tcp_history"],
            dtype=torch.float32,
        )

        # ----------------------------------------------------
        # Actions
        # ----------------------------------------------------

        actions = torch.tensor(
            sample["actions"],
            dtype=torch.float32,
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        tcp_history = (
            tcp_history - self.tcp_mean
        ) / (self.tcp_std + 1e-8)

        actions = (
            actions - self.action_mean
        ) / (self.action_std + 1e-8)

        return {
            "rgb_history": rgb_history,
            "tcp_history": tcp_history,
            "actions": actions,
            "target_rgb": target_rgb,
        }


# ============================================================
# MODEL COMPONENTS
# ============================================================

# ============================================================
# IMAGE ENCODER — SPATIAL FEATURES
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

            # 8x8 -> 4x4
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
# IMAGE DECODER — SPATIAL LATENT
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
                256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            # 8 -> 16
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            # 16 -> 32
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            # 32 -> 64
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            # 64 -> 128
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            # Final RGB
            nn.Conv2d(
                16,
                3,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, z):

        return torch.sigmoid(
            self.net(z)
        )


# ============================================================
# WORLD MODEL — SPATIAL TOKEN TRANSFORMER
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
        #
        # CNN spatial feature:
        #
        # 256 channels
        #
        # becomes:
        #
        # 256-D Transformer token
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

        # Temporal position:
        #
        # frame 0 ... frame 15

        self.temporal_position = nn.Parameter(
            torch.zeros(
                1,
                HISTORY,
                1,
                d_model,
            )
        )

        # Spatial position:
        #
        # 16 spatial locations in 4x4 grid

        self.spatial_position = nn.Parameter(
            torch.zeros(
                1,
                1,
                NUM_VISUAL_TOKENS,
                d_model,
            )
        )

        # ====================================================
        # TOKEN TYPE EMBEDDINGS
        #
        # 0 = visual
        # 1 = TCP
        # 2 = action
        # ====================================================

        self.token_type_embedding = nn.Embedding(
            4,
            d_model,
        )

        self.rgb_query_tokens = nn.Parameter(
            torch.zeros(
                1,
                NUM_VISUAL_TOKENS,
                d_model,
            )
        )

        nn.init.normal_(
            self.rgb_query_tokens,
            mean=0.0,
            std=0.02,
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
        #
        # Each output visual token becomes a 128-D
        # spatial latent feature.
        # ====================================================

        self.rgb_latent_head = nn.Sequential(

            nn.Linear(
                d_model,
                image_latent,
            ),

            nn.ReLU(),
        )

        # # ====================================================
        # # TCP HEAD
        # # ====================================================

        # self.tcp_head = nn.Sequential(

        #     nn.Linear(
        #         d_model,
        #         256,
        #     ),

        #     nn.GELU(),

        #     nn.Linear(
        #         256,
        #         7,
        #     ),
        # )

        # ====================================================
        # IMAGE DECODER
        # ====================================================

        self.decoder = ImageDecoder(
            latent_dim=image_latent
        )

        # ====================================================
        # INITIALIZATION
        # ====================================================

        nn.init.normal_(
            self.temporal_position,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.spatial_position,
            mean=0.0,
            std=0.02,
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
        # RGB → SPATIAL FEATURES
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

        # Shape:
        #
        # [B*T, 256, 4, 4]

        # ----------------------------------------------------
        # Convert spatial map → spatial tokens
        # ----------------------------------------------------

        visual_features = (
            visual_features
            .flatten(2)
            .transpose(1, 2)
        )

        # Shape:
        #
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
        # Project to Transformer dimension
        # ----------------------------------------------------

        visual_tokens = self.visual_projection(
            visual_features
        )

        # Shape:
        #
        # [B, 16, 16, 256]

        # ----------------------------------------------------
        # POSITIONAL EMBEDDINGS
        # ----------------------------------------------------

        visual_tokens = (
            visual_tokens
            + self.temporal_position[:, :T]
            + self.spatial_position[:, :, :NUM_VISUAL_TOKENS]
        )

        # ----------------------------------------------------
        # TOKEN TYPE = VISUAL
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

        # Shape:
        #
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
        # INTERLEAVE FRAME TOKENS
        # ====================================================

        # Each frame becomes:
        #
        # visual1 ... visual16 TCP
        #
        # giving:
        #
        # 17 tokens/frame

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

        return frame_tokens

    # ========================================================
    # ONE-STEP TRANSFORMER DYNAMICS
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

        action_token = action_token + action_type

        # --------------------------------------------------------
        # RGB QUERY TOKENS
        #
        # These tokens explicitly represent the future RGB frame
        # that the Transformer is being asked to generate.
        # --------------------------------------------------------

        query_type = self.token_type_embedding(
            torch.tensor(
                3,
                device=rgb_history.device,
            )
        )

        query_tokens = (
            self.rgb_query_tokens
            + self.spatial_position.squeeze(1)[
                :,
                :NUM_VISUAL_TOKENS,
                :
            ]
            + query_type
        )

        query_tokens = query_tokens.expand(
            B,
            -1,
            -1,
        )

        # --------------------------------------------------------
        # HISTORY + ACTION + RGB QUERIES
        # --------------------------------------------------------

        tokens = torch.cat(
            [
                history_tokens,
                action_token,
                query_tokens,
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
        # EXTRACT RGB QUERY OUTPUTS
        #
        # The last NUM_VISUAL_TOKENS positions correspond to the
        # learned queries for the next RGB frame.
        # --------------------------------------------------------

        rgb_query_outputs = transformed[
            :,
            -NUM_VISUAL_TOKENS:,
            :
        ]

        # --------------------------------------------------------
        # PREDICT NEXT SPATIAL LATENT
        # --------------------------------------------------------

        predicted_visual_tokens = (
            self.rgb_latent_head(
                rgb_query_outputs
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
        # DECODE RGB
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
# LOSS
# ============================================================

def compute_loss(
    pred_rgb,
    target_rgb,
):

    # --------------------------------------------------------
    # L1 / MAE
    # --------------------------------------------------------

    l1_loss = F.l1_loss(
        pred_rgb,
        target_rgb,
    )

    # --------------------------------------------------------
    # SSIM
    #
    # Compute in float32 outside AMP because SSIM can be
    # numerically sensitive in float16.
    # --------------------------------------------------------

    with torch.autocast(
        device_type="cuda",
        enabled=False,
    ):

        ssim_value = ssim(
            pred_rgb.float(),
            target_rgb.float(),
            data_range=1.0,
            size_average=True,
        )

        ssim_loss = 1.0 - ssim_value

    # --------------------------------------------------------
    # Combined RGB loss
    # --------------------------------------------------------

    loss = (
        0.7 * l1_loss
        + 0.3 * ssim_loss
    )

    return (
        loss,
        l1_loss,
        ssim_loss,
    )


# ============================================================
# DATASETS
# ============================================================

print("\nCreating datasets...")

# --------------------------------------------------------
# TRAINING DATA SELECTION
# --------------------------------------------------------

if (
    MAX_TRAIN_SAMPLES is not None
    and len(train_indices) > MAX_TRAIN_SAMPLES
):
    rng = np.random.default_rng(SEED)

    train_indices = rng.choice(
        train_indices,
        size=MAX_TRAIN_SAMPLES,
        replace=False,
    )

if (
    MAX_VAL_SAMPLES is not None
    and len(val_indices) > MAX_VAL_SAMPLES
):
    rng = np.random.default_rng(SEED)

    val_indices = rng.choice(
        val_indices,
        size=MAX_VAL_SAMPLES,
        replace=False,
    )

print("Training samples used  :", len(train_indices))
print("Validation samples used:", len(val_indices))

train_dataset = MultistepDataset(
    samples,
    train_indices,
    action_mean,
    action_std,
    tcp_mean,
    tcp_std,
)

val_dataset = MultistepDataset(
    samples,
    val_indices,
    action_mean,
    action_std,
    tcp_mean,
    tcp_std,
)

def worker_init_fn(worker_id):
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2,
    worker_init_fn=worker_init_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2,
    worker_init_fn=worker_init_fn,
)

print("Train batches:", len(train_loader))
print("Val batches  :", len(val_loader))


# ============================================================
# MODEL
# ============================================================

model = WorldModel().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-5,
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(DEVICE == "cuda"),
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
)


num_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nModel parameters:", num_parameters)

print(
    "Transformer parameters:",
    sum(
        p.numel()
        for p in model.history_transformer.parameters()
        if p.requires_grad
    ),
)


# ============================================================
# TRAINING
# ============================================================

best_val = float("inf")

history = []

import time

print("\nTesting DataLoader...")

loader_start = time.perf_counter()

first_batch = next(iter(train_loader))

print("\nTesting Transformer forward pass...")

with torch.no_grad():

    test_rgb = first_batch["rgb_history"].to(
        DEVICE
    ).float().div_(255.0)

    test_tcp = first_batch["tcp_history"].to(
        DEVICE
    )

    test_actions = first_batch["actions"].to(
        DEVICE
    )

    test_rgb_pred = (
        model.predict_one_step(
            test_rgb,
            test_tcp,
            test_actions[:, 0],
        )
    )

print(
    "Test RGB output:",
    tuple(test_rgb_pred.shape),
)

del test_rgb
del test_tcp
del test_actions
del test_rgb_pred

if DEVICE == "cuda":
    torch.cuda.empty_cache()

loader_time = time.perf_counter() - loader_start

print(
    f"First batch loading time: "
    f"{loader_time:.3f} seconds"
)

for k, v in first_batch.items():
    print(
        f"  {k:15s}",
        tuple(v.shape),
        v.dtype,
    )


for epoch in range(1, EPOCHS + 1):

    print("\n" + "=" * 60)
    print(f"EPOCH {epoch}/{EPOCHS}")
    print("=" * 60)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_total = 0.0
    train_rgb = 0.0
    train_ssim = 0.0

    for batch_idx, batch in enumerate(train_loader):
        rgb_history = batch["rgb_history"].to(
            DEVICE,
            non_blocking=True,
        ).float().div_(255.0)

        tcp_history = batch["tcp_history"].to(
            DEVICE,
            non_blocking=True,
        )

        actions = batch["actions"].to(
            DEVICE,
            non_blocking=True,
        )

        target_rgb = batch["target_rgb"].to(
            DEVICE,
            non_blocking=True,
        ).float().div_(255.0)

        # ====================================================
        # PER-PREDICTION TRAINING
        # ====================================================

        batch_total = 0.0
        batch_rgb = 0.0
        batch_ssim = 0.0

        # ----------------------------------------------------
        # Start with the real history
        # ----------------------------------------------------

        current_rgb_history = rgb_history.clone()
        current_tcp_history = tcp_history

        for t in range(HORIZON):

            # ------------------------------------------------
            # ZERO GRAD FOR THIS PREDICTION
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # ONE-STEP FORWARD
            # ------------------------------------------------

            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=(DEVICE == "cuda")
            ):

                pred_rgb_t = model.predict_one_step(
                    current_rgb_history,
                    current_tcp_history,
                    actions[:, t],
                )

                loss, rgb_loss, ssim_loss = compute_loss(
                    pred_rgb_t,
                    target_rgb[:, t],
                )

            # ------------------------------------------------
            # BACKPROPAGATE THIS PREDICTION ONLY
            # ------------------------------------------------

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            # ------------------------------------------------
            # UPDATE MODEL NOW
            # ------------------------------------------------

            scaler.step(
                optimizer
            )

            scaler.update()

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            loss_value = loss.item()
            rgb_value = rgb_loss.item()
            ssim_value = ssim_loss.item()

            batch_total += loss_value
            batch_rgb += rgb_value
            batch_ssim += ssim_value

            # ------------------------------------------------
            # PREPARE NEXT HISTORY
            #
            # IMPORTANT:
            # detach because the previous optimization step
            # is now finished.
            # ------------------------------------------------

            with torch.no_grad():
                current_rgb_history[:, :-1].copy_(
                    current_rgb_history[:, 1:]
                )

                current_rgb_history[:, -1].copy_(
                    pred_rgb_t.detach()
                )

            # ------------------------------------------------
            # CLEAN UP TEMPORARY TENSORS
            # ------------------------------------------------

            del pred_rgb_t
            del loss
            del rgb_loss
            del ssim_loss

            # ------------------------------------------------
            # OPTIONAL PROGRESS
            # ------------------------------------------------

            if (batch_idx + 1) % 100 == 0 and t == HORIZON - 1:

                print(
                    f"  batch {batch_idx + 1}/{len(train_loader)} "
                    f"last_step_loss={loss_value:.6f} "
                    f"rgb={rgb_value:.6f} "
                    f"ssim={ssim_value:.6f}"
                )

        # ----------------------------------------------------
        # AVERAGE THE 10 PREDICTIONS FOR THIS BATCH
        # ----------------------------------------------------

        batch_total /= HORIZON
        batch_rgb /= HORIZON
        batch_ssim /= HORIZON

        train_total += batch_total
        train_rgb += batch_rgb
        train_ssim += batch_ssim

        if (batch_idx + 1) % 100 == 0:

            print(
                f"  batch {batch_idx + 1}/{len(train_loader)} "
                f"loss={batch_total:.6f} "
                f"rgb={batch_rgb:.6f} "
                f"ssim={batch_ssim:.6f}"
            )

    train_total /= len(train_loader)
    train_rgb /= len(train_loader)
    train_ssim /= len(train_loader)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_total = 0.0
    val_rgb = 0.0
    val_ssim = 0.0

    with torch.no_grad():

        for batch in val_loader:

            rgb_history = batch["rgb_history"].to(
                DEVICE,
                non_blocking=True,
            ).float().div_(255.0)

            tcp_history = batch["tcp_history"].to(
                DEVICE,
                non_blocking=True,
            )

            actions = batch["actions"].to(
                DEVICE,
                non_blocking=True,
            )

            target_rgb = batch["target_rgb"].to(
                DEVICE,
                non_blocking=True,
            ).float().div_(255.0)

            current_rgb_history = rgb_history.clone()
            current_tcp_history = tcp_history

            batch_total = 0.0
            batch_rgb = 0.0
            batch_ssim = 0.0

            for t in range(HORIZON):

                pred_rgb_t = (
                    model.predict_one_step(
                        current_rgb_history,
                        current_tcp_history,
                        actions[:, t],
                    )
                )

                loss, rgb_loss, ssim_loss = (
                    compute_loss(
                        pred_rgb_t,
                        target_rgb[:, t],
                    )
                )

                batch_total += loss.item()
                batch_rgb += rgb_loss.item()
                batch_ssim += ssim_loss.item()

                # Roll the predicted state forward
                current_rgb_history[:, :-1].copy_(
                    current_rgb_history[:, 1:]
                )

                current_rgb_history[:, -1].copy_(
                    pred_rgb_t
                )

            batch_total /= HORIZON
            batch_rgb /= HORIZON
            batch_ssim /= HORIZON

            val_total += batch_total
            val_rgb += batch_rgb
            val_ssim += batch_ssim

    val_total /= len(val_loader)
    val_rgb /= len(val_loader)
    val_ssim /= len(val_loader)

    scheduler.step(val_total)

    lr = optimizer.param_groups[0]["lr"]

    print("\nEpoch results:")
    print(f"Train loss : {train_total:.6f}")
    print(f"Train RGB  : {train_rgb:.6f}")
    print(f"Train SSIM  : {train_ssim:.6f}")

    print(f"Val loss   : {val_total:.6f}")
    print(f"Val RGB    : {val_rgb:.6f}")
    print(f"Val SSIM    : {val_ssim:.6f}")

    print(f"LR         : {lr:.2e}")

    history.append({
        "epoch": epoch,
        "train_loss": train_total,
        "train_rgb": train_rgb,
        "train_ssim": train_ssim,
        "val_loss": val_total,
        "val_rgb": val_rgb,
        "val_ssim": val_ssim,
        "lr": lr,
    })

    # --------------------------------------------------------
    # SAVE LAST
    # --------------------------------------------------------

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_total,
        },
        OUTPUT_DIR / "last.pt",
    )

    # --------------------------------------------------------
    # SAVE BEST
    # --------------------------------------------------------

    if val_total < best_val:

        best_val = val_total

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_total,
                "val_rgb": val_rgb,
                "val_ssim": val_ssim,
                "history": history,
            },
            OUTPUT_DIR / "best.pt",
        )

        print("  New best model saved.")


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    OUTPUT_DIR / "training_history.json",
    "w",
) as f:

    json.dump(
        history,
        f,
        indent=2,
    )


print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("Best validation loss:", best_val)
print("Output:", OUTPUT_DIR)
