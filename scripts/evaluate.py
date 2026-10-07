#!/usr/bin/env python3
"""
33_evaluate_multistep_rgb_queries.py

Evaluation script for the RH20T RGB Query Token world model trained by
32_train_multistep_spatial_transformer.py.

The model is evaluated with the same 16-frame history, 10-step horizon,
128x128 RGB inputs, normalized TCP/action inputs, and autoregressive RGB
feedback used during training.

Example:
    python scripts/33_evaluate_multistep_rgb_queries.py \
        --data-root /path/to/prepared/RH20T \
        --checkpoint checkpoints/best.pt \
        --output-dir results/evaluation
"""

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_msssim import ssim
from torch.utils.data import Dataset, DataLoader


# ============================================================
# MODEL / DATA SETTINGS — MATCH TRAINING SCRIPT EXACTLY
# ============================================================

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

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# ============================================================
# IMAGE CACHE
# ============================================================

class ImageCache:
    def __init__(self, image_file: Path, index_file: Path):
        with index_file.open("r") as f:
            index = json.load(f)

        self.paths = index["paths"]
        self.num_images = int(index["num_images"])
        self.images = np.memmap(
            image_file,
            dtype=np.uint8,
            mode="r",
            shape=(self.num_images, 3, IMAGE_SIZE, IMAGE_SIZE),
        )

    def load(self, path: str) -> torch.Tensor:
        image_id = self.paths.get(str(path))
        if image_id is None:
            raise KeyError(f"Image path not found in cache:\n{path}")
        arr = np.array(self.images[image_id], dtype=np.uint8, copy=True)
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
        image_cache: ImageCache,
    ):
        self.samples = samples
        self.indices = indices
        self.action_mean = action_mean
        self.action_std = action_std
        self.tcp_mean = tcp_mean
        self.tcp_std = tcp_std
        self.image_cache = image_cache

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        sample = self.samples[int(self.indices[i])]

        rgb_history = torch.stack(
            [self.image_cache.load(p) for p in sample["rgb_history"]]
        )
        target_rgb = torch.stack(
            [self.image_cache.load(p) for p in sample["target_rgb"]]
        )

        tcp_history = torch.tensor(
            sample["tcp_history"], dtype=torch.float32
        )
        actions = torch.tensor(
            sample["actions"], dtype=torch.float32
        )

        tcp_history = (tcp_history - self.tcp_mean) / (self.tcp_std + 1e-8)
        actions = (actions - self.action_mean) / (self.action_std + 1e-8)

        return {
            "rgb_history": rgb_history,
            "tcp_history": tcp_history,
            "actions": actions,
            "target_rgb": target_rgb,
        }


# ============================================================
# MODEL — EXACT STATE-DICT COMPATIBLE ARCHITECTURE
# ============================================================

class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, feature_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((SPATIAL_GRID, SPATIAL_GRID)),
        )

    def forward(self, x):
        return self.net(x)


class ImageDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
        )

    def forward(self, z):
        return torch.sigmoid(self.net(z))


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

        self.image_encoder = ImageEncoder(feature_dim=feature_dim)
        self.visual_projection = nn.Linear(feature_dim, d_model)

        self.tcp_encoder = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
        )
        self.tcp_projection = nn.Linear(64, d_model)

        self.action_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
        )
        self.action_projection = nn.Linear(64, d_model)

        self.temporal_position = nn.Parameter(
            torch.zeros(1, HISTORY, 1, d_model)
        )
        self.spatial_position = nn.Parameter(
            torch.zeros(1, 1, NUM_VISUAL_TOKENS, d_model)
        )

        self.token_type_embedding = nn.Embedding(4, d_model)

        self.rgb_query_tokens = nn.Parameter(
            torch.zeros(1, NUM_VISUAL_TOKENS, d_model)
        )
        nn.init.normal_(self.rgb_query_tokens, mean=0.0, std=0.02)

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
        self.transformer_norm = nn.LayerNorm(d_model)

        self.rgb_latent_head = nn.Sequential(
            nn.Linear(d_model, image_latent),
            nn.ReLU(),
        )

        self.decoder = ImageDecoder(latent_dim=image_latent)

        nn.init.normal_(self.temporal_position, mean=0.0, std=0.02)
        nn.init.normal_(self.spatial_position, mean=0.0, std=0.02)

    def encode_history(self, rgb_history, tcp_history):
        B, T, C, H, W = rgb_history.shape

        rgb = rgb_history.reshape(B * T, C, H, W)
        visual_features = self.image_encoder(rgb)
        visual_features = visual_features.flatten(2).transpose(1, 2)
        visual_features = visual_features.reshape(
            B, T, NUM_VISUAL_TOKENS, self.feature_dim
        )

        visual_tokens = self.visual_projection(visual_features)
        visual_tokens = (
            visual_tokens
            + self.temporal_position[:, :T]
            + self.spatial_position[:, :, :NUM_VISUAL_TOKENS]
        )

        visual_type = self.token_type_embedding(
            torch.tensor(0, device=rgb_history.device)
        )
        visual_tokens = visual_tokens + visual_type

        tcp_z = self.tcp_encoder(tcp_history)
        tcp_tokens = self.tcp_projection(tcp_z)
        tcp_tokens = tcp_tokens + self.temporal_position[:, :T, 0, :]

        tcp_type = self.token_type_embedding(
            torch.tensor(1, device=rgb_history.device)
        )
        tcp_tokens = tcp_tokens + tcp_type

        frame_tokens = torch.cat(
            [visual_tokens, tcp_tokens.unsqueeze(2)], dim=2
        )
        return frame_tokens.reshape(
            B,
            T * (NUM_VISUAL_TOKENS + 1),
            self.d_model,
        )

    def predict_one_step(self, rgb_history, tcp_history, action):
        B = rgb_history.shape[0]

        history_tokens = self.encode_history(
            rgb_history, tcp_history
        )

        action_z = self.action_encoder(action)
        action_token = self.action_projection(action_z).unsqueeze(1)
        action_type = self.token_type_embedding(
            torch.tensor(2, device=rgb_history.device)
        )
        action_token = action_token + action_type

        query_type = self.token_type_embedding(
            torch.tensor(3, device=rgb_history.device)
        )
        query_tokens = (
            self.rgb_query_tokens
            + self.spatial_position.squeeze(1)[:, :NUM_VISUAL_TOKENS, :]
            + query_type
        )
        query_tokens = query_tokens.expand(B, -1, -1)

        tokens = torch.cat(
            [history_tokens, action_token, query_tokens], dim=1
        )

        transformed = self.history_transformer(tokens)
        transformed = self.transformer_norm(transformed)

        rgb_query_outputs = transformed[:, -NUM_VISUAL_TOKENS:, :]
        predicted_visual_tokens = self.rgb_latent_head(rgb_query_outputs)

        rgb_latent = (
            predicted_visual_tokens
            .reshape(
                B,
                SPATIAL_GRID,
                SPATIAL_GRID,
                self.image_latent,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return self.decoder(rgb_latent)

    def forward(self, rgb_history, tcp_history, actions):
        current_rgb_history = rgb_history
        current_tcp_history = tcp_history
        predicted_rgb = []

        for t in range(actions.shape[1]):
            rgb = self.predict_one_step(
                current_rgb_history,
                current_tcp_history,
                actions[:, t],
            )
            predicted_rgb.append(rgb)
            current_rgb_history = torch.cat(
                [
                    current_rgb_history[:, 1:],
                    rgb.detach().unsqueeze(1),
                ],
                dim=1,
            )

        return torch.stack(predicted_rgb, dim=1)


# ============================================================
# CHECKPOINT / METRICS
# ============================================================

def load_checkpoint(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def psnr_from_mse(mse):
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


# ============================================================
# PLOTS
# ============================================================

def save_prediction_figure(pred, target, path):
    fig, axes = plt.subplots(2, HORIZON, figsize=(20, 4.5))

    for t in range(HORIZON):
        target_img = np.transpose(target[t], (1, 2, 0))
        pred_img = np.transpose(pred[t], (1, 2, 0))

        axes[0, t].imshow(target_img)
        axes[0, t].set_title(f"GT t+{t + 1}")
        axes[0, t].axis("off")

        axes[1, t].imshow(pred_img)
        axes[1, t].set_title(f"Pred t+{t + 1}")
        axes[1, t].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_error_plot(mse, mae, path):
    steps = np.arange(1, HORIZON + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, mse, marker="o", label="MSE")
    plt.plot(steps, mae, marker="s", label="MAE")
    plt.xlabel("Prediction step")
    plt.ylabel("Error")
    plt.title("RGB Error per Prediction Step")
    plt.xticks(steps)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def save_ssim_plot(ssim_values, path):
    steps = np.arange(1, HORIZON + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, ssim_values, marker="o")
    plt.xlabel("Prediction step")
    plt.ylabel("SSIM")
    plt.title("SSIM per Prediction Step")
    plt.xticks(steps)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# ============================================================
# CLI
# ============================================================

def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate the RH20T RGB Query Token world model."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Prepared data root containing multistep_index, "
            "multistep_training and preprocessed_images."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "checkpoints" / "best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Evaluate only this many randomly selected test samples.",
    )
    parser.add_argument("--num-visualizations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    data_root = args.data_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("MULTISTEP RGB QUERY TOKEN WORLD MODEL EVALUATION")
    print("=" * 70)
    print("Device     :", device)
    if device == "cuda":
        print("GPU        :", torch.cuda.get_device_name(0))
    print("Data root  :", data_root)
    print("Checkpoint :", checkpoint_path)
    print("Output     :", output_dir)

    index_path = data_root / "multistep_index" / "samples.npy"
    train_dir = data_root / "multistep_training"
    cache_dir = data_root / "preprocessed_images"
    cache_file = cache_dir / "images.dat"
    cache_index_file = cache_dir / "index.json"
    test_indices_path = train_dir / "test_indices.npy"

    required = [
        index_path,
        test_indices_path,
        train_dir / "action_mean.npy",
        train_dir / "action_std.npy",
        train_dir / "tcp_mean.npy",
        train_dir / "tcp_std.npy",
        cache_file,
        cache_index_file,
        checkpoint_path,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found:\n{path}")

    print("\nLoading sample index...")
    samples = np.load(index_path, allow_pickle=True)
    print("Total samples:", len(samples))

    test_indices_all = np.load(test_indices_path)
    print("Total test samples:", len(test_indices_all))

    if args.num_samples is None or len(test_indices_all) <= args.num_samples:
        eval_indices = np.asarray(test_indices_all, dtype=np.int64)
    else:
        rng = np.random.default_rng(args.seed)
        eval_indices = rng.choice(
            test_indices_all,
            size=args.num_samples,
            replace=False,
        ).astype(np.int64)

    print("Evaluation samples:", len(eval_indices))

    action_mean = torch.tensor(
        np.load(train_dir / "action_mean.npy"), dtype=torch.float32
    )
    action_std = torch.tensor(
        np.load(train_dir / "action_std.npy"), dtype=torch.float32
    )
    tcp_mean = torch.tensor(
        np.load(train_dir / "tcp_mean.npy"), dtype=torch.float32
    )
    tcp_std = torch.tensor(
        np.load(train_dir / "tcp_std.npy"), dtype=torch.float32
    )

    print("\nLoading preprocessed image cache...")
    image_cache = ImageCache(cache_file, cache_index_file)
    print("Cached images:", f"{image_cache.num_images:,}")

    dataset = MultistepDataset(
        samples,
        eval_indices,
        action_mean,
        action_std,
        tcp_mean,
        tcp_std,
        image_cache,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    if args.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )

    loader = DataLoader(dataset, **loader_kwargs)

    model = WorldModel().to(device)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    print(
        "Trainable parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )
    if isinstance(checkpoint, dict):
        print("Checkpoint epoch:", checkpoint.get("epoch"))
        print("Checkpoint val loss:", checkpoint.get("val_loss"))

    mse_sum = np.zeros(HORIZON, dtype=np.float64)
    mae_sum = np.zeros(HORIZON, dtype=np.float64)
    ssim_sum = np.zeros(HORIZON, dtype=np.float64)
    sample_count = 0
    viz_count = 0

    print("\nRunning autoregressive evaluation...")

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            rgb_history = (
                batch["rgb_history"].to(device, non_blocking=True)
                .float()
                .div_(255.0)
            )
            tcp_history = batch["tcp_history"].to(
                device, non_blocking=True
            )
            actions = batch["actions"].to(
                device, non_blocking=True
            )
            target_rgb = (
                batch["target_rgb"].to(device, non_blocking=True)
                .float()
                .div_(255.0)
            )

            current_rgb_history = rgb_history.clone()
            current_tcp_history = tcp_history
            predicted_steps = []

            for t in range(HORIZON):
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(device == "cuda"),
                ):
                    pred_rgb_t = model.predict_one_step(
                        current_rgb_history,
                        current_tcp_history,
                        actions[:, t],
                    )

                predicted_steps.append(pred_rgb_t.float())
                current_rgb_history = torch.cat(
                    [
                        current_rgb_history[:, 1:],
                        pred_rgb_t.float().detach().unsqueeze(1),
                    ],
                    dim=1,
                )

            pred_rgb = torch.stack(predicted_steps, dim=1)
            batch_size = pred_rgb.shape[0]

            for t in range(HORIZON):
                pred_t = pred_rgb[:, t]
                target_t = target_rgb[:, t]

                mse_batch = (pred_t - target_t).pow(2).mean(dim=(1, 2, 3))
                mae_batch = (pred_t - target_t).abs().mean(dim=(1, 2, 3))
                ssim_batch = ssim(
                    pred_t.float(),
                    target_t.float(),
                    data_range=1.0,
                    size_average=False,
                )

                mse_sum[t] += mse_batch.double().sum().item()
                mae_sum[t] += mae_batch.double().sum().item()
                ssim_sum[t] += ssim_batch.double().sum().item()

            while viz_count < args.num_visualizations and viz_count < batch_size:
                save_prediction_figure(
                    pred_rgb[viz_count].detach().cpu().numpy(),
                    target_rgb[viz_count].detach().cpu().numpy(),
                    output_dir / f"rgb_prediction_{viz_count:02d}.png",
                )
                viz_count += 1

            sample_count += batch_size

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                print(f"  batch {batch_idx + 1}/{len(loader)}")

    if sample_count == 0:
        raise RuntimeError("No test samples were evaluated.")

    per_step_mse = mse_sum / sample_count
    per_step_mae = mae_sum / sample_count
    per_step_ssim = ssim_sum / sample_count
    per_step_psnr = np.array(
        [psnr_from_mse(x) for x in per_step_mse],
        dtype=np.float64,
    )

    rgb_mse = float(per_step_mse.mean())
    rgb_mae = float(per_step_mae.mean())
    mean_ssim = float(per_step_ssim.mean())
    psnr_db = float(psnr_from_mse(rgb_mse))
    l1_ssim_loss = float(
        0.7 * rgb_mae + 0.3 * (1.0 - mean_ssim)
    )

    save_error_plot(
        per_step_mse,
        per_step_mae,
        output_dir / "rgb_error_per_step.png",
    )
    save_ssim_plot(
        per_step_ssim,
        output_dir / "ssim_per_step.png",
    )

    checkpoint_metadata = {}
    if isinstance(checkpoint, dict):
        for key in ("epoch", "val_loss", "val_rgb", "val_ssim"):
            if key in checkpoint:
                checkpoint_metadata[key] = checkpoint[key]

    metrics = {
        "evaluation_samples": int(sample_count),
        "history": HISTORY,
        "horizon": HORIZON,
        "image_size": IMAGE_SIZE,
        "spatial_grid": SPATIAL_GRID,
        "num_visual_tokens": NUM_VISUAL_TOKENS,
        "rgb_mse": rgb_mse,
        "rgb_mae": rgb_mae,
        "ssim": mean_ssim,
        "psnr_db": psnr_db,
        "l1_ssim_loss": l1_ssim_loss,
        "t_plus_1_mse": float(per_step_mse[0]),
        "t_plus_10_mse": float(per_step_mse[-1]),
        "per_step_mse": [float(x) for x in per_step_mse],
        "per_step_mae": [float(x) for x in per_step_mae],
        "per_step_ssim": [float(x) for x in per_step_ssim],
        "per_step_psnr_db": [float(x) for x in per_step_psnr],
        "checkpoint": str(checkpoint_path),
        "device": device,
        "seed": args.seed,
        "checkpoint_metadata": checkpoint_metadata,
    }

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"RGB MSE : {rgb_mse:.8f}")
    print(f"RGB MAE : {rgb_mae:.8f}")
    print(f"SSIM    : {mean_ssim:.8f}")
    print(f"PSNR    : {psnr_db:.4f} dB")
    print(f"t+1 MSE : {per_step_mse[0]:.8f}")
    print(f"t+10 MSE: {per_step_mse[-1]:.8f}")
    print("\nMetrics saved to:", output_dir / "metrics.json")


if __name__ == "__main__":
    main()
