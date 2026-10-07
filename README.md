# RH20T RGB Query Token World Model

A multimodal world model for robotic manipulation trained on the RH20T dataset.

This repository implements a spatial Transformer world model that predicts future RGB observations from:

- a history of RGB observations,
- robot TCP state history, and
- a sequence of future actions.

The model uses spatial visual tokens and learned RGB query tokens to generate autoregressive multi-step visual predictions.

## Overview

The model receives a 16-frame observation history and predicts the next 10 RGB observations conditioned on the future action sequence.

The architecture progressively converts the visual history into spatial tokens, combines them with robot TCP state and action information using a Transformer, and uses learned RGB query tokens to construct the future visual representation.

The RGB query tokens are trained jointly with the Transformer and are used to aggregate the information required to reconstruct the future RGB observation.

They are not fixed image patches. They are learned query representations that interact with the spatiotemporal history and action information.

## Architecture

### Input

For each training sample:

- RGB history: 16 frames
- RGB resolution: 128 × 128
- TCP history: 16 states
- TCP dimension: 7
- Future action sequence: 10 actions
- Action dimension: 6

### Visual representation

Each RGB observation is encoded into a spatial 4 × 4 grid:

- Spatial grid: 4 × 4
- Visual tokens per frame: 16
- History visual tokens: 16 × 16 = 256

Each frame also contributes a TCP token.

The Transformer therefore receives:

- 256 visual history tokens
- 16 TCP history tokens
- 1 action token
- 16 learned RGB query tokens

for a total of 289 tokens during RGB prediction.

### Transformer

- Embedding dimension: 256
- Attention heads: 4
- Transformer layers: 2
- Feed-forward dimension: 512
- Dropout: 0.1

### RGB query tokens

The model contains 16 trainable RGB query tokens corresponding to the 4 × 4 output latent structure.

The query tokens attend to the spatiotemporal observation history and action information.

Their resulting representations are transformed into the future RGB latent representation and passed through a convolutional decoder to generate the predicted RGB observation.

The query tokens are shared trainable parameters. They are not associated with fixed physical image regions.

### Autoregressive Prediction

The model predicts a sequence of 10 future RGB observations.

At each prediction step:

1. The Transformer processes the observation history and action information.
2. The RGB query tokens aggregate the relevant information.
3. The query representations are converted into a spatial RGB latent representation.
4. The RGB decoder generates the next image.
5. The predicted RGB observation is fed back into the model for subsequent prediction.

This allows the model to perform multi-step autoregressive rollout.

## Dataset

This project uses the RH20T robotic manipulation dataset.

The dataset itself is not included in this repository.

The preprocessing pipeline expects the RH20T data to be available locally.

The complete preparation pipeline converts the original dataset into the representation required by the world model.

The final prepared dataset used for the reported experiments contains:

- 893 scenes
- 497,502 multistep samples
- 403,611 training samples
- 46,868 validation samples
- 47,023 test samples
- 16-frame history
- 10-step prediction horizon

The train/validation/test split is performed at the scene level, preventing samples from the same scene from appearing across different splits.

Normalization statistics are computed using the training scenes only.

## Repository Structure

```text
RH20T-RGB-Query-World-Model/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── LICENSE
│
├── src/
│   └── rgb_query_world_model.py
│
├── scripts/
│   ├── 24_build_all_scenes_multistep_dataset.py
│   ├── 25_build_clean_aligned_all_scenes.py
│   ├── 25_build_multistep_index.py
│   ├── 27_prepare_multistep_training.py
│   ├── 29_preprocess_images.py
│   ├── train.py
│   └── evaluate.py
│
├── checkpoints/
│   └── best.pt
│
└── results/
    ├── metrics.json
    ├── training_history.json
    ├── rgb_prediction_00.png
    ├── rgb_prediction_01.png
    ├── rgb_prediction_02.png
    ├── rgb_prediction_03.png
    ├── rgb_prediction_04.png
    ├── rgb_error_per_step.png
    └── ssim_per_step.png
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/zaryanaliansari9/RH20T-RGB-Query-World-Model.git
cd RH20T-RGB-Query-World-Model
```

### 2. Create a Python environment

Python 3.10+ is recommended.

For example:

```bash
python3 -m venv iml_env
source iml_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with your CUDA installation.

Verify PyTorch and CUDA:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

A CUDA-capable NVIDIA GPU is strongly recommended for training.

## Dataset Preparation

The repository does not contain the RH20T dataset or the generated preprocessing cache.

You must provide the required RH20T source data locally before running the preprocessing pipeline.

The preprocessing pipeline consists of the following stages.

### Stage 1 — Build the clean aligned dataset

This stage combines aligned RGB information, robot state information, transition information, and RGB frames into a consistent per-scene representation.

Run:

```bash
python scripts/25_build_clean_aligned_all_scenes.py \
    --rh20t-project /path/to/RH20T_project \
    --rgb-dir /path/to/RH20T_OnlineWorldModel/dataset/processed/rgb_frames \
    --output-dir /path/to/prepared/RH20T/clean_aligned
```

Replace the paths with the locations of your local RH20T data.

The output contains the cleaned aligned scenes used by the next preprocessing stage.

### Stage 2 — Build the multistep world-model dataset

This stage constructs the history/action/target sequences used by the world model.

The model uses:

- history length = 16
- prediction horizon = 10
- action dimension = 6
- TCP dimension = 7

Run:

```bash
python scripts/24_build_all_scenes_multistep_dataset.py \
    --data-root /path/to/prepared/RH20T
```

This creates:

```text
/path/to/prepared/RH20T/multistep_world_model/
```

### Stage 3 — Build the global multistep index

Create the global sample index:

```bash
python scripts/25_build_multistep_index.py \
    --data-root /path/to/prepared/RH20T
```

This creates:

```text
/path/to/prepared/RH20T/multistep_index/samples.npy
```

and:

```text
/path/to/prepared/RH20T/multistep_index/summary.json
```

### Stage 4 — Create the train/validation/test split

Run:

```bash
python scripts/27_prepare_multistep_training.py \
    --data-root /path/to/prepared/RH20T
```

The split is performed by scene rather than by individual sample.

This prevents temporal samples originating from the same scene from leaking between train, validation, and test sets.

The script also computes:

- action mean
- action standard deviation
- TCP mean
- TCP standard deviation

using only the training split.

The resulting files are stored in:

```text
/path/to/prepared/RH20T/multistep_training/
```

### Stage 5 — Preprocess and cache RGB images

The training pipeline uses a memory-mapped image cache instead of repeatedly decoding the original image files.

Run:

```bash
python scripts/29_preprocess_images.py \
    --data-root /path/to/prepared/RH20T \
    --num-workers 2
```

The script:

1. reads the RGB images,
2. converts them to RGB,
3. resizes them to 128 × 128,
4. converts HWC images to CHW format,
5. stores the images in a memory-mapped `images.dat` file,
6. creates an index mapping image paths to cache locations.

The cache is stored under:

```text
/path/to/prepared/RH20T/preprocessed_images/
```

This cache can be large. Make sure sufficient disk space is available before running the preprocessing step.

## Training

Once the dataset preparation is complete, the model can be trained directly from the repository.

Example:

```bash
python scripts/train.py \
    --data-root /path/to/prepared/RH20T \
    --output-dir ./training_output \
    --epochs 2 \
    --batch-size 16 \
    --num-workers 2
```

The training script automatically selects CUDA when available.

For CPU-only systems, training is possible but will be substantially slower.

### Training Arguments

The main command-line options are:

#### `--data-root`

Path to the prepared RH20T dataset containing:

```text
multistep_index/
multistep_training/
preprocessed_images/
```

#### `--output-dir`

Directory where checkpoints and training history are written.

#### `--epochs`

Number of training epochs.

#### `--batch-size`

Training batch size.

#### `--num-workers`

Number of PyTorch DataLoader workers.

For example:

```bash
python scripts/train.py \
    --data-root /path/to/prepared/RH20T \
    --output-dir ./training_output \
    --epochs 2 \
    --batch-size 16 \
    --num-workers 2
```

### Memory Considerations

The model itself contains approximately 3.1 million trainable parameters.

However, GPU memory usage is dominated by the Transformer activations, image tensors, batch size, and autoregressive prediction.

If you encounter CUDA out-of-memory errors, reduce the batch size:

```bash
python scripts/train.py \
    --data-root /path/to/prepared/RH20T \
    --output-dir ./training_output \
    --epochs 2 \
    --batch-size 4 \
    --num-workers 2
```

The batch size can be reduced further if necessary.

## Evaluation

The repository includes the trained RGB query token checkpoint:

```text
checkpoints/best.pt
```

To evaluate the checkpoint:

```bash
python scripts/evaluate.py \
    --data-root /path/to/prepared/RH20T \
    --checkpoint checkpoints/best.pt \
    --output-dir ./evaluation \
    --num-samples 100
```

For evaluation on the complete test set:

```bash
python scripts/evaluate.py \
    --data-root /path/to/prepared/RH20T \
    --checkpoint checkpoints/best.pt \
    --output-dir ./evaluation
```

The evaluation script uses the test split generated during dataset preparation.

### Evaluation Metrics

The evaluator reports:

- RGB MSE
- RGB MAE
- SSIM
- PSNR
- t+1 RGB MSE
- t+10 RGB MSE

The `--num-samples` argument can be used for a quick evaluation.

For example:

```text
--num-samples 100
```

evaluates 100 randomly selected test samples.

If the argument is omitted, the complete test set is evaluated.

## Reported Results

The final checkpoint was evaluated on 47,023 unseen test samples.

| Metric   | Result      |
| -------- | ----------- |
| RGB MSE  | 0.005411    |
| RGB MAE  | 0.031076    |
| SSIM     | 0.846749    |
| PSNR     | 22.6673 dB  |
| t+1 MSE  | 0.004916    |
| t+10 MSE | 0.006139    |

These results correspond to the checkpoint and evaluation protocol used for the reported experiment.

A quick 100-sample evaluation will produce slightly different aggregate values because it evaluates a subset of the test set.

For example, the repository checkpoint produced approximately:

```text
RGB MSE: 0.005543
RGB MAE: 0.030885
SSIM: 0.846612
PSNR: 22.5626 dB
t+1 MSE: 0.005070
t+10 MSE: 0.006229
```

on a 100-sample evaluation.

## Reproducing the Reported Experiment

To reproduce the complete reported experiment:

1. Prepare the RH20T data.
2. Build the clean aligned dataset.
3. Build the multistep world-model dataset.
4. Build the multistep sample index.
5. Create the scene-level train/validation/test split.
6. Preprocess the RGB images into the memory-mapped cache.
7. Train the model using the same model configuration.
8. Evaluate on the complete test split.

The reported configuration uses:

- History: 16
- Prediction horizon: 10
- Image size: 128 × 128
- Spatial grid: 4 × 4
- Transformer dimension: 256
- Attention heads: 4
- Transformer layers: 2
- Feed-forward dimension: 512
- Dropout: 0.1
- Batch size: 16
- Learning rate: 1e-4
- Epochs: 2
- Random seed: 42

## Training Objective

The RGB prediction model is trained using an RGB reconstruction objective combining L1 loss and SSIM.

The objective is:

```text
L = 0.7 L1 + 0.3 (1 - SSIM)
```

The model therefore focuses on both pixel-level reconstruction accuracy and structural similarity.

## Checkpoint

The repository contains the trained checkpoint:

```text
checkpoints/best.pt
```

The checkpoint is stored using Git LFS.

The checkpoint corresponds to the RGB query token architecture implemented in:

```text
src/rgb_query_world_model.py
```

The training and evaluation scripts contain the corresponding model implementation required to load the checkpoint.

## Important Notes

### Dataset paths

The repository scripts use command-line paths rather than hard-coded paths from the original development machine.

Therefore, another user should not need to modify the source code simply because their RH20T dataset is stored in a different directory.

Use:

```text
--data-root
```

to specify the location of the prepared dataset.

### Dataset is not included

The RH20T dataset is not distributed with this repository.

The generated files are also intentionally excluded because they can be very large, particularly:

```text
dataset/multistep_world_model/
dataset/multistep_index/
dataset/multistep_training/
dataset/preprocessed_images/
```

The memory-mapped image cache can contain hundreds of thousands of RGB images and should be generated locally.

### Training from scratch vs checkpoint evaluation

There are two ways to use this repository.

#### Evaluate the provided model

Use:

```bash
python scripts/evaluate.py \
    --data-root /path/to/prepared/RH20T \
    --checkpoint checkpoints/best.pt \
    --output-dir ./evaluation
```

#### Train a new model

Use:

```bash
python scripts/train.py \
    --data-root /path/to/prepared/RH20T \
    --output-dir ./training_output \
    --epochs 2 \
    --batch-size 16 \
    --num-workers 2
```

## Project Structure

The repository is organized into three main components.

### Model

```text
src/rgb_query_world_model.py
```

Contains the RGB query token world-model architecture.

### Data preprocessing

```text
scripts/25_build_clean_aligned_all_scenes.py
scripts/24_build_all_scenes_multistep_dataset.py
scripts/25_build_multistep_index.py
scripts/27_prepare_multistep_training.py
scripts/29_preprocess_images.py
```

These scripts transform the RH20T data into the format expected by the model.

### Training and evaluation

```text
scripts/train.py
scripts/evaluate.py
```

These scripts train and evaluate the RGB query token world model.
