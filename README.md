# RH20T RGB Query Token World Model

An action-conditioned visual world model for robotic manipulation based on the
RH20T dataset.

The model predicts future RGB observations from:

- a history of RGB observations,
- robot TCP state,
- and future actions.

The architecture combines spatial visual tokens, Transformer-based temporal
dynamics, and learned RGB query tokens for autoregressive multi-step visual
prediction.

---

## Overview

The model receives a 16-frame observation history and predicts the next 10 RGB
observations autoregressively.

### Architecture

Each RGB observation is encoded into a spatial `4 × 4` grid of visual tokens.

For each historical timestep, the model uses:

- 16 spatial visual tokens
- 1 TCP-state token

The future action sequence is represented using an action token.

The Transformer processes the complete spatiotemporal history together with the
future action information. In addition, 16 learned RGB query tokens are appended
to the Transformer input.

The final query representations are mapped to a `4 × 4` RGB latent
representation and decoded into the predicted RGB observation.

The learned RGB query tokens are not fixed image patches. They are trainable
query representations that aggregate information required to construct the
future visual observation.

---

# Model Configuration

| Parameter | Value |
|---|---:|
| Dataset | RH20T |
| History length | 16 frames |
| Prediction horizon | 10 steps |
| RGB resolution | 128 × 128 |
| Spatial grid | 4 × 4 |
| Visual tokens per frame | 16 |
| Learned RGB query tokens | 16 |
| Transformer dimension | 256 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Feed-forward dimension | 512 |
| Dropout | 0.1 |
| Action dimension | 6 |
| TCP dimension | 7 |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Training epochs | 2 |
| Random seed | 42 |

---

# Repository Structure

```text
RH20T-RGB-Query-World-Model/
│
├── src/
│   └── rgb_query_world_model.py
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   └── evaluate.py
│
├── checkpoints/
│   └── best.pt
│
├── results/
│   ├── metrics.json
│   ├── training_history.json
│   ├── rgb_prediction_00.png
│   ├── rgb_prediction_01.png
│   ├── rgb_prediction_02.png
│   ├── rgb_prediction_03.png
│   ├── rgb_prediction_04.png
│   ├── rgb_error_per_step.png
│   └── ssim_per_step.png
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
└── .gitattributes
