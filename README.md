# RH20T RGB Query Token World Model

An action-conditioned visual world model for robotic manipulation based on the
[RH20T](https://rh20t.github.io/) dataset.

The model predicts future RGB observations from:

- a history of RGB observations,
- robot TCP state,
- and a sequence of future actions.

The architecture combines spatial visual tokens, temporal Transformer attention,
and learned RGB query tokens for multi-step visual prediction.

---

## Overview

The model uses a 16-frame observation history and predicts the next 10 RGB
observations autoregressively.

### Key components

1. **Spatial visual encoding**
   - Each RGB frame is encoded into a `4 × 4` spatial token grid.
   - This produces 16 visual tokens per frame.

2. **Robot-state encoding**
   - The robot TCP state is represented as a separate token for each history frame.

3. **Action conditioning**
   - Future actions are represented using a dedicated action token.

4. **Transformer dynamics**
   - Spatial and temporal information from the observation history is processed
     jointly using a Transformer encoder.

5. **Learned RGB query tokens**
   - 16 trainable query tokens are appended to the Transformer input.
   - These queries aggregate information from the observation history and action.
   - Their output representations are converted into a `4 × 4` future RGB latent
     representation and decoded into the predicted image.

The RGB query tokens are **not fixed image patches**. They are learned query
representations used to extract the information required for constructing the
future RGB observation.

---

# Model Configuration

| Parameter | Value |
|---|---:|
| Dataset | RH20T |
| History length | 16 frames |
| Prediction horizon | 10 steps |
| RGB resolution | 128 × 128 |
| Spatial grid | 4 × 4 |
| Visual tokens / frame | 16 |
| Learned RGB query tokens | 16 |
| Transformer dimension | 256 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Output | Future RGB observations |

---

# Repository Structure

```text
RH20T-RGB-Query-World-Model/
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
├── scripts/
│   ├── train.py
│   └── evaluate.py
│
├── src/
│   └── rgb_query_world_model.py
│
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
