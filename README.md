# RH20T RGB Query Token World Model

A multimodal world model for robotic manipulation trained on the RH20T dataset.

The model predicts future RGB observations from a history of RGB observations, robot TCP state, and future actions. The architecture combines spatial visual tokens with learned RGB query tokens for autoregressive multi-step prediction.

## Model

- Dataset: RH20T
- History length: 16 frames
- Prediction horizon: 10 steps
- RGB resolution: 128 × 128
- Spatial visual grid: 4 × 4
- Visual tokens per frame: 16
- Learned RGB query tokens: 16
- Transformer embedding dimension: 256
- Transformer heads: 4
- Transformer layers: 2
- Output: future RGB observations

### RGB Query Tokens

The model introduces 16 learned RGB query tokens corresponding to the 4 × 4 output latent structure.

The query tokens attend to the spatiotemporal observation history and the action token. Their resulting representations are transformed into the future RGB latent representation and decoded into the predicted image.

The learned query tokens are not fixed image patches. They are trainable queries that aggregate information required to construct the future visual observation.

## Results

Evaluation was performed on 47,023 unseen test samples.

| Metric | Result |
|---|---:|
| RGB MSE | 0.005411 |
| RGB MAE | 0.031076 |
| SSIM | 0.846749 |
| PSNR | 22.6673 dB |
| t+1 MSE | 0.004916 |
| t+10 MSE | 0.006139 |

## Repository Structure

```text
RH20T-RGB-Query-World-Model/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── LICENSE
├── src/
│   └── rgb_query_world_model.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── checkpoints/
│   └── best.pt
└── results/
    ├── training_history.json
    ├── metrics.json
    ├── rgb_prediction_00.png
    ├── rgb_prediction_01.png
    ├── rgb_prediction_02.png
    ├── rgb_prediction_03.png
    ├── rgb_prediction_04.png
    ├── rgb_error_per_step.png
    └── ssim_per_step.png
