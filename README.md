# Image-Vector Conditional Generation Model Evaluation Report

## 1. Project Overview

This project evaluates a lightweight FiLM-conditioned U-Net model for image reconstruction and enhancement. The model operates in the Lab color space and integrates multi-dimensional imaging metadata (camera settings + spectral features) as conditional inputs.

### Inputs
- RGB images
- 7-dimensional condition vectors

### Output
- Enhanced RGB images (via residual prediction)

---

## 2. Dataset Structure

### Input Images
inputs_dir/*.jpg

### Target Images
targets_dir/*.jpg

### Condition Vector (vector.json)

Each sample contains a 7D vector:

- aperture
- exposure_time
- iso
- band_G
- band_R
- band_NIR
- band_RE

---

## 3. Data Preprocessing

### RGB ↔ Lab Conversion

The model operates in Lab color space:

- RGB → Lab: cv2.COLOR_RGB2LAB
- Lab → RGB: cv2.COLOR_LAB2RGB

Normalization range: [0, 1]

---

## 4. Model Architecture

### FiLM Module

Feature-wise Linear Modulation:

x' = x × (1 + γ(v)) + β(v)

where:
- v: 7-dimensional condition vector
- γ, β: learned linear projections

---

### Network Structure

Encoder → Bottleneck → Decoder → Residual Output

Final output:
Output = Input + Residual

---

## 5. Evaluation Metrics

### 5.1 Color Error (Lab ΔE approximation)

ΔE = sqrt((L1-L2)^2 + (a1-a2)^2 + (b1-b2)^2)

Final metric:
Color Error = mean(ΔE) × 255

---

### 5.2 MAPE

MAPE = |pred - target| / (|target| + ε) × 100%

---

## 6. Inference Efficiency

- Total inference time
- Time per sample
- FPS (frames per second)

---

## 7. Outputs

- Color Error (Lab space)
- MAPE (%)
- Latency
- FPS



---

## 8. Summary

This system achieves:

Image + acquisition conditions → high-quality reconstruction in Lab space

while balancing:

- Accuracy
- Interpretability
- Computational efficiency
