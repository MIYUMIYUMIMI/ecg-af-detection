# ECG Atrial Fibrillation Detection

Deep-learning-based atrial fibrillation detection from single-lead ECG signals using the PhysioNet/Computing in Cardiology Challenge 2017 dataset.

This project compares multiple neural architectures, ensemble strategies, and explainability methods for imbalanced AF classification. It also includes an Adaptive Confidence-Guided Ensemble (ACGE), which learns sample-specific weights over multiple base models.

## Dataset

The experiments use the PhysioNet/CinC Challenge 2017 single-lead ECG dataset.

After label filtering and preprocessing:

- 8,244 recordings
- 738 AF samples
- 7,506 Non-AF samples
- Sampling rate: 300 Hz
- Fixed input length: 9,000 samples (30 seconds)
- Per-record z-score normalisation

For the seed-42 split:

- Training: 5,770
- Validation: 1,237
- Test: 1,237

## Models

The following models were evaluated:

- Regularised 1D CNN
- CNN-LSTM
- CNN-GRU
- ResNet1D
- Equal GRU-ResNet fusion
- Fixed weighted fusion
- Logistic-regression stacking
- Adaptive Confidence-Guided Ensemble (ACGE)

## Adaptive Confidence-Guided Ensemble

ACGE combines three frozen base experts:

- Baseline 1D CNN
- CNN-GRU
- ResNet1D

For each ECG sample, a 13-dimensional gating feature vector is constructed from:

- 3 base-model probabilities
- 3 confidence features
- 3 pairwise disagreement features
- 4 probability summary statistics:
  - mean
  - standard deviation
  - maximum
  - minimum

The gating network uses:

```text
13-D input
   ↓
Dense(16, ReLU)
   ↓
Dropout(0.20)
   ↓
Dense(8, ReLU)
   ↓
Softmax(3)
   ↓
Sample-specific expert weights