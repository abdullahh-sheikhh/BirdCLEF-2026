# BirdCLEF+ 2026

Acoustic species identification for the Pantanal wetlands. Kaggle competition + university semester project.

**Competition**: [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) | Deadline: June 3, 2026  
**Student**: Abdullah Sheikh | Sorbonne Universite | MS Computer Science  
**Metric**: Macro-averaged ROC-AUC

---

## Problem

Identify 234 wildlife species (birds, insects, amphibians, mammals) from passive acoustic monitoring (PAM) recordings in Brazil's Pantanal wetlands. Each test file is a long soundscape sliced into 5-second windows -- output is 234 probabilities per window (multilabel).

Key challenges:
- **Domain gap**: training data is clean single-species clips; test data is noisy overlapping field recordings
- **Zero-shot species**: 28 species (mostly insects and amphibians) have no training audio
- **Class imbalance**: Aves 34,799 clips vs. Reptilia 1 clip
- **Constraint**: CPU-only inference at submission, must complete in 90 minutes

---

## Dataset

| Split | Files | Notes |
|---|---|---|
| `train_audio/` | 35,549 clips | 206 species, 5-260s .ogg files |
| `train_soundscapes/` | 10,658 recordings | Unlabeled long field recordings |
| `train_soundscapes_labels.csv` | 1,478 windows | 5s labeled windows, avg 4.22 species/window |
| `test_soundscapes/` | Hidden | Populated only at submission time |
| `taxonomy.csv` | 234 species | All species to predict |

Data sources: Xeno-canto (birds, quality-rated) and iNaturalist (non-birds, unrated). ~4% of iNat files are silent.

---

## Two-Track Implementation

This project has two separate notebooks -- one for Kaggle submission (deep learning) and one for university credit (classical ML only).

### Track 1 -- Kaggle Submission (`birdclef-2026.ipynb`)

**Approach**: Transfer learning on mel-spectrograms with EfficientNet-B0.

**Pipeline**:
```
5s audio chunk
    -> mel-spectrogram (128 x 313, normalized [0,1])
    -> stack to 3 channels
    -> EfficientNet-B0 (pretrained ImageNet, fine-tuned)
    -> 234-dim sigmoid output (BCEWithLogitsLoss)
```

**Audio config**: SR=32000, N_MELS=128, N_FFT=1024, HOP_LENGTH=512, FMIN=50, FMAX=14000  
**Training**: 5 epochs, Adam lr=1e-3, batch=64, T4 GPU  
**Spectrograms**: Precomputed to `.npy` -- 8.7s/batch (raw audio) reduced to 0.01s/batch  
**Validation**: ROC-AUC on held-out soundscape windows (closest proxy to test)

**Results**:

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.0288 | 0.0185 |
| 2 | 0.0150 | 0.0134 |
| 3 | 0.0113 | 0.0113 |
| 4 | 0.0092 | 0.0105 |
| 5 | 0.0076 | 0.0098 |

**Validation ROC-AUC: 0.6144** (5 epochs, baseline)

Why EfficientNet-B0: small enough for 90-min CPU inference (~4.3M params), proven in past BirdCLEF competitions, ImageNet pretraining generalizes to spectrogram edge/texture patterns.

### Track 2 -- University Submission (`university-submission/`)

Classical ML only -- no neural networks. See [university-submission/README.md](university-submission/README.md) for full details.

**Best result: Random Forest ROC-AUC 0.7564 | Logistic Regression 5-fold CV: 0.7912 +/- 0.0448**

---

## Submission Workflow

1. Train on Kaggle (GPU enabled), save `best_model.pt`
2. Upload `best_model.pt` to a Kaggle Dataset
3. In notebook: set `TRAINING_MODE = False`, `USE_GPU = False`
4. Commit notebook -- Kaggle runs inference only, outputs `submission.csv`

---

## Pending Improvements

- More epochs (loss still decreasing at epoch 5, try 10-15)
- Use `train_soundscapes_labels` as additional training data (direct domain gap mitigation)
- Audio augmentation: gaussian noise, time shift, frequency masking
- Larger model: EfficientNet-B1 or B2
- Learning rate schedule: CosineAnnealingLR
- Quality filtering: XC clips with rating >= 3 only

---

## Project Structure

```
BirdCLEF-2026/
├── birdclef-2026.ipynb          -- Kaggle deep learning notebook
├── CLAUDE.md                    -- Full development context
└── university-submission/
    ├── birdclef-university.ipynb -- Classical ML notebook
    ├── report.md                 -- Academic report
    └── figures/                  -- Plots and visualizations
```