# BirdCLEF+ 2026 -- University Submission

Classical ML solution for acoustic species identification. Constraint: no neural networks.

**Course**: Machine Learning | Sorbonne Universite | MS Computer Science  
**Student**: Abdullah Sheikh  
**Metric**: Macro-averaged ROC-AUC on soundscape windows

---

## Problem

Multilabel classification: given a 5-second audio window from a field recording, predict which of 234 species are active. Training data is clean single-species clips; test data is noisy overlapping soundscapes -- this domain gap is the central challenge.

Constraint imposed for university: classical ML only (no neural networks, no pretrained deep models).

---

## Approach

### Feature Extraction

122 hand-crafted features per 5-second window:

| Feature | Count | Rationale |
|---|---|---|
| MFCC mean/std/max (20 coefficients x 3 stats) | 60 | Compact timbral representation |
| Delta MFCC mean/std/max (20 coefficients x 3 stats) | 60 | Temporal dynamics of timbre |
| Spectral centroid mean | 1 | Average brightness of sound |
| Zero crossing rate mean | 1 | Noisiness / percussiveness |

Max statistics are included alongside mean/std because brief calls get diluted by mean averaging in long overlapping windows.

Features are cached after first extraction (~2h -> ~5s on reruns).

### Multi-label Strategy

Binary Relevance: one independent classifier per species (234 classifiers total). Simple, parallelizable, interpretable -- ignores label co-occurrence, which is a known limitation.

### Classifiers

**Logistic Regression**
- `lbfgs` solver, `max_iter=1000`, `class_weight='balanced'`
- Features standardized (StandardScaler)
- Handles ~0.4% positive rate per species via balanced class weights

**Random Forest**
- 100 trees, `max_depth=15`, `min_samples_leaf=5`
- Native multi-output (single fit for all 234 species)
- No feature scaling needed

**Linear SVM**
- `LinearSVC` wrapped in `CalibratedClassifierCV` for probability output
- `max_iter=2000`, features standardized

### Data Preprocessing

- Audio: center-crop or zero-pad to exactly 5 seconds, skip silent files (max amplitude < 0.0001)
- Train/validation split: recording-level (20% of recordings held out, not clips) -- prevents leakage
- Soundscape windows oversampled 5x (`SC_REPEATS=5`) to compensate for domain imbalance
- Species with < 2 training clips dropped (35,538 clips remain)
- Training set: 35,538 clips + 1,184 soundscape windows x 5 = **41,458 total rows**
- Validation: 294 held-out soundscape windows (closest proxy to test conditions)

---

## Results

### Classifier Comparison (Single Split)

| Classifier | ROC-AUC | Train Time |
|---|---|---|
| Random Forest | **0.7564** | 442.5s |
| Logistic Regression | 0.7258 | 43.4s |
| Linear SVM | 0.7087 | 265.8s |

### 5-Fold Cross-Validation (Logistic Regression)

Recording-level stratified CV -- each fold scored only on species with >= 1 positive in that fold's validation set.

| Fold | ROC-AUC | Species scored |
|---|---|---|
| 1 | 0.7620 | 32 |
| 2 | 0.8495 | 36 |
| 3 | 0.7905 | 36 |
| 4 | 0.7251 | 44 |
| 5 | 0.8290 | 35 |
| **Mean** | **0.7912 +/- 0.0448** | -- |

### Top Feature Importances (Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | MFCC0_mean | 0.0855 |
| 2 | MFCC9_mean | 0.0668 |
| 3 | MFCC15_mean | 0.0465 |
| 4 | MFCC2_mean | 0.0390 |
| 5 | MFCC5_mean | 0.0375 |

MFCC0 (average energy/loudness) dominates -- detecting whether a species is vocalizing at all is the first discriminative signal.

---

## Limitations

- **Domain gap**: features extracted from clean clips generalize imperfectly to noisy soundscapes
- **Zero-shot species**: 28 species with no training audio are predicted at 0.0 probability, dragging macro-AUC down
- **Label independence**: Binary Relevance ignores co-occurrence patterns (e.g., species that always appear together)
- **Feature compactness**: 122 features vs. full spectrogram -- some species-specific patterns are lost

---

## Files

| File | Description |
|---|---|
| `birdclef-university.ipynb` | Full notebook: EDA -> feature extraction -> training -> evaluation |
| `report.md` | Academic report (8 sections, full methodology and analysis) |
| `figures/data.png` | Dataset overview -- class breakdown and top-20 species |
| `figures/spectogram.png` | Audio visualization -- waveform, mel-spectrogram, MFCCs |
| `figures/classifier_comparison.png` | RF vs LR vs SVM bar chart + top-10 feature importances |
| `figures/cv_folds.png` | 5-fold CV results per fold |
| `figures/feature_importance.png` | Random Forest feature importance ranking |