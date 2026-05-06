# BirdCLEF+ 2026 -- Species Identification from Acoustic Recordings
**Abdullah Sheikh | Sorbonne Universite | M1 Computer Science**

---

## 1. Problem

BirdCLEF+ 2026 is a Kaggle competition focused on acoustic species identification in
the Pantanal wetlands of Brazil. Passive acoustic monitoring (PAM) devices are deployed
across the ecosystem and record continuous audio. The task is to detect which wildlife
species are present in each 5-second window of those recordings.

This is a **multilabel classification** problem: multiple species can be active at the
same time in the same window. The output is a probability score for each of 234 species,
and the competition metric is **macro-averaged ROC-AUC** -- the mean AUC across all
species, weighted equally regardless of how many examples each species has.

The constraint for this university submission is to use **classical machine learning
only** (no neural networks).

---

## 2. Dataset

The dataset consists of two sources:

| Source | Description | Size |
|---|---|---|
| `train_audio/` | Short labeled clips (5-260s), one species per clip | 35,549 clips, 206 species |
| `train_soundscapes/` | Long unlabeled field recordings | 10,658 files |
| `train_soundscapes_labels.csv` | Labeled 5s windows from soundscapes | 1,478 windows |
| `taxonomy.csv` | Full list of species to predict | 234 species |

Key facts from exploratory analysis:

- **206 of 234 species** have labeled training clips. The remaining 28 (mostly insects
  and amphibians) have no training audio at all -- zero-shot species.
- **Animal class breakdown**: Birds dominate (34,799 clips). Non-bird classes have far
  fewer examples: Amphibia 451, Insecta 199, Mammalia 99, Reptilia 1.
- **Multilabel structure**: Soundscape windows contain 4.22 species on average (max 10).
- **Domain gap**: Training clips are clean, single-species recordings from Xeno-canto
  and iNaturalist. Test soundscapes are noisy field recordings with multiple overlapping
  species -- a fundamentally different distribution.
- Approximately 4% of iNaturalist files are silent and are skipped during preprocessing.

---

## 3. Pre-Processing

**Audio loading**: Each training clip is center-cropped to 5 seconds. If shorter than
5 seconds, it is zero-padded to reach the target length. Silent clips (max absolute
amplitude < 0.0001) are skipped.

**Audio config**:
```
Sample rate  : 32,000 Hz
Duration     : 5 seconds
Target length: 160,000 samples
```

**Train/validation split**: Soundscape recordings are split at the **recording level**
-- all 5-second windows from the same recording file go entirely to either train or
validation, never both. This prevents data leakage: a model that memorises one window
from a recording cannot trivially score well on a different window from the same file.
20% of soundscape recordings are held out for validation (294 windows).

**Soundscape oversampling**: The soundscape windows are repeated 5 times in the training
set (`SC_REPEATS = 5`). This partially compensates for the domain gap: the training
data is dominated by clean clips (35,538), and without oversampling the model barely
sees any multi-species noisy examples.

**Label filtering**: Training clips with species that appear fewer than 2 times are
excluded (35,538 clips remain from 35,549).

---

## 4. Feature Extraction

Classical ML models cannot operate directly on raw audio or spectrograms. A fixed-length
feature vector is extracted from each 5-second window using **MFCC-based statistics**.

**Feature set (82 features from cached data, 122 with fresh extraction):**

| Feature group | Count | What it captures |
|---|---|---|
| MFCC mean | 20 | Average spectral envelope shape |
| MFCC std | 20 | Variability of spectral shape over time |
| MFCC max | 20 | Peak spectral shape -- catches brief calls diluted by mean |
| Delta MFCC mean | 20 | Average rate of spectral change (temporal dynamics) |
| Delta MFCC std | 20 | Variability of temporal dynamics |
| Delta MFCC max | 20 | Peak rate of change |
| Spectral centroid mean | 1 | Perceived brightness of the sound |
| Zero crossing rate mean | 1 | Noisiness / tonality ratio |

**Why MFCC**: MFCCs are the standard compact representation of audio timbre. They
summarise the shape of the mel-frequency spectrum using a small number of coefficients,
and they are robust to pitch shifts -- two birds of the same species singing at slightly
different pitches will produce similar MFCCs.

**Why add max statistics**: Soundscape windows contain multiple overlapping species with
brief calls. Mean-only statistics get diluted when a species is only active for 0.5s of
a 5s window. The max statistic captures the peak of brief calls that the mean misses.

**Feature extraction is cached** to a `.npz` file on first run. Subsequent runs load
from cache, reducing startup time from ~2 hours to under 5 seconds.

---

## 5. Models and Training

The multilabel problem is handled using **Binary Relevance**: one independent binary
classifier is trained per species. For 234 species, this means 234 classifiers.

Three classifiers are compared:

### Logistic Regression
- Solver: `lbfgs`, `max_iter=1000`
- `class_weight='balanced'` -- compensates for severe class imbalance (~0.4% positive
  rate per species)
- Features are standardised (zero mean, unit variance) before training
- Only trained on species with at least one positive training example (230/234)

### Random Forest
- 100 trees, `max_depth=15`, `min_samples_leaf=5`
- Native multi-output support -- trains one forest on all 234 columns simultaneously
- No feature scaling needed
- Depth and leaf size limits prevent overfitting on rare species

### Linear SVM
- `LinearSVC` wrapped in `CalibratedClassifierCV` for probability output
- `max_iter=2000`
- Features standardised (same as LR)
- No `class_weight='balanced'` -- causes excessive training time with 234 classifiers
  due to very small effective C for imbalanced classes

**Training data**: 35,538 clips + 1,184 soundscape windows x5 = 41,458 total rows.

---

## 6. Validation and Inference

### Validation setup

Models are evaluated on 294 held-out soundscape windows (the 20% recording-level split
from `train_soundscapes_labels.csv`). These windows represent the closest proxy to the
hidden test set: noisy, multi-species, real-world field recordings.

**Detection Score (macro ROC-AUC)**: For each species, the area under the ROC curve
measures how well the model ranks true positives above true negatives. The macro average
weights each species equally. A score of 1.0 is perfect; 0.5 is random.

Only species with at least one positive example in the validation set are scored. With
a single 80/20 split, ~40 of 234 species appear in validation -- the rest are silently
skipped.

### 5-fold cross-validation

To get a more reliable estimate, **5-fold recording-level cross-validation** is run on
Logistic Regression. Each fold holds out a different 20% of soundscape recordings, so
across all 5 folds every recording is evaluated exactly once. This covers more species
(32-44 per fold) and gives a stable mean and standard deviation.

RF and SVM are excluded from CV: each RF fold takes ~7 minutes and SVM ~4 minutes,
making 5-fold CV impractical within the time budget.

### Inference

At prediction time, each model's `predict_proba` output is the positive-class
probability for each species. For zero-shot species (no training data), the probability
is fixed at 0.

---

## 7. Results Comparison

### Single validation split (294 soundscape windows)

| Classifier | Detection Score | Train time |
|---|---|---|
| Random Forest | **0.7564** | 442.5s |
| Logistic Regression | 0.7258 | 43.4s |
| Linear SVM | 0.7087 | 265.8s |

### 5-fold cross-validation -- Logistic Regression

| Fold | Detection Score | Windows | Species scored |
|---|---|---|---|
| 1 | 0.7620 | 298 | 32 |
| 2 | 0.8495 | 294 | 36 |
| 3 | 0.7905 | 296 | 36 |
| 4 | 0.7251 | 292 | 44 |
| 5 | 0.8290 | 298 | 35 |
| **Mean** | **0.7912 +/- 0.0448** | | |

### Top-5 features (Random Forest importance)

| Feature | Importance |
|---|---|
| MFCC0_mean | 0.0855 |
| MFCC9_mean | 0.0668 |
| MFCC15_mean | 0.0465 |
| MFCC2_mean | 0.0390 |
| MFCC5_mean | 0.0375 |

The RF relies primarily on mean MFCC coefficients -- the average spectral shape of the
clip. MFCC0 (average energy/loudness) is the single most discriminative feature.

### Discussion

- **RF outperforms LR and SVM** on the single split. RF handles non-linear feature
  interactions naturally and is less sensitive to the class imbalance problem.
- **LR is competitive at 13x less training time** -- a meaningful trade-off if training
  budget is limited.
- **SVM scores lowest** despite comparable training time to RF. Without
  `class_weight='balanced'` (removed due to timeout), the SVM treats rare species as
  noise and under-predicts them.
- **The single-split score (~0.72-0.76) slightly overestimates true performance**: only
  ~40 well-represented species are scored, and these tend to be the species the model
  handles best. The CV estimate (0.7912) spans more species but still covers under 20%
  of the 234 target species.

---

## 8. Final Conclusion

A classical ML pipeline using MFCC-based features and Binary Relevance classifiers
achieves a Detection Score of **0.7564** (RF) on held-out soundscape windows, with a
cross-validated estimate of **0.7912 +/- 0.0448** for Logistic Regression.

**Main limitations**:

1. **Domain gap**: The model is trained mostly on clean single-species clips but
   evaluated on noisy multi-species soundscapes. This is the largest source of
   error and cannot be fully addressed without soundscape-specific training data.

2. **Feature compactness**: 82-122 hand-crafted features cannot capture all the
   discriminative information in a 5-second audio clip. A spectrogram-based deep
   learning model has access to the full time-frequency representation.

3. **Zero-shot species**: 28 species have no training data. All are predicted at 0.0
   probability, contributing to macro-AUC degradation when those species appear in
   the test set.

4. **Binary Relevance ignores label co-occurrence**: Species that frequently appear
   together (e.g. species sharing a habitat) could benefit from a classifier that
   models label dependencies.

**What would improve the score**:

- Use `train_soundscapes_labels.csv` more aggressively -- the 1,478 soundscape windows
  are the only in-domain training data and are currently underweighted.
- Add pitch-invariant features or augmentation (time-shift, noise injection).
- Use a Classifier Chain or Label Powerset instead of Binary Relevance to capture
  label co-occurrence.
- Extract features from multiple 5s segments per clip instead of a single center crop.
