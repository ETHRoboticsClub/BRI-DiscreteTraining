# HRI — discrete training

Train gesture classifiers on forearm sEMG: filtering → windowing → hand-crafted features → normalization →
decoder (LDA or any scikit-learn classifier). It holds the shared Group 2 modules and the Group 3 classifier, so
the offline features match the calibration and online paths exactly.

Continuous decoding (joint angles, TRR) lives in `HRI-ContinuousTraining`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/fetch_ninapro.py --subjects 1-10   # or dvc pull, once the data is pushed
```

## Structure

```
params.yaml                  all parameters, one section per module
dvc.yaml                     pipeline: build_features -> train
scripts/
  fetch_ninapro.py           NinaPro DB2 -> data/raw/ninapro_db2 (recording format below)
  build_features.py          recordings -> labeled feature windows, σ_rest and signal quality per recording
  train.py                   evaluation protocol + one calibration profile per user
src/
  config.py                  params.yaml -> typed, validated config + config hash        (shared)
  filtering.py               causal band-pass + notch, state per channel                   (shared)
  windowing.py               ring buffer -> overlapping windows with end timestamps        (shared)
  feature_extraction.py      feature registry: MAV, WL, WAMP, AR, MNF, MDF                  (shared)
  normalization.py           identity / z-score, fitted on calibration or training data    (shared)
  pipeline.py                Pipeline, Recording, FeatureBatch                             (shared)
  signal_quality.py          saturation, flatline, power-line ratio, SNR -> ok/warning/bad (shared)
  dataset.py                 file reader -> Recording                                      (offline)
  labeling.py                per-sample labels -> one label per window + keep mask         (offline, calibration)
  evaluation.py              splits, metrics, robustness, calibration effort               (offline)
  classifier.py              shrinkage LDA with calibration adaptation, numpy only         (Group 3)
  decoders.py                decoder factory: LDA or any scikit-learn classifier
tests/                       end-to-end and unit tests on fake data
```

The calibration (`calibration_collection.py`, `calibration_adaptation.py`) and online modules (`realtime_loop.py`,
`monitoring.py`, `recorder.py`, `controller.py`, `force_control.py`, `hand_io.py`) run on the device and do not
belong here; they import `src.pipeline`, `src.classifier` and the profiles written by `train.py`.

## Run

```bash
dvc repro                                  # features + train, outputs in runs/default/
dvc metrics show                           # accuracy, macro-F1, per user, per condition, calibration curve
dvc plots show                             # accuracy vs seconds of training data per class
dvc exp run -S decoder.name=svm            # another decoder, compare with: dvc exp show
dvc exp run -S 'features.names=[MAV,WL]'   # another feature set (rebuilds the features)
```

or without DVC: `python scripts/build_features.py` then `python scripts/train.py`.

## Data

Each recording is one continuous `.npz` file in `data/raw/<name>/`:

| key             | shape  | content                                               |
|-----------------|--------|-------------------------------------------------------|
| `emg`           | [C, T] | raw sEMG, float32, µV                                 |
| `fs`            | scalar | sampling rate (Hz), must match `hardware.fs`          |
| `channel_names` | [C]    | electrode names                                       |
| `labels`        | [T]    | class ID per sample (cue), 0 = rest                   |
| `label_names`   | [K]    | `label_names[class ID]`                               |
| `trial`         | [T]    | trial (repetition) ID per sample                      |
| `timestamps`    | [T]    | optional, seconds (default: sample index / fs)        |
| `meta_<key>`    | scalar | int, for every key in `dataset.meta_keys`             |

Add new data with `dvc add data/raw/<name> && git add data/raw/<name>.dvc && dvc push`. Another file format only
needs a reader in `READERS` of `src/dataset.py`.

### NinaPro DB2

[NinaPro DB2](https://ninapro.hevs.ch/instructions/DB2.html), converted by `scripts/fetch_ninapro.py`:

- `emg`: 12 Delsys electrodes at 2 kHz, converted from V to µV.
- `labels`: `restimulus`, 0 = rest, 1-17 exercise 1, 18-40 exercise 2 (41-49 exercise 3 with `--exercises 1-3`).
- One file per subject and exercise (`data/raw/ninapro_db2/s<subject>/e<exercise>.npz`); `trial` is the repetition.
- Default: exercise 1, one model per subject, test on repetitions 2 and 5 (the standard NinaPro split).

## Processing

- **Same code everywhere.** `build_features.py` streams every recording through `Pipeline.process` in
  `dataset.chunk_seconds` chunks, exactly like the online loop. The filter is causal (`filtering.mode: zero_phase`
  is for analysis only).
- **Windows** cover `[kS, kS + L)` and are labeled by `labeling.py` (`last` sample by default, the causal choice).
  Windows below `labeling.purity`, spanning two trials, inside the filter transient or with a dead channel are
  dropped.
- **σ_rest** per user is the std of the filtered rest samples (online it comes from calibration). WAMP uses it
  as a relative threshold and signal quality uses it for SNR.
- **Splits** use trials or sessions (`evaluation.split`), never random windows, because windows overlap.
  Normalization and decoder are fitted on the training split only.
- **Config hash**: everything that changes the features (`hardware`, `filtering`, `windowing`, `features`,
  `normalization`) is hashed. It is stored with the features, metrics and profiles, and `Pipeline.load_profile`
  rejects a profile made with another config.

## Decoders

`decoder.name` is `lda` (our numpy LDA in `src/classifier.py`, with `adapt` for calibration), an alias of
`src/decoders.py` (`sklearn_lda`, `qda`, `logreg`, `svm`, `linear_svm`, `knn`, `random_forest`, `extra_trees`,
`gradient_boosting`, `mlp`) or any dotted class path. `decoder.params` are passed to the constructor:

```yaml
decoder:
  name: sklearn.ensemble.ExtraTreesClassifier
  params: {n_estimators: 300, min_samples_leaf: 2}
```

The only interface needed is `fit(X, y)`, `predict_proba(X)` and `classes_`. Classifiers without `predict_proba`
are wrapped in `CalibratedClassifierCV`.

## Adding a feature

Write a function `windows [N, C, L] -> [N, C, k]` in `src/feature_extraction.py` and register it in `REGISTRY`
with the names of its `k` values. It can then be listed in `features.names`; its parameters go in `FeaturesConfig`.

## Outputs

`runs/default/`:

- `metrics.json`: `test` (accuracy, macro-F1, recall per class, decision switches per second while a gesture is
  held), `per_user`, `conditions` (accuracy and drop per value of `evaluation.conditions`), `calibration`
  (accuracy vs seconds of training data per class), number of bad channels, config hash.
- `confusion.png`, `calibration.csv`.
- `profiles/<user>.joblib`: calibration profile with σ_rest, normalization statistics, the fitted decoder, feature
  and label names, config hash and validation accuracy. Load it with `Pipeline(cfg, profile=...)`.
