# HRI — discrete training

Train gesture classifiers on forearm sEMG: filtering → windowing → hand-crafted features → normalization →
decoder (LDA or any scikit-learn classifier). It holds the shared Group 2 modules and the Group 3 classifier, so
the offline features match the calibration and online paths exactly.

Continuous decoding (joint angles, TRR) lives in `HRI-ContinuousTraining`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/fetch_ninapro.py --db 5 --subjects 1-10 --exercises 1   # or dvc pull, once the data is pushed
```

## Structure

```
params.yaml                  all parameters, one section per module (NinaPro DB5, 200 Hz)
configs/                     other datasets, passed with --params: ninapro_db2.yaml (2 kHz), femg.yaml (below)
dvc.yaml                     pipeline: build_features -> train
scripts/
  fetch_ninapro.py           NinaPro DB2 / DB5 -> data/raw/ninapro_db<n> (recording format below)
  build_features.py          recordings -> labeled feature windows, σ_rest and signal quality per recording
  train.py                   evaluation protocol + one calibration profile per user
src/
  config.py                  params.yaml -> typed, validated config + config hash        (shared)
  filtering.py               causal band-pass + notch, state per channel                   (shared)
  windowing.py               ring buffer -> overlapping windows with end timestamps        (shared)
  feature_extraction.py      feature registry: MAV, RMS, WL, WAMP, AR, MNF, MDF            (shared)
  normalization.py           identity / z-score, fitted on calibration or training data    (shared)
  pipeline.py                Pipeline, Recording, FeatureBatch                             (shared)
  signal_quality.py          saturation, flatline, power-line ratio, SNR -> ok/warning/bad (shared)
  dataset.py                 file readers (npz, femg) -> Recording                         (offline)
  labeling.py                per-sample labels -> one label per window + keep mask         (offline, calibration)
  evaluation.py              splits, metrics, robustness, calibration effort               (offline)
  classifier.py              shrinkage LDA with calibration adaptation, numpy only         (Group 3)
  decoders.py                decoder factory: LDA, the NN decoders or any scikit-learn classifier
  nn.py                      NN decoders (mlp, lda_stack, nn_lda, cnn): torch training, numpy inference
  report.py                  notebook helpers: every regime, saved summaries, plots
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

`evaluation.regime` sets who the decoder is trained on: `per_user` (one decoder per subject, default), `pooled` (one
for everyone, tested on their held-out repetitions) or `loso` (leave one subject out; its calibration curve adapts the
decoder with the first n seconds per class of the new subject). `evaluation.normalize: user | global` sets whether
pooled / loso z-score each subject with their own statistics. `metrics.json` reports balanced accuracy next to
accuracy (rest is ~60 % of the windows) and `latency_us` (features per step, decoder per decision).

Notebooks (`pip install -e ".[notebook]"`): `01`-`05` are the MATLAB port of the baseline, kept as the reference;
`10_db5_lda_baseline` runs it through `src/` for every regime, `11`-`14` the NN decoders, `15_compare` all of them
(run in order: each saves its summary to `runs/notebooks/`).

or without DVC: `python scripts/build_features.py` then `python scripts/train.py`. Both take `--params`
(default `params.yaml`), e.g. `--params configs/ninapro_db2.yaml`.

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

### NinaPro DB5 (default)

[NinaPro DB5](https://ninapro.hevs.ch/instructions/DB5.html), converted by `scripts/fetch_ninapro.py --db 5`:

- `emg`: 16 channels (two Myo armbands) at 200 Hz, raw Myo units (int8 range, not µV).
- `labels`: `restimulus` offset per exercise, 0 = rest, 1-12 exercise 1, 13-29 exercise 2, 30-52 exercise 3.
- Default (`params.yaml`): exercise 1, MAV / RMS / WL, 20 Hz high-pass + 50 Hz notch (no low-pass below Nyquist),
  train on repetitions 1-4, test on 5 and 6 (the split of the notebook baseline in `notebooks/01-05`).

### BRI-ForearmEMGWorldModelDataset (`configs/femg.yaml`)

The processed tree of [the dataset repo](https://github.com/ETHRoboticsClub/BRI-ForearmEMGWorldModelDataset): every
dataset on one virtual band of 8 electrodes at 200 Hz, normalized per channel, labels as shared gestures.

```bash
dvc import git@github.com:ETHRoboticsClub/BRI-ForearmEMGWorldModelDataset data/processed -o data/forearm
python scripts/build_features.py --params configs/femg.yaml && python scripts/train.py --params configs/femg.yaml
```

- Reader `femg` (`src/dataset.py`): `<dataset>/<subject>/<session>/{virtual,gesture,stimulus}.npy` + `meta.json`. The
  labels (at `label_fs`) are repeated to the EMG rate; `-1` (no cue, or a cue outside the shared vocabulary) is never
  a class, its windows are dropped. One `Recording` per segment, so the filter restarts and no window spans two.
- `dataset.label_level`: `gesture` (shared ids, `gesture.npy`), `family` (coarse classes, merges near matches across
  datasets) or `stimulus` (a dataset's own cue ids, for one dataset or sessions built before `gesture.npy`).
- `dataset.where` takes names: `{dataset: [ninapro_db5, grabmyo], subject: [ninapro_db5/s01]}`. The integer ids in
  `columns.npy` are the positions of the sorted names; `info.json` `meta_names` maps them back.
- No repetition ids in the files: `trial` numbers each gesture's runs in order (rest after a movement belongs to its
  repetition). `trial_from_end` (every reader) counts from the last one, so `evaluation.test: {trial_from_end: [1, 2]}`
  holds out the last two repetitions of every dataset, whatever their number.

### NinaPro DB2 (`configs/ninapro_db2.yaml`)

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

`decoder.name` is `lda` (our numpy LDA in `src/classifier.py`, with `adapt` for calibration), one of the NN decoders
below, an alias of `src/decoders.py` (`sklearn_lda`, `qda`, `logreg`, `svm`, `linear_svm`, `knn`, `random_forest`,
`extra_trees`, `gradient_boosting`, `sklearn_mlp`) or any dotted class path. `decoder.params` are passed to the constructor:

```yaml
decoder:
  name: sklearn.ensemble.ExtraTreesClassifier
  params: {n_estimators: 300, min_samples_leaf: 2}
```

The only interface needed is `fit(X, y)`, `predict_proba(X)` and `classes_`. Classifiers without `predict_proba`
are wrapped in `CalibratedClassifierCV`.

### NN decoders

`src/nn.py`, trained with torch (`pip install -e ".[nn]"`), with numpy inference from the exported weights (a profile
loads and runs without torch). Training: AdamW, class-balanced cross-entropy, early stopping on held-out contiguous
2 s blocks of each class. `adapt` (calibration of the `loso` regime) is cheap for all of them.

| name        | input                       | model                                              | `adapt`                    |
|-------------|-----------------------------|----------------------------------------------------|----------------------------|
| `mlp`       | features                    | hidden layers (`hidden: [128, 128]`)               | fine-tunes the output layer |
| `lda_stack` | features                    | MLP on [out-of-fold LDA log-posteriors, features]  | fine-tunes the MLP's output layer (the LDA stays) |
| `nn_lda`    | features                    | MLP encoder (`hidden: [128, 32]`) -> LDA head      | `LDA.adapt` on the embedding |
| `cnn`       | `features.names: [RAW]`     | Conv1d x 3 -> mean over time -> linear (`channels` required) | fine-tunes the output layer |

`device: cpu | cuda | mps | auto` sets where training runs (`cnn` defaults to `auto`: about 25x faster on an Apple
GPU than on its CPU). Notebooks `11`-`14` run each decoder in every regime, `15_compare` puts them side by side.

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
