"""Parse recording files into Recording containers. The only module that knows file formats."""

from pathlib import Path

import numpy as np

from src.pipeline import Recording


def read_npz(path, meta_keys):
    """One .npz per recording, see the README for the keys."""
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    return Recording(
        signal=d["emg"].astype(np.float32),
        fs=float(d["fs"]),
        channel_names=[str(c) for c in d["channel_names"]],
        timestamps=d.get("timestamps"),
        labels=d["labels"].astype(np.int64) if "labels" in d else None,
        trial=d["trial"].astype(np.int64) if "trial" in d else None,
        meta={k: int(d[f"meta_{k}"]) for k in meta_keys},
        label_names=[str(n) for n in d["label_names"]] if "label_names" in d else None,
    )


READERS = {"npz": (read_npz, "*.npz")}


def validate(rec, hw, path=""):
    errors = []
    if rec.fs != hw.fs:
        errors.append(f"recorded at {rec.fs} Hz, config expects {hw.fs} Hz")
    if rec.signal.shape[0] != hw.n_channels:
        errors.append(f"{rec.signal.shape[0]} channels, config expects {hw.n_channels}")
    if not np.isfinite(rec.signal).all():
        errors.append("signal contains NaN or inf")
    if rec.labels is not None:
        if len(rec.labels) != rec.signal.shape[1]:
            errors.append(f"{len(rec.labels)} labels for {rec.signal.shape[1]} samples")
        if rec.label_names is not None and (rec.labels.min() < 0 or rec.labels.max() >= len(rec.label_names)):
            errors.append(f"label IDs outside the label map of {len(rec.label_names)} classes")
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))


def matches(meta, where):
    return all(meta[k] in values for k, values in (where or {}).items() if k in meta)


def load_recordings(cfg):
    """Yield the validated recordings of cfg.dataset.raw_dir that match cfg.dataset.where."""
    reader, pattern = READERS[cfg.dataset.reader]
    paths = sorted(Path(cfg.dataset.raw_dir).rglob(pattern))
    if not paths:
        raise FileNotFoundError(f"no {pattern} recordings in {cfg.dataset.raw_dir}")
    for path in paths:
        rec = reader(path, cfg.dataset.meta_keys)
        if not matches(rec.meta, cfg.dataset.where):
            continue
        validate(rec, cfg.hardware, path)
        yield rec


def select(columns, names, where):
    """Mask of the rows of columns [N, K] (named names) whose values match where = {name: [values]}."""
    mask = np.ones(len(columns), dtype=bool)
    for key, values in (where or {}).items():
        mask &= np.isin(columns[:, names.index(key)], values)
    return mask
