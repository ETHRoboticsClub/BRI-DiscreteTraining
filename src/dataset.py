"""Parse recording files into Recording containers. The only module that knows file formats."""

import json
from pathlib import Path

import numpy as np

from src.labeling import repetitions
from src.pipeline import Recording

NO_LABEL = -1  # a sample without a label (no cue, or a cue outside the classes); its windows are dropped


def read_npz(path, meta_keys, ids=None, label_level=None):
    """One .npz per recording, see the README for the keys."""
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    yield Recording(
        signal=d["emg"].astype(np.float32),
        fs=float(d["fs"]),
        channel_names=[str(c) for c in d["channel_names"]],
        timestamps=d.get("timestamps"),
        labels=d["labels"].astype(np.int64) if "labels" in d else None,
        trial=d["trial"].astype(np.int64) if "trial" in d else None,
        meta={k: int(d[f"meta_{k}"]) for k in meta_keys},
        label_names=[str(n) for n in d["label_names"]] if "label_names" in d else None,
    )


def femg_ids(paths, root):
    """Stable integer ids of the names in a femg tree (<root>/<dataset>/<subject>/<session>/meta.json): the position of
    each dataset, subject (dataset/subject) and session name among the sorted names found."""
    names = [femg_names(p, root) for p in paths]
    return {key: {name: i for i, name in enumerate(sorted({n[key] for n in names}))} for key in names[0]}


def femg_names(path, root):
    """{dataset, subject, session} of a session's meta.json path, the names femg_ids numbers."""
    dataset, subject, session = Path(path).parent.relative_to(root).parts
    return {"dataset": dataset, "subject": f"{dataset}/{subject}", "session": session}


def femg_labels(folder, meta, label_level):
    """Per label frame: shared gesture ids (gesture.npy), their families, or the dataset's own cues (stimulus.npy)
    -> (labels, label_names); NO_LABEL where there is none."""
    if label_level == "stimulus":
        stimulus = np.load(folder / "stimulus.npy").astype(np.int64)
        return np.where(stimulus >= 0, stimulus, NO_LABEL), None
    if not (folder / "gesture.npy").exists():
        raise FileNotFoundError(
            f"{folder}: no gesture.npy (built before the shared gestures): rebuild, or use "
            "dataset.label_level: stimulus for a single dataset"
        )
    gesture = np.load(folder / "gesture.npy").astype(np.int64)
    names = meta["gesture_names"]
    if label_level == "gesture":
        return np.where(gesture >= 0, gesture, NO_LABEL), names
    families = list(dict.fromkeys(meta["gesture_families"]))  # rest first, then in vocabulary order
    family_of = np.array([families.index(f) for f in meta["gesture_families"]])
    return np.where(gesture >= 0, family_of[gesture.clip(min=0)], NO_LABEL), families


def read_femg(path, meta_keys, ids, label_level="gesture"):
    """A processed session of BRI-ForearmEMGWorldModelDataset (virtual.npy at fs, labels at label_fs), one Recording
    per segment (continuous recording): the filter restarts and no window spans two. trial: the repetition of the
    last movement onset (each gesture's runs numbered in order, the rest after a movement in its repetition)."""
    folder = Path(path).parent
    meta = json.loads(Path(path).read_text())
    emg = np.load(folder / "virtual.npy", mmap_mode="r")  # [T, 8], normalized (1 ~ strong contraction)
    frames, label_names = femg_labels(folder, meta, label_level)
    r = meta["fs"] // meta["label_fs"]
    labels = np.full(len(emg), NO_LABEL, dtype=np.int64)
    n = min(len(frames) * r, len(emg))
    labels[:n] = np.repeat(frames, r)[:n]
    trial = repetitions(labels)
    names = {"dataset": meta["dataset"], "subject": meta["subject"], "session": meta["session"]}
    for start, end in meta["segments"]:
        a, b = start * r, min(end * r, len(emg))
        if b - a <= 0:
            continue
        yield Recording(
            signal=np.ascontiguousarray(emg[a:b].T, dtype=np.float32),
            fs=float(meta["fs"]),
            channel_names=[f"virtual_{i + 1}" for i in range(emg.shape[1])],
            labels=labels[a:b],
            trial=trial[a:b],
            meta={k: ids[k][names[k]] for k in meta_keys},
            label_names=label_names,
            names={k: names[k] for k in meta_keys},
        )


# name -> (reader, file pattern under raw_dir, indexer of the paths -> integer ids or None)
READERS = {
    "npz": (read_npz, "*.npz", None),
    "femg": (read_femg, "*/*/*/meta.json", femg_ids),
}


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
        if rec.label_names is not None and (rec.labels.min() < NO_LABEL or rec.labels.max() >= len(rec.label_names)):
            errors.append(f"label IDs outside the label map of {len(rec.label_names)} classes")
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))


def matches(rec, where):
    """where = {key: [values]}: integer ids, or names for the keys the reader names (femg: dataset, subject...)."""
    return all(
        rec.meta[k] in values or rec.names.get(k) in values for k, values in (where or {}).items() if k in rec.meta
    )


def load_recordings(cfg):
    """Yield the validated recordings of cfg.dataset.raw_dir that match cfg.dataset.where."""
    reader, pattern, indexer = READERS[cfg.dataset.reader]
    root = Path(cfg.dataset.raw_dir)
    paths = sorted(root.rglob(pattern) if indexer is None else root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no {pattern} recordings in {cfg.dataset.raw_dir}")
    ids = indexer(paths, root) if indexer else None
    if indexer:  # skip the sessions excluded by name before reading them (ids stay those of the whole tree)
        where = {k: v for k, v in (cfg.dataset.where or {}).items() if k in ("dataset", "subject", "session")}
        named = [(p, femg_names(p, root)) for p in paths]
        paths = [p for p, n in named if all(n[k] in v or ids[k][n[k]] in v for k, v in where.items())]
    for path in paths:
        for rec in reader(path, cfg.dataset.meta_keys, ids, cfg.dataset.label_level):
            if not matches(rec, cfg.dataset.where):
                continue
            validate(rec, cfg.hardware, path)
            yield rec


def select(columns, names, where):
    """Mask of the rows of columns [N, K] (named names) whose values match where = {name: [values]}."""
    mask = np.ones(len(columns), dtype=bool)
    for key, values in (where or {}).items():
        mask &= np.isin(columns[:, names.index(key)], values)
    return mask
