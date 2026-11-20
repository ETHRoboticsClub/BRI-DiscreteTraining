"""Recordings -> labeled feature windows in dataset.out_dir, through the same pipeline as the online path.

σ_rest (for WAMP and SNR) is estimated per user from the rest samples of their recordings, as calibration would.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.config import load_config
from src.dataset import load_recordings
from src.labeling import from_end, label_windows
from src.pipeline import Pipeline
from src.signal_quality import assess


def user_of(rec, cfg):
    return rec.meta[cfg.dataset.user_key] if cfg.dataset.user_key else 0


def estimate_sigma_rest(cfg):
    """{user: σ_rest [C]} pooled over all rest samples of the user's recordings."""
    sums = defaultdict(lambda: np.zeros(3, dtype=object))
    for rec in load_recordings(cfg):
        pipe = Pipeline(cfg, rec.channel_names)
        filtered = pipe.filter.process(rec.signal).astype(np.float64)
        rest = rec.labels == cfg.dataset.rest_class
        rest[: pipe.filter.transient] = False
        sums[user_of(rec, cfg)] += np.array(
            [rest.sum(), filtered[:, rest].sum(1), (filtered[:, rest] ** 2).sum(1)], dtype=object
        )
    sigma = {}
    for user, (n, s, ss) in sums.items():
        if not n:
            raise ValueError(f"user {user}: no rest samples (class {cfg.dataset.rest_class}) to estimate σ_rest")
        sigma[user] = np.sqrt(np.maximum(ss / n - (s / n) ** 2, 0)).astype(np.float32)
    return sigma


def build(cfg):
    out = Path(cfg.dataset.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sigma_rest = estimate_sigma_rest(cfg)
    columns_names = [*cfg.dataset.meta_keys, "trial", "recording", "trial_from_end"]

    X, y, columns, end_times, quality, dropped, meta_names = [], [], [], [], {}, 0, {}
    for i, rec in enumerate(load_recordings(cfg)):
        pipe = Pipeline(cfg, rec.channel_names, expose=("filtered",))
        pipe.sigma_rest = sigma_rest[user_of(rec, cfg)]
        batch = pipe.process_recording(rec, cfg.dataset.chunk_seconds)
        labels, keep, trial = label_windows(
            rec.labels, batch.end_samples, pipe.windower.length, rec.fs, cfg.labeling, rec.trial
        )
        keep &= ~batch.invalid & (batch.end_samples >= pipe.filter.transient) & (labels >= 0)  # -1: no label
        if cfg.dataset.classes is not None:
            keep &= np.isin(labels, cfg.dataset.classes)
        dropped += int((~keep).sum())

        name = "_".join(f"{k}{v}" for k, v in rec.meta.items()) or f"recording{i}"
        name = name if name not in quality else f"{name}_recording{i}"  # several segments of one session
        metrics, status = assess(
            rec.signal,
            batch.intermediate["filtered"],
            rec.fs,
            cfg.signal_quality,
            cfg.hardware,
            cfg.filtering,
            pipe.sigma_rest,
            active=(rec.labels != cfg.dataset.rest_class) & (rec.labels >= 0),
        )
        quality[name] = {
            "status": dict(zip(rec.channel_names, status.tolist(), strict=True)),
            **{k: v.round(4).tolist() for k, v in metrics.items()},
        }

        trial = trial if trial is not None else np.zeros(len(labels), dtype=np.int64)
        meta = np.array([rec.meta[k] for k in cfg.dataset.meta_keys], dtype=np.int64)
        X.append(batch.features[keep])
        y.append(labels[keep])
        columns.append(np.column_stack([np.tile(meta, (keep.sum(), 1)), trial[keep], np.full(keep.sum(), i)]))
        end_times.append(batch.end_times[keep])
        label_names = rec.label_names
        for key, value in rec.names.items():
            meta_names.setdefault(key, {})[str(rec.meta[key])] = value

    np.save(out / "features.npy", np.concatenate(X))
    np.save(out / "labels.npy", np.concatenate(y))
    columns = np.concatenate(columns).astype(np.int64)
    n_meta = len(cfg.dataset.meta_keys)  # columns: meta keys, trial, recording
    sessions = np.unique(columns[:, :n_meta], axis=0, return_inverse=True)[1].ravel()
    trial_from_end = from_end(columns[:, n_meta], np.concatenate(y), sessions, cfg.dataset.rest_class)
    np.save(out / "columns.npy", np.column_stack([columns, trial_from_end]))
    np.save(out / "end_times.npy", np.concatenate(end_times))
    info = {
        "config_hash": cfg.hash,
        "feature_names": pipe.feature_names,
        "columns": columns_names,
        "label_names": label_names,
        "meta_names": meta_names,  # {key: {id: name}} of the readers that name their ids (femg)
        "channel_names": pipe.channel_names,
        "fs": cfg.hardware.fs,
        "window_samples": pipe.windower.length,
        "window_rate": pipe.windower.rate,
        "dropped_windows": dropped,
    }
    (out / "info.json").write_text(json.dumps(info, indent=2))
    (out / "sigma_rest.json").write_text(json.dumps({str(u): s.tolist() for u, s in sigma_rest.items()}, indent=2))
    (out / "quality.json").write_text(json.dumps(quality, indent=2))
    n = sum(len(v) for v in y)
    print(f"{n} windows x {len(pipe.feature_names)} features ({dropped} dropped) -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml", help="e.g. configs/ninapro_db2.yaml")
    build(load_config(parser.parse_args().params))
