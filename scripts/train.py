"""Evaluate the decoder with the configured split, then fit one calibration profile per user on their
training data. Writes metrics.json, confusion.png, calibration.csv and profiles/ to evaluation.out_dir."""

import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from matplotlib.figure import Figure

from src import evaluation
from src.config import load_config


def plot_confusion(cm, names):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig = Figure(figsize=(0.35 * len(names) + 3, 0.35 * len(names) + 2.5))
    ax = fig.add_subplot()
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xlabel="predicted", ylabel="true")
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    fig.colorbar(im, label="recall")
    return fig


def main():
    cfg = load_config()
    data = Path(cfg.dataset.out_dir)
    info = json.loads((data / "info.json").read_text())
    if info["config_hash"] != cfg.hash:
        raise ValueError(f"features built with config {info['config_hash']}, current is {cfg.hash}: run dvc repro")
    X, y = np.load(data / "features.npy"), np.load(data / "labels.npy")
    columns, names, feature_names = np.load(data / "columns.npy"), info["columns"], info["feature_names"]
    label_names = info["label_names"]
    out = Path(cfg.evaluation.out_dir)
    (out / "profiles").mkdir(parents=True, exist_ok=True)

    y_true, y_pred, cols, per_user, curve = evaluation.run(X, y, columns, names, feature_names, cfg, info)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    segment = np.unique(cols[:, [names.index("recording"), names.index("trial")]], axis=0, return_inverse=True)[1]
    test = evaluation.classification_metrics(
        y_true, y_pred, classes, label_names, segment.ravel(), info["window_rate"], cfg.dataset.rest_class
    )
    test["mean_user_accuracy"] = float(np.mean(list(per_user.values())))
    quality = json.loads((data / "quality.json").read_text())
    results = {
        "config_hash": cfg.hash,
        "decoder": cfg.decoder.name,
        "split": cfg.evaluation.split,
        "test": test,
        "per_user": per_user,
        "conditions": evaluation.per_condition(y_true, y_pred, cols, names, cfg.evaluation.conditions),
        "calibration": {str(s): a for s, a in curve.items()},
        "bad_channels": sum(s == "bad" for q in quality.values() for s in q["status"].values()),
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    with open(out / "calibration.csv", "w", newline="") as f:
        csv.writer(f).writerows([("seconds_per_class", "accuracy"), *curve.items()])
    ticks = [label_names[c] for c in classes] if label_names else [str(c) for c in classes]
    plot_confusion(evaluation.confusion(y_true, y_pred, classes), ticks).savefig(out / "confusion.png", dpi=120)
    print(f"test accuracy {test['accuracy']:.3f} | macro-F1 {test['macro_f1']:.3f} | {len(per_user)} users")

    # Final profiles, one per user: fitted on the training split (holdout) or on all windows (leave_one_group_out)
    sigma_rest = json.loads((data / "sigma_rest.json").read_text())
    key = cfg.dataset.user_key
    users = columns[:, names.index(key)] if key else np.zeros(len(y), dtype=np.int64)
    for user in np.unique(users):
        u = users == user
        if cfg.evaluation.split == "holdout":
            train = evaluation.folds(columns[u], names, cfg.evaluation)[0][1]
        else:
            train = np.ones(u.sum(), dtype=bool)
        normalizer, decoder = evaluation.fit(X[u][train], y[u][train], feature_names, cfg, cfg.hash)
        profile = {
            "config_hash": cfg.hash,
            "user": int(user),
            "sigma_rest": np.asarray(sigma_rest[str(user)], dtype=np.float32),
            "normalization": normalizer.state_dict(),
            "decoder_name": cfg.decoder.name,
            "decoder": decoder,
            "feature_names": feature_names,
            "label_names": label_names,
            "validation_accuracy": per_user.get(str(user)),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        joblib.dump(profile, out / "profiles" / f"{key or 'all'}{user}.joblib")


if __name__ == "__main__":
    main()
