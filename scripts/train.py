"""Evaluate the decoder with the configured regime and split, then fit one calibration profile per user (per_user:
their own decoder; pooled / loso: the decoder of everyone with their own normalization). Writes metrics.json,
confusion.png, calibration.csv and profiles/ to evaluation.out_dir."""

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from matplotlib.figure import Figure

from src import evaluation
from src.config import load_config
from src.pipeline import Pipeline


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


def feature_latency_us(cfg, info, sigma_rest, repeats=200):
    """Median time of the pipeline for one decision step (filter, window, features) on a stream, µs."""
    pipe = Pipeline(cfg, info["channel_names"])
    pipe.sigma_rest = sigma_rest
    rng = np.random.default_rng(0)
    pipe.process(rng.standard_normal((len(info["channel_names"]), info["window_samples"])).astype(np.float32))
    step = cfg.step_samples
    chunks = rng.standard_normal((repeats, len(info["channel_names"]), step)).astype(np.float32)
    times = []
    for chunk in chunks:
        start = time.perf_counter()
        pipe.process(chunk)
        times.append(time.perf_counter() - start)
    return float(np.median(times) * 1e6)


def main(params):
    cfg = load_config(params)
    data = Path(cfg.dataset.out_dir)
    X, y, columns, info = evaluation.load_features(cfg)
    names, feature_names, label_names = info["columns"], info["feature_names"], info["label_names"]
    out = Path(cfg.evaluation.out_dir)
    (out / "profiles").mkdir(parents=True, exist_ok=True)

    evaluated, y_true, y_pred = evaluation.evaluate(X, y, columns, info, cfg)
    test, per_user = evaluated["test"], evaluated["per_user"]
    quality = json.loads((data / "quality.json").read_text())
    results = {
        "config_hash": cfg.hash,
        "decoder": cfg.decoder.name,
        "regime": cfg.evaluation.regime,
        "normalize": cfg.evaluation.normalize,
        "split": cfg.evaluation.split,
        **evaluated,
        "bad_channels": sum(s == "bad" for q in quality.values() for s in q["status"].values()),
    }
    sigma_rest = json.loads((data / "sigma_rest.json").read_text())
    key = cfg.dataset.user_key
    models = evaluation.final_models(X, y, columns, names, feature_names, cfg, cfg.hash)
    normalizer, decoder = next(iter(models.values()))
    results["latency_us"] = {
        "features": feature_latency_us(cfg, info, np.asarray(next(iter(sigma_rest.values())), dtype=np.float32)),
        "decoder": evaluation.decoder_latency_us(decoder, normalizer, X),
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    with open(out / "calibration.csv", "w", newline="") as f:
        rows = [(s, c["accuracy"], c["balanced_accuracy"]) for s, c in evaluated["calibration"].items()]
        csv.writer(f).writerows([("seconds_per_class", "accuracy", "balanced_accuracy"), *rows])
    classes = np.unique(np.concatenate([y_true, y_pred]))
    ticks = [label_names[c] for c in classes] if label_names else [str(c) for c in classes]
    plot_confusion(evaluation.confusion(y_true, y_pred, classes), ticks).savefig(out / "confusion.png", dpi=120)
    print(
        f"{cfg.evaluation.regime}: accuracy {test['accuracy']:.3f} | balanced {test['balanced_accuracy']:.3f} | "
        f"macro-F1 {test['macro_f1']:.3f} | {len(per_user)} users | latency features "
        f"{results['latency_us']['features']:.0f} µs, decoder {results['latency_us']['decoder']:.0f} µs"
    )

    # Final profiles, one per user: fitted on the training split (holdout) or on all windows (leave_one_group_out)
    for user, (normalizer, decoder) in models.items():
        profile = {
            "config_hash": cfg.hash,
            "user": int(user),
            "sigma_rest": np.asarray(sigma_rest[str(user)], dtype=np.float32),
            "normalization": normalizer.state_dict(),
            "decoder_name": cfg.decoder.name,
            "decoder": decoder,
            "feature_names": feature_names,
            "label_names": label_names,
            "regime": cfg.evaluation.regime,
            "validation_accuracy": per_user.get(str(user), {}).get("accuracy"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        joblib.dump(profile, out / "profiles" / f"{key or 'all'}{user}.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml", help="e.g. configs/ninapro_db2.yaml")
    main(parser.parse_args().params)
