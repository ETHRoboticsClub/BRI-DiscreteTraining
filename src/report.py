"""Notebook helpers: evaluate a decoder in every regime, save the summary to runs/notebooks/ and plot it. The
comparison notebook (15) reads the saved summaries, so each decoder notebook runs once."""

import json
import sys
import time
from pathlib import Path

import numpy as np

from src import evaluation
from src.config import load_config

REGIMES = ("per_user", "pooled", "loso")
OUT = Path("runs/notebooks")


def features(overrides=None):
    """Config and windows of the given overrides, built with scripts/build_features.py if missing or outdated."""
    cfg = load_config(overrides=overrides)
    info = Path(cfg.dataset.out_dir) / "info.json"
    if not info.exists() or json.loads(info.read_text())["config_hash"] != cfg.hash:
        sys.path.insert(0, "scripts")
        from build_features import build

        build(cfg)
    return cfg, *evaluation.load_features(cfg)


def run_regimes(overrides, data, regimes=REGIMES):
    """{regime: summary} of the decoder in overrides, on data = (X, y, columns, info) of the same features."""
    X, y, columns, info = data
    out = {}
    for regime in regimes:
        cfg = load_config(overrides={**overrides, "evaluation": {**overrides.get("evaluation", {}), "regime": regime}})
        start = time.perf_counter()
        results, y_true, y_pred = evaluation.evaluate(X, y, columns, info, cfg)
        seconds = time.perf_counter() - start
        per_user = [u["balanced_accuracy"] for u in results["per_user"].values()]
        test = results["test"]
        out[regime] = {
            **{k: test[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "switches_per_s")},
            "worst_user_balanced": min(per_user),
            "best_user_balanced": max(per_user),
            "evaluation_s": seconds,
            "calibration": results["calibration"],
            "recall": test["recall"],
            "y_true": y_true,
            "y_pred": y_pred,
        }
        print(
            f"{regime:8s} balanced {test['balanced_accuracy']:.3f} | macro-F1 {test['macro_f1']:.3f} | {seconds:.0f} s"
        )
    return out


def latency_us(overrides, data):
    """Decoder time per decision of the per-user model of the first user, µs."""
    X, y, columns, info = data
    cfg = load_config(overrides={**overrides, "evaluation": {"regime": "per_user"}})
    users = evaluation.user_column(columns, info["columns"], cfg)
    u = users == users[0]
    normalizer, decoder = evaluation.fit(X[u], y[u], info["feature_names"], cfg)
    return evaluation.decoder_latency_us(decoder, normalizer, X[u])


def save(name, summary, latency):
    OUT.mkdir(parents=True, exist_ok=True)
    keep = {r: {k: v for k, v in s.items() if k not in ("y_true", "y_pred")} for r, s in summary.items()}
    (OUT / f"{name}.json").write_text(json.dumps({"decoder": name, "latency_us": latency, "regimes": keep}, indent=1))


def load_all():
    return {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))}


def table(summaries):
    """DataFrame: one row per decoder and regime."""
    import pandas as pd

    rows = []
    for name, s in summaries.items():
        for regime, r in s["regimes"].items():
            rows.append(
                {
                    "decoder": name,
                    "regime": regime,
                    **{k: r[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "worst_user_balanced")},
                    "switches_per_s": r["switches_per_s"],
                    "latency_us": s["latency_us"],
                    "evaluation_s": r["evaluation_s"],
                }
            )
    return pd.DataFrame(rows).set_index(["decoder", "regime"])


def plot_calibration(ax, summaries, regimes=("per_user", "loso")):
    """Balanced accuracy vs calibration seconds per class, one line per decoder and regime."""
    styles = {"per_user": "--", "loso": "-"}
    for i, (name, s) in enumerate(summaries.items()):
        for regime in regimes:
            curve = s["regimes"].get(regime, {}).get("calibration", {})
            if curve:
                seconds = [float(k) for k in curve]
                values = [c["balanced_accuracy"] for c in curve.values()]
                ax.plot(seconds, values, styles.get(regime, ":"), marker="o", color=f"C{i}", label=f"{name} {regime}")
    ax.set(xlabel="calibration seconds per class", ylabel="balanced accuracy", ylim=(0, 1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_confusion(ax, y_true, y_pred, label_names):
    classes = np.unique(y_true)
    cm = evaluation.confusion(y_true, y_pred, classes)
    im = ax.imshow(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1), vmin=0, vmax=1, cmap="Blues")
    names = [label_names[c] for c in classes]
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xlabel="predicted", ylabel="true")
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticklabels(names)
    return im
