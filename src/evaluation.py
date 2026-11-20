"""One fixed protocol to compare design choices: splits by trial or session (never random windows, they overlap),
training regimes (per user, pooled, leave one subject out), classification metrics, robustness per condition and
calibration effort."""

import copy
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from src.config import NormalizationConfig
from src.dataset import select
from src.decoders import make_decoder
from src.normalization import Normalizer


def load_features(cfg):
    """The windows of build_features.py -> X [N, d], y [N], columns [N, K], info (names, rates, label names...)."""
    data = Path(cfg.dataset.out_dir)
    info = json.loads((data / "info.json").read_text())
    if info["config_hash"] != cfg.hash:
        raise ValueError(f"features built with config {info['config_hash']}, current is {cfg.hash}: run dvc repro")
    return np.load(data / "features.npy"), np.load(data / "labels.npy"), np.load(data / "columns.npy"), info


def folds(columns, names, cfg):
    """[(name, train_mask, test_mask)] of the configured split."""
    if cfg.split == "holdout":
        test = select(columns, names, cfg.test) if cfg.test else np.zeros(len(columns), dtype=bool)
        return [("holdout", ~test, test)]
    groups = columns[:, names.index(cfg.group_key)]
    return [(f"{cfg.group_key}{g}", groups != g, groups == g) for g in np.unique(groups)]


def user_column(columns, names, cfg):
    key = cfg.dataset.user_key
    return columns[:, names.index(key)] if key else np.zeros(len(columns), dtype=np.int64)


def experiments(columns, names, cfg):
    """[(name, train, test, user)] masks over all rows. user is the test subject (None when pooled).

    per_user: the split inside every user, one decoder each. pooled: the split over everyone, one decoder.
    loso: every user's split test windows against a decoder trained on all the other users."""
    ev, users = cfg.evaluation, user_column(columns, names, cfg)
    if ev.regime == "pooled":
        return [(name, train, test, None) for name, train, test in folds(columns, names, ev)]
    out = []
    for user in np.unique(users):
        u = users == user
        for name, tr, te in folds(columns[u], names, ev):
            train, test = np.zeros(len(u), dtype=bool), np.zeros(len(u), dtype=bool)
            train[u], test[u] = tr, te
            out.append((f"{cfg.dataset.user_key}{user}_{name}", ~u if ev.regime == "loso" else train, test, user))
    return out


def fit(X, y, feature_names, cfg, config_hash=None, normalize=True):
    """Normalization and decoder fitted on the same training data (normalize=False: X is normalized already)."""
    norm_cfg = cfg.normalization if normalize else NormalizationConfig(method="identity")
    normalizer = Normalizer(norm_cfg).fit(X, feature_names, config_hash)
    decoder = make_decoder(cfg.decoder, seed=cfg.evaluation.seed).fit(normalizer.transform(X), y)
    return normalizer, decoder


def predict(normalizer, decoder, X):
    proba = decoder.predict_proba(normalizer.transform(X))
    return np.asarray(decoder.classes_)[proba.argmax(axis=1)], proba


def normalize_per_user(X, users, reference, feature_names, cfg, config_hash=None):
    """X with every user z-scored by the statistics of their own reference windows -> (X, {user: Normalizer})."""
    out, normalizers = np.empty(X.shape, dtype=np.float32), {}
    for user in np.unique(users):
        u = users == user
        ref = u & reference if (u & reference).any() else u
        normalizers[user] = Normalizer(cfg.normalization).fit(X[ref], feature_names, config_hash)
        out[u] = normalizers[user].transform(X[u])
    return out, normalizers


def switches_per_second(y_true, y_pred, segment, window_rate, rest_class=0):
    """Changes of the decision per second while one gesture is held (time-ordered windows)."""
    held = (y_true[1:] == y_true[:-1]) & (segment[1:] == segment[:-1]) & (y_true[1:] != rest_class)
    seconds = held.sum() / window_rate
    return float((y_pred[1:] != y_pred[:-1])[held].sum() / seconds) if seconds else 0.0


def classification_metrics(y_true, y_pred, classes, label_names=None, segment=None, window_rate=None, rest_class=0):
    name = (lambda c: label_names[c]) if label_names else str
    recall = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    out = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "recall": {name(c): float(r) for c, r in zip(classes, recall, strict=True)},
    }
    if segment is not None and window_rate:
        out["switches_per_s"] = switches_per_second(y_true, y_pred, segment, window_rate, rest_class)
    return out


def per_condition(y_true, y_pred, columns, names, keys):
    """Accuracy per value of each condition column and its drop relative to the overall accuracy."""
    overall = (y_true == y_pred).mean()
    out = {}
    for key in keys:
        col = columns[:, names.index(key)]
        out[key] = {}
        for v in np.unique(col):
            acc = float((y_true[col == v] == y_pred[col == v]).mean())
            out[key][str(v)] = {"accuracy": acc, "delta": float(overall - acc)}
    return out


def per_user(y_true, y_pred, users):
    return {
        str(u): {
            "accuracy": float((y_true[users == u] == y_pred[users == u]).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y_true[users == u], y_pred[users == u])),
        }
        for u in np.unique(users)
    }


def first_seconds(y, seconds, window_rate):
    """Mask of the first `seconds` of windows of every class (calibration subsets)."""
    n = round(seconds * window_rate)
    mask = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        mask[np.flatnonzero(y == c)[:n]] = True
    return mask


def calibrate(decoder, X, y, Xn, train, test, user, users, feature_names, cfg, info):
    """Accuracy on the test windows after n seconds per class of the test user's own training windows, for every n
    of evaluation.calibration_seconds -> {n: (y_true, y_pred)}.

    per_user: a decoder fitted from scratch on them. loso: the decoder of the other users adapted with them
    (decoders with adapt only). pooled: nothing, the user's data is in the training set already."""
    regime, out = cfg.evaluation.regime, {}
    if regime == "pooled" or (regime == "loso" and not hasattr(decoder, "adapt")):
        return out
    own = (users == user) & ~test
    for seconds in cfg.evaluation.calibration_seconds:
        sub = np.flatnonzero(own)[first_seconds(y[own], seconds, info["window_rate"])]
        if len(np.unique(y[sub])) < 2:
            continue
        if regime == "per_user":
            n, d = fit(X[sub], y[sub], feature_names, cfg)
            pred = predict(n, d, X[test])[0]
        else:
            d = copy.deepcopy(decoder).adapt(Xn[sub], y[sub])
            pred = np.asarray(d.classes_)[d.predict_proba(Xn[test]).argmax(axis=1)]
        out[seconds] = (y[test], pred)
    return out


def run(X, y, columns, names, feature_names, cfg, info):
    """Evaluate every experiment of the regime -> (pooled test predictions, their columns, calibration curve)."""
    users = user_column(columns, names, cfg)
    per_user_norm = cfg.evaluation.regime == "per_user" or cfg.evaluation.normalize == "user"
    rows, calibration = [], {s: [] for s in cfg.evaluation.calibration_seconds}
    for _, train, test, user in experiments(columns, names, cfg):
        if not test.any() or len(np.unique(y[train])) < 2:
            continue
        if per_user_norm:  # statistics from every user's own non-test windows (labels unused)
            Xn = normalize_per_user(X, users, ~test, feature_names, cfg, info["config_hash"])[0]
            decoder = fit(Xn[train], y[train], feature_names, cfg, normalize=False)[1]
        else:
            normalizer, decoder = fit(X[train], y[train], feature_names, cfg, info["config_hash"])
            Xn = normalizer.transform(X)
        pred = np.asarray(decoder.classes_)[decoder.predict_proba(Xn[test]).argmax(axis=1)]
        rows.append((y[test], pred, columns[test]))
        for seconds, result in calibrate(decoder, X, y, Xn, train, test, user, users, feature_names, cfg, info).items():
            calibration[seconds].append(result)
    if not rows:
        raise ValueError("no experiment with test windows: check evaluation.test / evaluation.group_key")
    y_true, y_pred, cols = (np.concatenate(r) for r in zip(*rows, strict=True))
    curve = {}
    for seconds, results in calibration.items():
        if results:
            true, pred = (np.concatenate(r) for r in zip(*results, strict=True))
            curve[seconds] = {
                "accuracy": float((true == pred).mean()),
                "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
            }
    return y_true, y_pred, cols, curve


def evaluate(X, y, columns, info, cfg):
    """run() and its metrics -> ({test, per_user, conditions, calibration}, y_true, y_pred)."""
    names, label_names = info["columns"], info["label_names"]
    y_true, y_pred, cols, curve = run(X, y, columns, names, info["feature_names"], cfg, info)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    segment = np.unique(cols[:, [names.index("recording"), names.index("trial")]], axis=0, return_inverse=True)[1]
    test = classification_metrics(
        y_true, y_pred, classes, label_names, segment.ravel(), info["window_rate"], cfg.dataset.rest_class
    )
    users = per_user(y_true, y_pred, user_column(cols, names, cfg))
    for metric in ("accuracy", "balanced_accuracy"):
        test[f"mean_user_{metric}"] = float(np.mean([u[metric] for u in users.values()]))
    results = {
        "test": test,
        "per_user": users,
        "conditions": per_condition(y_true, y_pred, cols, names, cfg.evaluation.conditions),
        "calibration": {str(s): c for s, c in curve.items()},
    }
    return results, y_true, y_pred


def final_models(X, y, columns, names, feature_names, cfg, config_hash):
    """{user: (normalizer, decoder)} for the profiles, fitted on the training split (holdout) or all windows.

    per_user: one decoder per user. pooled / loso: one decoder for everyone, each user with their own normalization
    statistics (or the shared ones with evaluation.normalize: global)."""
    users = user_column(columns, names, cfg)
    if cfg.evaluation.split == "holdout":
        train = np.zeros(len(y), dtype=bool)
        for user in np.unique(users):
            u = users == user
            train[u] = folds(columns[u], names, cfg.evaluation)[0][1]
    else:
        train = np.ones(len(y), dtype=bool)
    if cfg.evaluation.regime == "per_user":
        return {
            user: fit(X[(users == user) & train], y[(users == user) & train], feature_names, cfg, config_hash)
            for user in np.unique(users)
        }
    if cfg.evaluation.normalize == "user":
        Xn, normalizers = normalize_per_user(X, users, train, feature_names, cfg, config_hash)
        decoder = fit(Xn[train], y[train], feature_names, cfg, normalize=False)[1]
        return {user: (normalizers[user], decoder) for user in np.unique(users)}
    normalizer, decoder = fit(X[train], y[train], feature_names, cfg, config_hash)
    return {user: (normalizer, decoder) for user in np.unique(users)}


def decoder_latency_us(decoder, normalizer, X, repeats=200):
    """Median time of one decision (normalization + predict_proba of a single window), µs."""
    rows = X[np.arange(repeats) % len(X)]
    times = []
    for row in rows:
        start = time.perf_counter()
        decoder.predict_proba(normalizer.transform(row[None]))
        times.append(time.perf_counter() - start)
    return float(np.median(times) * 1e6)


def confusion(y_true, y_pred, classes):
    return confusion_matrix(y_true, y_pred, labels=classes)
