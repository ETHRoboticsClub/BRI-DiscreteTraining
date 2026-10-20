"""One fixed protocol to compare design choices: splits by trial or session (never random windows, they overlap),
classification metrics, robustness per condition and calibration effort."""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from src.dataset import select
from src.decoders import make_decoder
from src.normalization import Normalizer


def folds(columns, names, cfg):
    """[(name, train_mask, test_mask)] of the configured split."""
    if cfg.split == "holdout":
        test = select(columns, names, cfg.test) if cfg.test else np.zeros(len(columns), dtype=bool)
        return [("holdout", ~test, test)]
    groups = columns[:, names.index(cfg.group_key)]
    return [(f"{cfg.group_key}{g}", groups != g, groups == g) for g in np.unique(groups)]


def fit(X, y, feature_names, cfg, config_hash=None):
    """Normalization and decoder fitted on the same training data."""
    normalizer = Normalizer(cfg.normalization).fit(X, feature_names, config_hash)
    decoder = make_decoder(cfg.decoder, seed=cfg.evaluation.seed).fit(normalizer.transform(X), y)
    return normalizer, decoder


def predict(normalizer, decoder, X):
    proba = decoder.predict_proba(normalizer.transform(X))
    return np.asarray(decoder.classes_)[proba.argmax(axis=1)], proba


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


def first_seconds(y, seconds, window_rate):
    """Mask of the first `seconds` of windows of every class (calibration subsets)."""
    n = round(seconds * window_rate)
    mask = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        mask[np.flatnonzero(y == c)[:n]] = True
    return mask


def run(X, y, columns, names, feature_names, cfg, info):
    """Evaluate every user and fold; returns (pooled test predictions, per-user accuracy, calibration curve)."""
    users = columns[:, names.index(cfg.dataset.user_key)] if cfg.dataset.user_key else np.zeros(len(y), int)
    rows, per_user, calibration = [], {}, {s: [] for s in cfg.evaluation.calibration_seconds}
    for user in np.unique(users):
        u = users == user
        Xu, yu, cu = X[u], y[u], columns[u]
        correct = total = 0
        for _, train, test in folds(cu, names, cfg.evaluation):
            if not test.any() or len(np.unique(yu[train])) < 2:
                continue
            normalizer, decoder = fit(Xu[train], yu[train], feature_names, cfg, info["config_hash"])
            pred, _ = predict(normalizer, decoder, Xu[test])
            rows.append((yu[test], pred, cu[test]))
            correct, total = correct + (pred == yu[test]).sum(), total + test.sum()
            for seconds in calibration:
                sub = first_seconds(yu[train], seconds, info["window_rate"])
                if len(np.unique(yu[train][sub])) < 2:
                    continue
                n, d = fit(Xu[train][sub], yu[train][sub], feature_names, cfg)
                calibration[seconds].append(((predict(n, d, Xu[test])[0] == yu[test]).sum(), test.sum()))
        if total:
            per_user[str(user)] = float(correct / total)
    if not rows:
        raise ValueError("no fold with test windows: check evaluation.test / evaluation.group_key")
    y_true, y_pred, cols = (np.concatenate(r) for r in zip(*rows, strict=True))
    curve = {s: float(sum(c for c, _ in v) / sum(t for _, t in v)) for s, v in calibration.items() if v}
    return y_true, y_pred, cols, per_user, curve


def confusion(y_true, y_pred, classes):
    return confusion_matrix(y_true, y_pred, labels=classes)
