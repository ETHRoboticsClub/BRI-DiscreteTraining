"""Decoder factory: our LDA or any scikit-learn classifier with fit / predict_proba.

params.yaml:
    decoder: {name: svm, params: {C: 10}}
    decoder: {name: sklearn.ensemble.ExtraTreesClassifier, params: {n_estimators: 300}}
"""

import importlib

from src.classifier import LDA

ALIASES = {
    "lda": LDA,
    "sklearn_lda": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
    "qda": "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis",
    "logreg": "sklearn.linear_model.LogisticRegression",
    "svm": "sklearn.svm.SVC",
    "linear_svm": "sklearn.svm.LinearSVC",
    "knn": "sklearn.neighbors.KNeighborsClassifier",
    "random_forest": "sklearn.ensemble.RandomForestClassifier",
    "extra_trees": "sklearn.ensemble.ExtraTreesClassifier",
    "gradient_boosting": "sklearn.ensemble.HistGradientBoostingClassifier",
    "mlp": "sklearn.neural_network.MLPClassifier",
}
DEFAULTS = {  # so that predict_proba exists or the defaults are sensible for EMG features
    "sklearn_lda": {"solver": "lsqr", "shrinkage": "auto"},
    "svm": {"probability": True},
    "logreg": {"max_iter": 1000},
}


def resolve(name):
    target = ALIASES.get(name, name)
    if not isinstance(target, str):
        return target
    module, _, cls = target.rpartition(".")
    if not module:
        raise ValueError(f"unknown decoder '{name}', use one of {sorted(ALIASES)} or a dotted class path")
    return getattr(importlib.import_module(module), cls)


def make_decoder(cfg, seed=None):
    cls = resolve(cfg.name)
    params = {**DEFAULTS.get(cfg.name, {}), **cfg.params}
    if seed is not None and "random_state" in getattr(cls, "_get_param_names", lambda: [])():
        params.setdefault("random_state", seed)
    decoder = cls(**params)
    if not hasattr(decoder, "predict_proba"):  # e.g. LinearSVC: calibrated probabilities
        from sklearn.calibration import CalibratedClassifierCV

        decoder = CalibratedClassifierCV(decoder)
    return decoder
