"""Decoder factory: our LDA or any scikit-learn classifier with fit / predict_proba.

params.yaml:
    decoder: {name: svm, params: {C: 10}}
    decoder: {name: mlp, params: {hidden: [128, 128]}}     # src/nn.py
    decoder: {name: sklearn.ensemble.ExtraTreesClassifier, params: {n_estimators: 300}}
"""

import importlib
import inspect

from src.classifier import LDA
from src.nn import CNN, MLP, NNLDA, LDAStack

ALIASES = {
    "lda": LDA,
    "mlp": MLP,  # src/nn.py: trained with torch, numpy inference
    "lda_stack": LDAStack,
    "nn_lda": NNLDA,
    "cnn": CNN,
    "sklearn_lda": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
    "qda": "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis",
    "logreg": "sklearn.linear_model.LogisticRegression",
    "svm": "sklearn.svm.SVC",
    "linear_svm": "sklearn.svm.LinearSVC",
    "knn": "sklearn.neighbors.KNeighborsClassifier",
    "random_forest": "sklearn.ensemble.RandomForestClassifier",
    "extra_trees": "sklearn.ensemble.ExtraTreesClassifier",
    "gradient_boosting": "sklearn.ensemble.HistGradientBoostingClassifier",
    "sklearn_mlp": "sklearn.neural_network.MLPClassifier",
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
    signature = inspect.signature(cls).parameters
    takes_seed = "random_state" in signature or any(p.kind == p.VAR_KEYWORD for p in signature.values())
    if seed is not None and takes_seed:
        params.setdefault("random_state", seed)
    decoder = cls(**params)
    if not hasattr(decoder, "predict_proba"):  # e.g. LinearSVC: calibrated probabilities
        from sklearn.calibration import CalibratedClassifierCV

        decoder = CalibratedClassifierCV(decoder)
    return decoder
