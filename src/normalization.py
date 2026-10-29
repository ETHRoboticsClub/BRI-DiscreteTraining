"""Feature scaling with statistics fitted on calibration or training data."""

import numpy as np


class Normalizer:
    def __init__(self, cfg):
        self.method, self.eps = cfg.method, cfg.eps
        self.mean = self.std = self.feature_names = self.config_hash = None

    @property
    def fitted(self):
        return self.mean is not None

    def fit(self, X, feature_names, config_hash=None):
        X = np.asarray(X, dtype=np.float64)
        self.mean, self.std = X.mean(axis=0), X.std(axis=0)
        self.feature_names, self.config_hash = list(feature_names), config_hash
        return self

    def transform(self, X, feature_names=None):
        if self.method == "identity" or not len(X):
            return np.asarray(X, dtype=np.float32)
        if not self.fitted:
            raise RuntimeError("normalization statistics not fitted or loaded")
        if feature_names is not None and list(feature_names) != self.feature_names:
            raise ValueError("feature names differ from the ones the normalization was fitted on")
        return ((X - self.mean) / (self.std + self.eps)).astype(np.float32)

    def fit_transform(self, X, feature_names, config_hash=None):
        return self.fit(X, feature_names, config_hash).transform(X)

    def state_dict(self):
        return {
            "method": self.method,
            "mean": self.mean,
            "std": self.std,
            "feature_names": self.feature_names,
            "config_hash": self.config_hash,
        }

    def load_state_dict(self, state, feature_names):
        """Load stored statistics; rejected if they belong to another feature configuration."""
        if list(state["feature_names"]) != list(feature_names):
            raise ValueError("stored normalization statistics do not match the current feature configuration")
        self.mean, self.std = np.asarray(state["mean"]), np.asarray(state["std"])
        self.feature_names, self.config_hash = list(state["feature_names"]), state.get("config_hash")
        return self
