"""Shrinkage LDA with calibration adaptation. numpy only, no file access (portable to a microcontroller)."""

import numpy as np


class LDA:
    """fit / adapt / predict_proba / get_params / set_params, the decoder interface Group 2 calls."""

    def __init__(self, shrinkage=0.1, priors="uniform", lambda_mu=0.5, lambda_sigma=0.5):
        self.shrinkage, self.priors = shrinkage, priors
        self.lambda_mu, self.lambda_sigma = lambda_mu, lambda_sigma
        self.classes_ = self.means = self.covariance = self.class_priors = self.W = self.b = None

    def _shrink(self, S):
        d = len(S)
        return (1 - self.shrinkage) * S + self.shrinkage * np.trace(S) / d * np.eye(d)

    def _solve(self):
        inv_means = np.linalg.solve(self.covariance, self.means.T).T  # [K, d]
        self.W = inv_means
        self.b = -0.5 * (inv_means * self.means).sum(axis=1) + np.log(self.class_priors)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
        self.classes_ = np.unique(y)
        self.means = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        centered = X - self.means[np.searchsorted(self.classes_, y)]
        self.covariance = self._shrink(centered.T @ centered / max(len(X) - len(self.classes_), 1))
        if self.priors == "uniform":
            self.class_priors = np.full(len(self.classes_), 1 / len(self.classes_))
        elif self.priors == "empirical":
            self.class_priors = np.array([(y == c).mean() for c in self.classes_])
        else:
            self.class_priors = np.asarray(self.priors, dtype=np.float64) / np.sum(self.priors)
        self._solve()
        return self

    def adapt(self, X_cal, y_cal):
        """mu_new = (1 - lambda_mu) mu_old + lambda_mu mu_cal per calibrated class, same for the covariance."""
        X_cal, y_cal = np.asarray(X_cal, dtype=np.float64), np.asarray(y_cal)
        centered = []
        for c in np.unique(y_cal):
            i = np.searchsorted(self.classes_, c)
            if i == len(self.classes_) or self.classes_[i] != c:
                raise ValueError(f"class {c} was not in the training data")
            mu_cal = X_cal[y_cal == c].mean(axis=0)
            self.means[i] = (1 - self.lambda_mu) * self.means[i] + self.lambda_mu * mu_cal
            centered.append(X_cal[y_cal == c] - mu_cal)
        centered = np.concatenate(centered)
        cov_cal = self._shrink(centered.T @ centered / max(len(centered) - 1, 1))
        self.covariance = (1 - self.lambda_sigma) * self.covariance + self.lambda_sigma * cov_cal
        self._solve()
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=np.float64) @ self.W.T + self.b

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]

    def get_params(self):
        return {
            "classes": self.classes_,
            "W": self.W,
            "b": self.b,
            "means": self.means,
            "covariance": self.covariance,
            "priors": self.class_priors,
        }

    def set_params(self, params):
        self.classes_, self.W, self.b = params["classes"], params["W"], params["b"]
        self.means, self.covariance, self.class_priors = params["means"], params["covariance"], params["priors"]
        return self
