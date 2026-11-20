"""Neural decoders with the interface of src/decoders.py: fit / predict_proba / classes_ / adapt.

Trained with torch (the `nn` extra); inference is numpy only, from the exported weights, so a profile loads and runs
without torch. Variants (decoder.name):

    mlp        features -> hidden layers -> classes
    lda_stack  [LDA log-posteriors, features] -> MLP: refines the LDA (trained on out-of-fold posteriors)
    nn_lda     MLP encoder (trained with a softmax head) -> LDA head on its embedding, closed-form calibration
    cnn        1D-CNN on the raw windows (features.names: [RAW]), channels x samples
"""

from collections import OrderedDict

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.classifier import LDA


def validation_blocks(t, fraction, block=40):
    """Mask of every k-th contiguous block of `block` windows per class (k = 1 / fraction). Contiguous blocks keep
    overlapping windows on one side, and the blocks spread over subjects and repetitions."""
    mask = np.zeros(len(t), dtype=bool)
    if not fraction:
        return mask
    k = max(2, round(1 / fraction))
    for c in np.unique(t):
        idx = np.flatnonzero(t == c)
        mask[idx[(np.arange(len(idx)) // block) % k == k - 1]] = True
    return mask


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


class TorchDecoder:
    """Training loop shared by the variants: AdamW, class-balanced cross-entropy, early stopping on held-out blocks.

    Subclasses define _module(d, K) (a torch module returning logits), _forward(X) (the same in numpy on
    self.weights) and _head (the state_dict prefix that adapt fine-tunes)."""

    _head = ""

    def __init__(
        self,
        lr=1e-3,
        weight_decay=1e-4,
        dropout=0.2,
        batch_size=512,
        max_epochs=100,
        patience=10,
        val_fraction=0.15,
        adapt_epochs=100,
        adapt_lr=1e-3,
        random_state=0,
        device="cpu",
    ):
        self.lr, self.weight_decay, self.dropout = lr, weight_decay, dropout
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.val_fraction, self.device = val_fraction, device
        self.adapt_epochs, self.adapt_lr, self.random_state = adapt_epochs, adapt_lr, random_state
        self.classes_ = self.weights = self.epochs_ = None

    def _targets(self, y):
        t = np.searchsorted(self.classes_, y)
        if (t >= len(self.classes_)).any() or (self.classes_[t.clip(max=len(self.classes_) - 1)] != y).any():
            raise ValueError("labels outside the classes the decoder was trained on")
        return t

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
        self.classes_ = np.unique(y)
        module = self._new_module(X.shape[1])
        self.epochs_ = self._train(module, X, self._targets(y), module.parameters(), self.lr, self.max_epochs)
        self._export(module)
        return self

    def adapt(self, X, y):
        """Fine-tune the output layer on calibration windows (fixed epochs, no early stopping)."""
        import torch

        module = self._new_module(np.asarray(X).shape[1])
        module.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in self.weights.items()})
        head = [p for name, p in module.named_parameters() if name.startswith(f"{self._head}.")]
        self._train(
            module,
            np.asarray(X, dtype=np.float32),
            self._targets(np.asarray(y)),
            head,
            self.adapt_lr,
            self.adapt_epochs,
            validate=False,
        )
        self._export(module)
        return self

    def _new_module(self, d):
        import torch

        torch.manual_seed(self.random_state)
        return self._module(d, len(self.classes_))

    def _device(self):
        """Training device: cpu, cuda, mps or auto (CUDA or Apple MPS if present). Inference is always numpy."""
        import torch

        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def _export(self, module):
        self.weights = {k: v.detach().cpu().numpy().copy() for k, v in module.state_dict().items()}

    def _train(self, module, X, t, params, lr, epochs, validate=True):
        import torch

        gen = torch.Generator().manual_seed(self.random_state)
        val = validation_blocks(t, self.val_fraction) if validate else np.zeros(len(t), dtype=bool)
        if validate and (len(np.unique(t[~val])) < len(np.unique(t)) or not val.any()):
            val[:] = False  # too little data for a validation split: fixed epochs
        dev = self._device()
        module.to(dev)
        Xt, tt = torch.from_numpy(X[~val]).to(dev), torch.from_numpy(t[~val]).long().to(dev)
        Xv, tv = torch.from_numpy(X[val]).to(dev), torch.from_numpy(t[val]).long().to(dev)
        counts = np.bincount(t[~val], minlength=len(self.classes_)).astype(np.float32)
        weight = torch.from_numpy(np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0))
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight.float().to(dev))
        opt = torch.optim.AdamW(list(params), lr=lr, weight_decay=self.weight_decay)
        best, best_state, waited, ran = np.inf, None, 0, 0
        for epoch in range(epochs):
            ran = epoch + 1
            module.train()
            for idx in torch.randperm(len(Xt), generator=gen).to(dev).split(self.batch_size):
                opt.zero_grad()
                loss_fn(module(Xt[idx]), tt[idx]).backward()
                opt.step()
            if not val.any():
                continue
            module.eval()
            with torch.no_grad():
                loss = float(loss_fn(module(Xv), tv))
            if loss < best - 1e-4:
                best, waited = loss, 0
                best_state = {k: v.clone() for k, v in module.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break
        if best_state is not None:
            module.load_state_dict(best_state)
        module.to("cpu").eval()
        return ran

    def decision_function(self, X):
        return self._forward(np.asarray(X, dtype=np.float32))

    def predict_proba(self, X):
        return np.exp(log_softmax(self.decision_function(X))).astype(np.float32)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]

    def get_params(self):
        return {"classes": self.classes_, "weights": self.weights}

    def set_params(self, params):
        self.classes_, self.weights = params["classes"], params["weights"]
        return self


class MLP(TorchDecoder):
    """features [d] -> Linear-ReLU-Dropout per hidden layer -> Linear -> class logits."""

    def __init__(self, hidden=(128, 128), **kw):
        super().__init__(**kw)
        self.hidden = tuple(hidden)
        self._head = str(3 * len(self.hidden))  # index of the output Linear in the Sequential

    def _module(self, d, K):
        from torch import nn

        layers, width = [], d
        for h in self.hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(self.dropout)]
            width = h
        return nn.Sequential(*layers, nn.Linear(width, K))

    def _linear(self, i):
        return self.weights[f"{3 * i}.weight"], self.weights[f"{3 * i}.bias"]

    def embed(self, X):
        """Activations of the last hidden layer."""
        h = np.asarray(X, dtype=np.float32)
        for i in range(len(self.hidden)):
            W, b = self._linear(i)
            h = np.maximum(h @ W.T + b, 0)
        return h

    def _forward(self, X):
        W, b = self._linear(len(self.hidden))
        return self.embed(X) @ W.T + b


class NNLDA:
    """MLP encoder trained with a softmax head, then an LDA on its last hidden layer (the bottleneck). Calibration is
    the closed-form LDA.adapt on the embedding of the new user's windows: no gradient steps on the device."""

    def __init__(self, hidden=(128, 32), shrinkage=0.1, lambda_mu=0.5, lambda_sigma=0.5, **kw):
        self.encoder = MLP(hidden=hidden, **kw)
        self.lda = LDA(shrinkage=shrinkage, lambda_mu=lambda_mu, lambda_sigma=lambda_sigma)

    @property
    def classes_(self):
        return self.lda.classes_

    def fit(self, X, y):
        self.encoder.fit(X, y)
        self.lda.fit(self.encoder.embed(X), y)
        return self

    def adapt(self, X, y):
        self.lda.adapt(self.encoder.embed(X), y)
        return self

    def predict_proba(self, X):
        return self.lda.predict_proba(self.encoder.embed(X))

    def predict(self, X):
        return self.lda.predict(self.encoder.embed(X))


class LDAStack:
    """An MLP on [LDA log-posteriors, features]. The training posteriors are out-of-fold (contiguous blocks), so the
    MLP does not learn to trust the LDA more than it deserves on new data. Calibration fine-tunes the MLP's output
    layer and leaves the LDA as it is: an adapted LDA gives sharper posteriors than the out-of-fold ones the MLP was
    trained on, and the fixed MLP misreads them (balanced accuracy near chance on DB5 leave-one-subject-out)."""

    def __init__(self, folds=5, use_features=True, shrinkage=0.1, lambda_mu=0.5, lambda_sigma=0.5, hidden=(64,), **kw):
        self.folds, self.use_features = folds, use_features
        self.lda_kw = {"shrinkage": shrinkage, "lambda_mu": lambda_mu, "lambda_sigma": lambda_sigma}
        self.lda = LDA(**self.lda_kw)
        self.mlp = MLP(hidden=hidden, **kw)

    @property
    def classes_(self):
        return self.lda.classes_

    def _inputs(self, log_posterior, X):
        return np.hstack([log_posterior, X]).astype(np.float32) if self.use_features else log_posterior

    def _log_posterior(self, lda, X):
        return np.maximum(log_softmax(lda.decision_function(X)), -30).astype(np.float32)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
        self.lda.fit(X, y)
        oof = np.zeros((len(y), len(self.lda.classes_)), dtype=np.float32)
        fold = np.zeros(len(y), dtype=np.int64)
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)
            fold[idx] = (np.arange(len(idx)) * self.folds) // len(idx)  # contiguous blocks per class
        for f in range(self.folds):
            lda = LDA(**self.lda_kw).fit(X[fold != f], y[fold != f])
            cols = np.searchsorted(self.lda.classes_, lda.classes_)
            oof[fold == f] = -30
            oof[np.ix_(fold == f, cols)] = self._log_posterior(lda, X[fold == f])
        self.mlp.fit(self._inputs(oof, X), y)
        return self

    def adapt(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        self.mlp.adapt(self._inputs(self._log_posterior(self.lda, X), X), y)
        return self

    def predict_proba(self, X):
        return self.mlp.predict_proba(self._inputs(self._log_posterior(self.lda, X), np.asarray(X, np.float32)))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def conv1d(x, W, b, stride=1, padding=0):
    """numpy Conv1d: x [N, C_in, L], W [C_out, C_in, k] -> [N, C_out, L_out] (as torch.nn.Conv1d)."""
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
    windows = sliding_window_view(x, W.shape[-1], axis=2)[:, :, ::stride]  # [N, C_in, L_out, k]
    return np.einsum("nclk,ock->nol", windows, W, optimize=True) + b[None, :, None]


class CNN(TorchDecoder):
    """Raw windows (the RAW feature, flattened channel-major) -> Conv1d-ReLU blocks -> mean over time -> Linear."""

    def __init__(self, channels=None, filters=(32, 32, 64), kernel=5, strides=(1, 2, 2), device="auto", **kw):
        super().__init__(device=device, **kw)
        if channels is None:
            raise ValueError("cnn needs decoder.params.channels (the number of EMG channels)")
        self.channels, self.filters, self.kernel, self.strides = channels, tuple(filters), kernel, tuple(strides)
        self._head = "head"

    def _module(self, d, K):
        from torch import nn

        layers, width = [nn.Unflatten(1, (self.channels, d // self.channels))], self.channels
        for f, s in zip(self.filters, self.strides, strict=True):
            layers += [nn.Conv1d(width, f, self.kernel, stride=s, padding=self.kernel // 2), nn.ReLU()]
            width = f
        body = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(self.dropout))
        return nn.Sequential(OrderedDict(body=body, head=nn.Linear(width, K)))

    def _forward(self, X):
        h = X.reshape(len(X), self.channels, -1)
        for i, s in enumerate(self.strides):
            key = f"body.{1 + 2 * i}"
            h = np.maximum(
                conv1d(h, self.weights[f"{key}.weight"], self.weights[f"{key}.bias"], s, self.kernel // 2), 0
            )
        return h.mean(axis=2) @ self.weights["head.weight"].T + self.weights["head.bias"]
