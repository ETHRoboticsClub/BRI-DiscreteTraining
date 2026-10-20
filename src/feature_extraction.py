"""Features per window and channel with a fixed, named column order.

Every feature function maps windows [N, C, L] to [N, C, k]. Add one by writing the function and
registering it in REGISTRY with the names of its k values.
"""

import numpy as np
from scipy import signal


def mav(x, **_):
    return np.abs(x).mean(axis=-1, keepdims=True)


def wl(x, **_):
    return np.abs(np.diff(x, axis=-1)).sum(axis=-1, keepdims=True)


def wamp(x, sigma_rest=None, wamp_k=3.0, **_):
    """Number of consecutive-sample differences above wamp_k * σ_rest (relative, so robust to gain changes)."""
    if sigma_rest is None:
        raise ValueError("WAMP needs σ_rest per channel (from calibration)")
    threshold = wamp_k * np.asarray(sigma_rest, dtype=np.float32)[None, :, None]
    return (np.abs(np.diff(x, axis=-1)) > threshold).sum(axis=-1, keepdims=True).astype(np.float32)


def ar(x, ar_order=4, ar_method="burg", **_):
    """Coefficients phi of x[n] = sum_k phi_k x[n - k] + e[n]. Zero for windows without power."""
    x = x.astype(np.float64) - x.mean(axis=-1, keepdims=True)
    if ar_method == "yule_walker":
        L = x.shape[-1]
        r = np.stack([(x[..., : L - k] * x[..., k:]).sum(-1) / L for k in range(ar_order + 1)], axis=-1)
        idx = np.abs(np.subtract.outer(np.arange(ar_order), np.arange(ar_order)))
        R = r[..., idx] + 1e-12 * np.eye(ar_order)
        return np.linalg.solve(R, r[..., 1:, None])[..., 0].astype(np.float32)
    # Burg, vectorized over windows and channels
    a = np.ones(x.shape[:-1] + (1,))
    f, b = x, x
    for _ in range(ar_order):
        f, b = f[..., 1:], b[..., :-1]
        k = -2 * (f * b).sum(-1) / ((f * f + b * b).sum(-1) + 1e-12)
        ext = np.concatenate([a, np.zeros(a.shape[:-1] + (1,))], axis=-1)
        a = ext + k[..., None] * ext[..., ::-1]
        f, b = f + k[..., None] * b, b + k[..., None] * f
    return (-a[..., 1:]).astype(np.float32)


def power_spectrum(x, fs, spectral_method="periodogram", spectral_taper="hann", spectral_band=None, **_):
    if spectral_method == "welch":
        freqs, p = signal.welch(x, fs, window=spectral_taper, nperseg=min(256, x.shape[-1]), axis=-1)
    else:
        freqs, p = signal.periodogram(x, fs, window=spectral_taper, axis=-1)
    if spectral_band:
        band = (freqs >= spectral_band[0]) & (freqs <= spectral_band[1])
        freqs, p = freqs[band], p[..., band]
    return freqs, p


def mnf(x, fs, **kw):
    """Mean frequency of the power spectrum. Also used by monitoring.py for fatigue."""
    freqs, p = power_spectrum(x, fs, **kw)
    total = p.sum(-1, keepdims=True)
    return np.where(total > 0, (p * freqs).sum(-1, keepdims=True) / np.where(total > 0, total, 1), 0).astype(np.float32)


def mdf(x, fs, **kw):
    """Median frequency of the power spectrum. Also used by monitoring.py for fatigue."""
    freqs, p = power_spectrum(x, fs, **kw)
    cum = np.cumsum(p, axis=-1)
    total = cum[..., -1:]
    idx = (cum < total / 2).sum(-1, keepdims=True).clip(max=len(freqs) - 1)
    return np.where(total > 0, freqs[idx], 0).astype(np.float32)


# name -> (function, names of its k values per channel)
REGISTRY = {
    "MAV": (mav, lambda p: ["MAV"]),
    "WL": (wl, lambda p: ["WL"]),
    "WAMP": (wamp, lambda p: ["WAMP"]),
    "AR": (ar, lambda p: [f"AR{i + 1}" for i in range(p["ar_order"])]),
    "MNF": (mnf, lambda p: ["MNF"]),
    "MDF": (mdf, lambda p: ["MDF"]),
}
AMPLITUDE = {"MAV", "WL"}


class FeatureExtractor:
    """windows [N, C, L] -> features [N, d] (or [N, C, F] without flatten), names [d], invalid flags [N]."""

    def __init__(self, cfg, fs, channel_names, sigma_rest=None):
        self.cfg, self.fs, self.channel_names = cfg, fs, list(channel_names)
        self.sigma_rest = sigma_rest
        self.params = {k: v for k, v in vars(cfg).items() if k not in ("names", "log_amplitude", "flatten")}
        self.per_channel = [n for name in cfg.names for n in REGISTRY[name][1](self.params)]
        self.names = [f"{ch}_{n}" for ch in self.channel_names for n in self.per_channel]

    def __call__(self, windows):
        if not len(windows):
            shape = (0, len(self.names)) if self.cfg.flatten else (0, len(self.channel_names), len(self.per_channel))
            return np.empty(shape, np.float32), np.empty(0, dtype=bool)
        values = []
        for name in self.cfg.names:
            v = REGISTRY[name][0](windows, fs=self.fs, sigma_rest=self.sigma_rest, **self.params)
            if self.cfg.log_amplitude and name in AMPLITUDE:
                v = np.log(v + 1e-6)
            values.append(v.astype(np.float32))
        features = np.concatenate(values, axis=-1) if values else np.empty(windows.shape[:2] + (0,), np.float32)
        invalid = ~(windows.std(axis=-1) > 0).all(axis=1)  # a channel without signal (electrode off)
        if self.cfg.flatten:
            features = features.reshape(len(features), -1)
        return features, invalid
