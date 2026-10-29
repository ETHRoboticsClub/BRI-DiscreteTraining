"""One class label per window from the per-sample labels, consistent with causal online decisions."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def label_windows(labels, end_samples, length, fs, cfg, trial=None):
    """-> labels [W], keep mask [W] (pure enough, inside one trial), trial per window [W] (or None)."""
    labels = np.asarray(labels)
    delay = round(cfg.delay_ms * fs / 1000)
    if delay:  # the EMG at sample n answers the cue given at n - delay
        labels = np.concatenate([np.full(delay, labels[0]), labels[:-delay]])
    end_samples = np.asarray(end_samples, dtype=np.int64)
    if not len(end_samples):
        empty = np.empty(0, dtype=labels.dtype)
        return empty, np.empty(0, dtype=bool), None if trial is None else empty
    covered = sliding_window_view(labels, length)[end_samples - length + 1]  # [W, L]
    if cfg.strategy == "last":
        y = covered[:, -1]
    elif cfg.strategy == "centre":
        y = covered[:, length // 2]
    else:  # majority
        classes = np.unique(covered)
        counts = np.stack([(covered == c).sum(axis=1) for c in classes], axis=1)
        y = classes[counts.argmax(axis=1)]
    purity = (covered == y[:, None]).mean(axis=1)
    keep = purity >= cfg.purity - 1e-9
    trial_w = None
    if trial is not None:
        trials = sliding_window_view(np.asarray(trial), length)[end_samples - length + 1]
        keep &= (trials == trials[:, :1]).all(axis=1)  # never across two trials
        trial_w = trials[:, -1]
    return y, keep, trial_w
