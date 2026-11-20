"""One class label per window from the per-sample labels, consistent with causal online decisions."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def repetitions(labels, rest=0):
    """Per sample: the repetition of the last movement onset, each class's onsets numbered 1, 2, ... in order. The
    rest (and unlabeled samples) after a movement belong to its repetition, before the first onset to repetition 1."""
    labels = np.asarray(labels)
    moving = (labels != rest) & (labels >= 0)
    onsets = np.flatnonzero(moving & np.r_[True, labels[1:] != labels[:-1]])
    trial = np.ones(len(labels), dtype=np.int64)
    count = {}
    for start, stop in zip(onsets, [*onsets[1:], len(labels)][: len(onsets)], strict=True):
        count[labels[start]] = count.get(labels[start], 0) + 1
        trial[start:stop] = count[labels[start]]
    return trial


def from_end(trial, labels, recordings, rest=0):
    """Per window: 1 for the last repetition of its class in its recording group, 2 for the one before... Rest windows
    take the value of the movement repetition they follow. Lets `evaluation.test: {trial_from_end: [1, 2]}` hold out
    the last two repetitions of every dataset, whatever their number."""
    out = np.zeros(len(trial), dtype=np.int64)
    for group in np.unique(recordings):
        g = recordings == group
        moving = g & (labels != rest)
        last = {c: trial[moving & (labels == c)].max() for c in np.unique(labels[moving])}
        out[moving] = [last[c] - t + 1 for c, t in zip(labels[moving], trial[moving], strict=True)]
        top = max(last.values()) if last else 1
        out[g & (labels == rest)] = top - trial[g & (labels == rest)] + 1
    return out


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
