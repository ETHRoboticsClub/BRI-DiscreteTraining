"""Overlapping fixed-length windows from a continuous stream. Label-free, so identical in every path."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Windower:
    """Window k covers samples [kS, kS + L) since the last reset() and is emitted once its last sample arrives."""

    def __init__(self, cfg, fs):
        self.fs = fs
        self.length = round(fs * cfg.window_ms / 1000)
        self.step = round(fs * cfg.step_ms / 1000)
        self.rate = fs / self.step  # decisions per second
        self.reset()

    def reset(self):
        self._buf = None  # last L - 1 samples [C, <= L - 1]
        self._tbuf = np.empty(0)
        self._seen = 0  # samples received since reset

    def process(self, chunk, timestamps=None):
        """chunk [C, N] -> windows [W, C, L], end timestamps [W], end sample indices [W] (since reset)."""
        n = chunk.shape[1]
        if timestamps is None:
            timestamps = (self._seen + np.arange(n)) / self.fs
        buf = chunk if self._buf is None else np.concatenate([self._buf, chunk], axis=1)
        tbuf = np.concatenate([self._tbuf, timestamps])
        start = self._seen - (buf.shape[1] - n)  # absolute index of buf[:, 0]
        total = self._seen + n
        L, S = self.length, self.step
        k_first = max(0, -(-(self._seen - L + 1) // S))
        k_last = (total - L) // S
        if k_last >= k_first:
            offsets = np.arange(k_first, k_last + 1) * S - start
            windows = sliding_window_view(buf, L, axis=1)[:, offsets].transpose(1, 0, 2)
            ends = offsets + L - 1
            out = (np.ascontiguousarray(windows), tbuf[ends], ends + start)
        else:
            out = (np.empty((0, chunk.shape[0], L), np.float32), np.empty(0), np.empty(0, dtype=np.int64))
        keep = min(L - 1, buf.shape[1])
        self._buf, self._tbuf = buf[:, buf.shape[1] - keep :], tbuf[len(tbuf) - keep :]
        self._seen = total
        return out
