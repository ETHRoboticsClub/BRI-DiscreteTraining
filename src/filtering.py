"""Causal band-pass + notch filter chain with state per channel, identical offline and online."""

import numpy as np
from scipy import signal


def design_sos(
    fs, highpass=None, lowpass=None, order=4, filter_type="butter", notch=None, notch_q=30.0, notch_harmonics=(), **_
):
    sections = []
    design = {"butter": signal.butter, "bessel": signal.bessel}[filter_type]
    if highpass and lowpass:
        sections.append(design(order, [highpass, lowpass], btype="bandpass", fs=fs, output="sos"))
    elif highpass:
        sections.append(design(order, highpass, btype="highpass", fs=fs, output="sos"))
    elif lowpass:
        sections.append(design(order, lowpass, btype="lowpass", fs=fs, output="sos"))
    if notch:
        for f0 in [notch, *notch_harmonics]:
            b, a = signal.iirnotch(f0, notch_q, fs=fs)
            sections.append(signal.tf2sos(b, a))
    return np.concatenate(sections) if sections else np.empty((0, 6))


class Filter:
    """process(chunk [C, N]) -> filtered chunk [C, N], keeping the filter state between chunks."""

    def __init__(self, cfg, fs):
        self.fs, self.mode, self.init = fs, cfg.mode, cfg.init
        self.sos = design_sos(fs, **vars(cfg))
        self.transient = self._transient_length()
        self.reset()

    def reset(self):
        self._zi = None

    def _transient_length(self, tol=1e-3):
        """Samples until the impulse response has decayed below tol of its peak."""
        if not len(self.sos):
            return 0
        h = np.abs(signal.sosfilt(self.sos, np.r_[1.0, np.zeros(int(2 * self.fs))]))
        return int(np.flatnonzero(h > tol * h.max())[-1]) + 1

    def process(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float64)
        if not len(self.sos) or chunk.shape[1] == 0:
            return chunk.astype(np.float32)
        if self.mode == "zero_phase":
            return signal.sosfiltfilt(self.sos, chunk, axis=-1).astype(np.float32)
        if self._zi is None:
            zi = signal.sosfilt_zi(self.sos)[:, None, :]  # [sections, 1, 2]
            scale = chunk[:, 0][None, :, None] if self.init == "first_sample" else 0.0
            self._zi = zi * scale + np.zeros((len(self.sos), chunk.shape[0], 2))
        out, self._zi = signal.sosfilt(self.sos, chunk, axis=-1, zi=self._zi)
        return out.astype(np.float32)
