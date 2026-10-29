"""Per-channel signal quality metrics and status (ok / warning / bad). Decisions are made in monitoring.py."""

import numpy as np
from scipy import signal

OK, WARNING, BAD = "ok", "warning", "bad"


def saturation(raw, adc_range_uv, margin=0.01):
    """Share of samples at or near the ADC limits."""
    return (np.abs(raw) >= (1 - margin) * adc_range_uv).mean(axis=1)


def flatline(filtered):
    """Standard deviation per channel; near zero means electrode off or broken cable."""
    return filtered.std(axis=1)


def powerline_ratio(raw, fs, f0=50.0, harmonics=(100.0, 150.0), band=(20.0, 450.0), width=2.0):
    """Power within ±width Hz of f0 and harmonics relative to the rest of the EMG band, on the raw signal."""
    freqs, p = signal.welch(raw, fs, nperseg=min(int(fs), raw.shape[1]), axis=-1)
    line = np.zeros_like(freqs, dtype=bool)
    for f in [f0, *harmonics]:
        line |= np.abs(freqs - f) <= width
    emg = (freqs >= band[0]) & (freqs <= band[1]) & ~line
    return p[:, line].sum(axis=1) / (p[:, emg].sum(axis=1) + 1e-12)


def snr(filtered, sigma_rest, active=None):
    """RMS during contraction (samples where active, all if None) over σ_rest."""
    x = filtered if active is None else filtered[:, active]
    return np.sqrt((x.astype(np.float64) ** 2).mean(axis=1)) / (np.asarray(sigma_rest) + 1e-12)


def _status(values, thresholds, higher_is_bad):
    status = np.full(len(values), OK, dtype=object)
    sign = 1 if higher_is_bad else -1
    status[sign * values > sign * thresholds["warning"]] = WARNING
    status[sign * values > sign * thresholds["bad"]] = BAD
    return status


def assess(raw, filtered, fs, cfg, hw, filtering=None, sigma_rest=None, active=None):
    """Metrics {name: [C]} and status [C] (the worst over the enabled metrics)."""
    metrics, statuses = {}, []
    if "saturation" in cfg.enabled:
        metrics["saturation"] = saturation(raw, hw.adc_range_uv, cfg.saturation_margin)
        statuses.append(_status(metrics["saturation"], cfg.saturation, True))
    if "flatline" in cfg.enabled:
        metrics["std"] = flatline(filtered)
        statuses.append(_status(metrics["std"], cfg.min_std, False))
    if "powerline" in cfg.enabled:
        f0 = (filtering.notch if filtering else None) or 50.0
        harmonics = filtering.notch_harmonics if filtering and filtering.notch else (2 * f0, 3 * f0)
        band = (filtering.highpass or 0, filtering.lowpass or fs / 2) if filtering else (20.0, 450.0)
        metrics["powerline_ratio"] = powerline_ratio(raw, fs, f0, harmonics, band)
        statuses.append(_status(metrics["powerline_ratio"], cfg.max_powerline_ratio, True))
    if "snr" in cfg.enabled and sigma_rest is not None:
        metrics["snr"] = snr(filtered, sigma_rest, active)
        statuses.append(_status(metrics["snr"], cfg.min_snr, False))
    rank = {OK: 0, WARNING: 1, BAD: 2}
    worst = np.array([max(s, key=rank.get) for s in zip(*statuses, strict=True)]) if statuses else None
    return metrics, worst
