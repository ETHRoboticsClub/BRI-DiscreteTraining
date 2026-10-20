"""filtering -> windowing -> feature_extraction -> normalization, one object for the offline, calibration and
online paths, plus the containers passed between modules."""

from dataclasses import dataclass, field

import numpy as np

from src.feature_extraction import FeatureExtractor
from src.filtering import Filter
from src.normalization import Normalizer
from src.windowing import Windower


@dataclass
class Recording:
    signal: np.ndarray  # [C, T] float32, µV
    fs: float
    channel_names: list
    timestamps: np.ndarray | None = None  # [T] s, sample index / fs if None
    labels: np.ndarray | None = None  # [T] class IDs
    trial: np.ndarray | None = None  # [T] trial (repetition) IDs
    meta: dict = field(default_factory=dict)  # subject, session, conditions...
    label_names: list | None = None  # label_names[class ID]
    config_hash: str | None = None

    def __post_init__(self):
        if self.timestamps is None:
            self.timestamps = np.arange(self.signal.shape[1]) / self.fs


@dataclass
class FeatureBatch:
    features: np.ndarray  # [W, d] normalized (unnormalized while no statistics are loaded)
    feature_names: list
    end_times: np.ndarray  # [W] s
    end_samples: np.ndarray  # [W] sample index since reset()
    invalid: np.ndarray  # [W] bool, a channel without signal
    mav: np.ndarray  # [W, C] unnormalized MAV, for the activation level of the controller
    intermediate: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.features)

    @staticmethod
    def concat(batches, feature_names):
        cat = {k: np.concatenate([getattr(b, k) for b in batches]) for k in _ARRAYS}
        inter = {  # "filtered" is per sample [C, N], the others per window
            k: np.concatenate([b.intermediate[k] for b in batches], axis=1 if k == "filtered" else 0)
            for k in batches[0].intermediate
        }
        return FeatureBatch(feature_names=feature_names, intermediate=inter, **cat)


_ARRAYS = ("features", "end_times", "end_samples", "invalid", "mav")


class Pipeline:
    def __init__(self, cfg, channel_names=None, profile=None, expose=()):
        """expose: intermediate outputs to return, any of "filtered", "windows", "raw_features"."""
        self.cfg, self.fs = cfg, cfg.hardware.fs
        default = [f"ch{i}" for i in range(cfg.hardware.n_channels)]
        self.channel_names = [str(c) for c in (channel_names or cfg.hardware.channel_names or default)]
        self.expose = set(expose)
        self.filter = Filter(cfg.filtering, self.fs)
        self.windower = Windower(cfg.windowing, self.fs)
        self.extractor = FeatureExtractor(cfg.features, self.fs, self.channel_names)
        self.normalizer = Normalizer(cfg.normalization)
        if profile is not None:
            self.load_profile(profile)

    @property
    def feature_names(self):
        return self.extractor.names

    @property
    def sigma_rest(self):
        return self.extractor.sigma_rest

    @sigma_rest.setter
    def sigma_rest(self, value):
        self.extractor.sigma_rest = None if value is None else np.asarray(value, dtype=np.float32)

    def load_profile(self, profile):
        """Calibration profile: {config_hash, sigma_rest, normalization, ...}. Rejected on a config mismatch."""
        if profile["config_hash"] != self.cfg.hash:
            raise ValueError(f"profile made with config {profile['config_hash']}, current config is {self.cfg.hash}")
        self.sigma_rest = profile["sigma_rest"]
        if profile.get("normalization") is not None:
            self.normalizer.load_state_dict(profile["normalization"], self.feature_names)

    def reset(self):
        """Between trials, recordings, or after recalibration."""
        self.filter.reset()
        self.windower.reset()

    def process(self, chunk, timestamps=None):
        """Raw chunk [C, N] µV -> FeatureBatch of the windows completed by this chunk (can be empty)."""
        filtered = self.filter.process(chunk)
        windows, end_times, end_samples = self.windower.process(filtered, timestamps)
        raw, invalid = self.extractor(windows)
        mav = np.abs(windows).mean(axis=-1)
        features = self.normalizer.transform(raw) if self.normalizer.fitted else raw
        inter = {"filtered": filtered, "windows": windows, "raw_features": raw}
        return FeatureBatch(
            features, self.feature_names, end_times, end_samples, invalid, mav, {k: inter[k] for k in self.expose}
        )

    def process_recording(self, rec, chunk_seconds=None):
        """Whole recording from a reset state, streamed in chunks (same result as one chunk)."""
        self.reset()
        n = rec.signal.shape[1]
        size = n if not chunk_seconds else max(1, int(chunk_seconds * self.fs))
        batches = [self.process(rec.signal[:, s : s + size], rec.timestamps[s : s + size]) for s in range(0, n, size)]
        return FeatureBatch.concat(batches, self.feature_names)


def rest_sigma(filtered, labels, rest_class=0, skip=0):
    """σ_rest per channel: std of the filtered signal during rest, after the filter transient."""
    rest = labels == rest_class
    rest[:skip] = False
    if not rest.any():
        raise ValueError("no rest samples to estimate σ_rest")
    return filtered[:, rest].std(axis=1).astype(np.float32)
