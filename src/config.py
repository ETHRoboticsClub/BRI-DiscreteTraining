"""Load params.yaml into typed configs, validate them and hash the parameters that change the features."""

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class HardwareConfig:
    fs: float = 2000.0
    n_channels: int = 12
    channel_names: list | None = None
    adc_range_uv: float = 11000.0  # ADC limit, ± µV


@dataclass
class DatasetConfig:
    raw_dir: str = "data/raw/ninapro_db2"
    out_dir: str = "data/processed/default"
    reader: str = "npz"
    meta_keys: list = field(default_factory=lambda: ["subject", "exercise"])
    where: dict | None = None  # selection of recordings, e.g. {subject: [1, 2]}
    classes: list | None = None  # class IDs to keep, None = all
    rest_class: int = 0
    user_key: str | None = "subject"  # one σ_rest, normalization and decoder per value (calibration profile)
    chunk_seconds: float = 10.0  # offline recordings are streamed through the pipeline in chunks of this length


@dataclass
class FilteringConfig:
    highpass: float | None = 20.0
    lowpass: float | None = 450.0
    order: int = 4
    filter_type: str = "butter"
    notch: float | None = 50.0
    notch_q: float = 30.0
    notch_harmonics: list = field(default_factory=lambda: [100.0, 150.0])
    mode: str = "causal"  # causal | zero_phase (offline analysis only, never for training a model that runs online)
    init: str = "first_sample"  # zeros | first_sample


@dataclass
class WindowingConfig:
    window_ms: float = 200.0
    step_ms: float = 50.0


@dataclass
class FeaturesConfig:
    names: list = field(default_factory=lambda: ["MAV", "WL", "WAMP", "AR", "MNF", "MDF"])
    ar_order: int = 4
    ar_method: str = "burg"  # burg | yule_walker
    wamp_k: float = 3.0  # WAMP threshold = wamp_k * σ_rest
    spectral_method: str = "periodogram"  # periodogram | welch
    spectral_taper: str = "hann"
    spectral_band: list | None = None  # [low, high] Hz, None = all frequencies
    log_amplitude: bool = False  # log of MAV and WL
    flatten: bool = True


@dataclass
class NormalizationConfig:
    method: str = "zscore"  # identity | zscore
    eps: float = 1e-6


@dataclass
class LabelingConfig:
    strategy: str = "last"  # last | majority | centre
    purity: float = 1.0  # windows with a lower share of their assigned label are dropped
    delay_ms: float = 0.0  # reaction time after the cue


@dataclass
class SignalQualityConfig:
    enabled: list = field(default_factory=lambda: ["saturation", "flatline", "powerline", "snr"])
    saturation_margin: float = 0.01  # share of the ADC range counted as "at the limit"
    saturation: dict = field(default_factory=lambda: {"warning": 0.001, "bad": 0.01})  # share of samples
    min_std: dict = field(default_factory=lambda: {"warning": 2.0, "bad": 0.5})  # µV
    max_powerline_ratio: dict = field(default_factory=lambda: {"warning": 0.3, "bad": 1.0})
    min_snr: dict = field(default_factory=lambda: {"warning": 3.0, "bad": 1.5})
    window_s: float | None = None  # analysis window, None = whole signal


@dataclass
class DecoderConfig:
    name: str = "lda"  # see src/decoders.py, or a dotted path like sklearn.svm.SVC
    params: dict = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    out_dir: str = "runs/default"
    split: str = "holdout"  # holdout | leave_one_group_out
    test: dict | None = None  # holdout: windows held out for the test, e.g. {trial: [2, 5]}
    group_key: str = "trial"  # leave_one_group_out: one fold per value of this column
    conditions: list = field(default_factory=list)  # columns to report accuracy per value of
    calibration_seconds: list = field(default_factory=list)  # accuracy vs training seconds per class
    seed: int = 0


@dataclass
class Config:
    hardware: HardwareConfig
    dataset: DatasetConfig
    filtering: FilteringConfig
    windowing: WindowingConfig
    features: FeaturesConfig
    normalization: NormalizationConfig
    labeling: LabelingConfig
    signal_quality: SignalQualityConfig
    decoder: DecoderConfig
    evaluation: EvaluationConfig

    @property
    def window_samples(self):
        return round(self.hardware.fs * self.windowing.window_ms / 1000)

    @property
    def step_samples(self):
        return round(self.hardware.fs * self.windowing.step_ms / 1000)

    @property
    def hash(self):
        """Hash of every parameter that changes the computed features. Stored with profiles and results."""
        keys = ("hardware", "filtering", "windowing", "features", "normalization")
        text = json.dumps({k: asdict(getattr(self, k)) for k in keys}, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    def to_dict(self):
        return asdict(self)


class ConfigError(ValueError):
    pass


def from_dict(params):
    sections = {}
    for f in fields(Config):
        try:
            sections[f.name] = f.type(**(params.get(f.name) or {}))
        except TypeError as e:
            raise ConfigError(f"params.yaml, section '{f.name}': {e}") from None
    unknown = set(params) - set(sections)
    if unknown:
        raise ConfigError(f"params.yaml: unknown sections {sorted(unknown)}")
    cfg = Config(**sections)
    validate(cfg)
    return cfg


def load_config(path="params.yaml"):
    return from_dict(yaml.safe_load(Path(path).read_text()))


def validate(cfg):
    from src.feature_extraction import REGISTRY

    errors = []
    hw, flt, win, feat = cfg.hardware, cfg.filtering, cfg.windowing, cfg.features
    nyquist = hw.fs / 2
    if flt.lowpass is not None and flt.lowpass >= nyquist:
        errors.append(f"filtering.lowpass {flt.lowpass} Hz must be below fs/2 = {nyquist} Hz")
    if flt.highpass is not None and flt.lowpass is not None and flt.highpass >= flt.lowpass:
        errors.append(f"filtering.highpass {flt.highpass} Hz must be below filtering.lowpass {flt.lowpass} Hz")
    for f0 in ([flt.notch] if flt.notch else []) + list(flt.notch_harmonics if flt.notch else []):
        if f0 >= nyquist:
            errors.append(f"notch frequency {f0} Hz must be below fs/2 = {nyquist} Hz")
    if flt.mode not in ("causal", "zero_phase"):
        errors.append(f"filtering.mode must be causal or zero_phase, not {flt.mode}")
    if not 0 < win.step_ms <= win.window_ms:
        errors.append(f"windowing: need 0 < step_ms ({win.step_ms}) <= window_ms ({win.window_ms})")
    if "AR" in feat.names and cfg.window_samples <= feat.ar_order:
        errors.append(f"window of {cfg.window_samples} samples is too short for AR order {feat.ar_order}")
    missing = [n for n in feat.names if n not in REGISTRY]
    if missing:
        errors.append(f"features.names: unknown {missing}, available: {sorted(REGISTRY)}")
    if cfg.normalization.method not in ("identity", "zscore"):
        errors.append(f"normalization.method must be identity or zscore, not {cfg.normalization.method}")
    if cfg.labeling.strategy not in ("last", "majority", "centre"):
        errors.append(f"labeling.strategy must be last, majority or centre, not {cfg.labeling.strategy}")
    if cfg.evaluation.split not in ("holdout", "leave_one_group_out"):
        errors.append(f"evaluation.split must be holdout or leave_one_group_out, not {cfg.evaluation.split}")
    if hw.channel_names is not None and len(hw.channel_names) != hw.n_channels:
        errors.append(f"hardware: {len(hw.channel_names)} channel_names for {hw.n_channels} channels")
    if errors:
        raise ConfigError("invalid params.yaml:\n  " + "\n  ".join(errors))
