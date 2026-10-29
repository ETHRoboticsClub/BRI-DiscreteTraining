import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
import yaml

from src.classifier import LDA
from src.config import ConfigError, from_dict
from src.labeling import label_windows
from src.normalization import Normalizer
from src.pipeline import Pipeline, Recording

ROOT = Path(__file__).parents[1]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())


def config(**overrides):
    params = json.loads(json.dumps(PARAMS))
    for section, values in overrides.items():
        params[section].update(values)
    return from_dict(params)


def fake_recording(rng, fs=2000, channels=12, classes=3, reps=6, seconds=1.5):
    """Rest - gesture - rest blocks; each gesture activates its own channel pattern."""
    patterns = rng.uniform(0.2, 1, (classes + 1, channels))
    patterns[0] = 0.05
    labels, trial = [], []
    for rep in range(1, reps + 1):
        for c in range(1, classes + 1):
            labels += [0] * int(fs * seconds) + [c] * int(fs * seconds)
            trial += [rep] * int(2 * fs * seconds)
    labels, trial = np.array(labels), np.array(trial)
    emg = 50 * patterns[labels].T * rng.standard_normal((channels, len(labels))) + rng.standard_normal(len(labels))
    return emg.astype(np.float32), labels, trial


def write_recordings(raw_dir, subjects=(1, 2)):
    rng = np.random.default_rng(0)
    for s in subjects:
        (raw_dir / f"s{s}").mkdir(parents=True)
        emg, labels, trial = fake_recording(rng)
        np.savez(
            raw_dir / f"s{s}" / "e1.npz",
            emg=emg,
            fs=2000,
            channel_names=[f"emg_{i + 1}" for i in range(12)],
            labels=labels,
            label_names=["rest", "m1", "m2", "m3"],
            trial=trial,
            meta_subject=s,
            meta_exercise=1,
        )


def test_streaming_matches_one_chunk():
    cfg = config()
    rng = np.random.default_rng(1)
    rec = Recording(rng.standard_normal((12, 7321)).astype(np.float32) * 20, 2000.0, [f"c{i}" for i in range(12)])
    pipe = Pipeline(cfg, rec.channel_names)
    pipe.sigma_rest = np.ones(12)
    whole = pipe.process_recording(rec)
    chunked = pipe.process_recording(rec, chunk_seconds=0.0137)
    L, S = cfg.window_samples, cfg.step_samples
    assert len(whole) == (7321 - L) // S + 1
    assert np.array_equal(whole.end_samples, np.arange(len(whole)) * S + L - 1)
    np.testing.assert_allclose(chunked.features, whole.features, rtol=1e-4, atol=1e-4)
    assert whole.features.shape[1] == len(pipe.feature_names) == 12 * 9


def test_labeling_strategies():
    labels = np.array([0] * 10 + [1] * 10)
    cfg = config().labeling
    ends = np.array([9, 14, 19])
    y, keep, _ = label_windows(labels, ends, 10, 1000, cfg)
    assert y.tolist() == [0, 1, 1] and keep.tolist() == [True, False, True]
    cfg.strategy, cfg.purity = "majority", 0.5
    y, keep, _ = label_windows(labels, ends, 10, 1000, cfg)
    assert y.tolist() == [0, 0, 1] and keep.all()
    _, keep, trial = label_windows(labels, ends, 10, 1000, cfg, trial=np.repeat([1, 2], 10))
    assert keep.tolist() == [True, False, True] and trial.tolist() == [1, 2, 2]


def test_lda_adapt_and_params():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(m, 1, (200, 5)) for m in (0, 3, 6)])
    y = np.repeat([0, 1, 2], 200)
    lda = LDA(shrinkage=0.1).fit(X, y)
    p = lda.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1) and (lda.predict(X) == y).mean() > 0.9
    shifted = X + 1.0
    before = (lda.predict(shifted) == y).mean()
    lda.adapt(shifted, y)
    assert (lda.predict(shifted) == y).mean() >= before
    clone = LDA().set_params(lda.get_params())
    np.testing.assert_allclose(clone.predict_proba(X), lda.predict_proba(X))


def test_normalization_rejects_other_features():
    norm = Normalizer(config().normalization).fit(np.ones((4, 2)), ["a", "b"])
    with pytest.raises(ValueError):
        Normalizer(config().normalization).load_state_dict(norm.state_dict(), ["a", "c"])


def test_config_validation():
    with pytest.raises(ConfigError, match="fs/2"):
        config(filtering={"lowpass": 1200})
    with pytest.raises(ConfigError, match="unknown"):
        config(features={"names": ["MAV", "NOPE"]})
    assert config().hash != config(windowing={"step_ms": 25}).hash
    assert config().hash == config(decoder={"name": "svm"}).hash


@pytest.mark.parametrize("decoder", [{"name": "lda"}, {"name": "random_forest", "params": {"n_estimators": 20}}])
def test_build_and_train(tmp_path, decoder):
    params = json.loads(json.dumps(PARAMS))
    params["dataset"]["where"] = None
    params["decoder"] = decoder
    params["evaluation"]["conditions"] = ["subject"]
    params["evaluation"]["calibration_seconds"] = [0.5, 1]
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))
    write_recordings(tmp_path / params["dataset"]["raw_dir"])
    env = {"PYTHONPATH": str(ROOT)}
    for script in ("build_features.py", "train.py"):
        subprocess.run([sys.executable, ROOT / "scripts" / script], cwd=tmp_path, env=env, check=True)

    out = tmp_path / params["evaluation"]["out_dir"]
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["test"]["accuracy"] > 0.8
    assert set(metrics["per_user"]) == {"1", "2"} and set(metrics["calibration"]) == {"0.5", "1"}
    assert (out / "confusion.png").exists()
    profile = joblib.load(out / "profiles" / "subject1.joblib")
    pipe = Pipeline(from_dict(params), [f"emg_{i + 1}" for i in range(12)])
    pipe.load_profile(profile)
    emg, _, _ = fake_recording(np.random.default_rng(5), reps=1)
    batch = pipe.process(emg[:, :4000])
    assert profile["decoder"].predict_proba(batch.features).shape == (len(batch), 4)
