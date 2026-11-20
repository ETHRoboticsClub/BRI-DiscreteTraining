import importlib.util
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
PARAMS = yaml.safe_load((ROOT / "configs" / "ninapro_db2.yaml").read_text())  # the fake recordings are DB2-like
DB5 = yaml.safe_load((ROOT / "params.yaml").read_text())


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


def test_db5_config_at_200_hz():
    cfg = from_dict(DB5)
    assert (cfg.hardware.fs, cfg.window_samples, cfg.step_samples) == (200, 40, 10)
    rng = np.random.default_rng(2)
    rec = Recording(rng.standard_normal((16, 2000)).astype(np.float32) * 5, 200.0, [f"c{i}" for i in range(16)])
    pipe = Pipeline(cfg, rec.channel_names)
    batch = pipe.process_recording(rec, chunk_seconds=1.3)
    assert batch.features.shape == ((2000 - 40) // 10 + 1, 16 * 3) and np.isfinite(batch.features).all()
    with pytest.raises(ConfigError, match="spectral_band"):
        from_dict({**DB5, "features": {**DB5["features"], "spectral_band": [20, 450]}})


def test_rms_feature():
    from src.feature_extraction import rms

    x = np.array([[[3.0, -3.0, 3.0, -3.0]], [[1.0, 1.0, 1.0, 1.0]]], dtype=np.float32)
    assert rms(x).ravel().tolist() == [3.0, 1.0]


def test_config_validation():
    with pytest.raises(ConfigError, match="fs/2"):
        config(filtering={"lowpass": 1200})
    with pytest.raises(ConfigError, match="unknown"):
        config(features={"names": ["MAV", "NOPE"]})
    assert config().hash != config(windowing={"step_ms": 25}).hash
    assert config().hash == config(decoder={"name": "svm"}).hash


@pytest.mark.parametrize("regime", ["per_user", "pooled", "loso"])
def test_experiment_masks(regime):
    from src.evaluation import experiments

    cfg = config(evaluation={"regime": regime})
    names = ["subject", "exercise", "trial", "recording"]
    columns = np.array([[s, 1, t, s] for s in (1, 2, 3) for t in range(1, 7) for _ in range(4)])
    runs = experiments(columns, names, cfg)
    assert len(runs) == (1 if regime == "pooled" else 3)
    for _, train, test, user in runs:
        assert not (train & test).any()
        assert set(columns[test, 2]) == {2, 5}  # the holdout trials of the config
        if regime == "loso":
            assert set(columns[train, 0]) == {1, 2, 3} - {user} and set(columns[test, 0]) == {user}
        if regime == "per_user":
            assert set(columns[train | test, 0]) == {user}


@pytest.mark.parametrize(
    "decoder, regime",
    [
        ({"name": "lda"}, "per_user"),
        ({"name": "random_forest", "params": {"n_estimators": 20}}, "per_user"),
        ({"name": "lda"}, "pooled"),
        ({"name": "lda"}, "loso"),
        pytest.param(
            {"name": "mlp", "params": {"max_epochs": 20, "adapt_epochs": 5}},
            "loso",
            marks=pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs the nn extra"),
        ),
    ],
)
def test_build_and_train(tmp_path, decoder, regime):
    params = json.loads(json.dumps(PARAMS))
    params["dataset"]["where"] = None
    params["decoder"] = decoder
    params["evaluation"]["regime"] = regime
    params["evaluation"]["conditions"] = ["subject"]
    params["evaluation"]["calibration_seconds"] = [0.5, 1]
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))
    write_recordings(tmp_path / params["dataset"]["raw_dir"])
    env = {"PYTHONPATH": str(ROOT)}
    for script in ("build_features.py", "train.py"):
        subprocess.run([sys.executable, ROOT / "scripts" / script], cwd=tmp_path, env=env, check=True)

    out = tmp_path / params["evaluation"]["out_dir"]
    metrics = json.loads((out / "metrics.json").read_text())
    if regime != "loso":  # every fake subject has its own random channel patterns: nothing transfers
        assert metrics["test"]["accuracy"] > 0.8 and metrics["test"]["balanced_accuracy"] > 0.8
    assert set(metrics["per_user"]) == {"1", "2"}
    assert set(metrics["calibration"]) == (set() if regime == "pooled" else {"0.5", "1"})
    assert 0 < metrics["latency_us"]["decoder"] < 10_000
    assert (out / "confusion.png").exists()
    profile = joblib.load(out / "profiles" / "subject1.joblib")
    pipe = Pipeline(from_dict(params), [f"emg_{i + 1}" for i in range(12)])
    pipe.load_profile(profile)
    emg, _, _ = fake_recording(np.random.default_rng(5), reps=1)
    batch = pipe.process(emg[:, :4000])
    assert profile["decoder"].predict_proba(batch.features).shape == (len(batch), 4)


def blobs(rng, classes=4, n=300, d=12):
    centres = rng.normal(0, 2, (classes, d))
    y = np.repeat(np.arange(classes), n)
    return (centres[y] + rng.normal(0, 1, (len(y), d))).astype(np.float32), y


@pytest.mark.parametrize("name", ["mlp", "lda_stack", "nn_lda", "cnn"])
def test_nn_decoders(name):
    torch = pytest.importorskip("torch")
    from src.decoders import make_decoder

    rng = np.random.default_rng(0)
    X, y = blobs(rng, d=4 * 10)  # cnn: 4 channels x 10 samples
    params = {"max_epochs": 30, "patience": 5, "adapt_epochs": 5}
    if name == "cnn":
        params["channels"] = 4
    decoder = make_decoder(config(decoder={"name": name, "params": params}).decoder, seed=0).fit(X, y + 1)
    proba = decoder.predict_proba(X)
    assert proba.shape == (len(y), 4) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)
    assert (decoder.classes_[proba.argmax(axis=1)] == y + 1).mean() > 0.8
    decoder.adapt(X[::10] + 0.1, y[::10] + 1)
    assert decoder.predict_proba(X).shape == (len(y), 4)
    with pytest.raises(ValueError):
        decoder.adapt(X[:5], np.full(5, 99))
    if name in ("mlp", "cnn"):  # the numpy inference equals the torch module
        module = decoder._new_module(X.shape[1])
        module.load_state_dict({k: torch.from_numpy(v) for k, v in decoder.weights.items()})
        with torch.no_grad():
            expected = module.eval()(torch.from_numpy(X)).numpy()
        np.testing.assert_allclose(decoder.decision_function(X), expected, rtol=1e-4, atol=1e-4)
        clone = type(decoder)(**({"channels": 4} if name == "cnn" else {})).set_params(decoder.get_params())
        np.testing.assert_allclose(clone.predict_proba(X), decoder.predict_proba(X))


def test_raw_feature_and_validation_blocks():
    from src.feature_extraction import raw
    from src.nn import validation_blocks

    x = np.arange(2 * 3 * 40, dtype=np.float32).reshape(2, 3, 40)
    assert np.array_equal(raw(x, raw_samples=40), x) and raw(x, raw_samples=5).shape == (2, 3, 5)
    t = np.repeat([0, 1], 400)
    val = validation_blocks(t, 0.25, block=40)
    assert val.sum() == 160 and all(val[t == c].sum() == 80 for c in (0, 1))  # blocks 3 and 7 of 10 per class
    assert (np.diff(np.flatnonzero(val)[:40]) == 1).all()  # contiguous blocks


def write_femg_session(root, dataset, subject, session, rng, reps=4, seconds=1.0, fs=200, label_fs=100):
    """A processed session of the dataset repo: rest / gesture blocks, two segments, unlabeled frames at the start."""
    patterns = rng.uniform(0.2, 1, (3, 8))
    patterns[0] = 0.05
    frames = []
    for _ in range(reps):
        for g in (1, 2):
            frames += [0] * int(label_fs * seconds) + [g] * int(label_fs * seconds)
    frames = np.array([-1] * 20 + frames)
    labels = np.repeat(frames.clip(min=0), fs // label_fs)
    emg = (patterns[labels] * rng.standard_normal((len(labels), 8))).astype(np.float32)
    folder = root / dataset / subject / session
    folder.mkdir(parents=True)
    np.save(folder / "virtual.npy", emg)
    np.save(folder / "stimulus.npy", np.where(frames > 0, frames + 10, frames).astype(np.int16))  # native cue ids
    np.save(folder / "gesture.npy", frames.astype(np.int16))
    half = len(frames) // 2
    meta = {
        "dataset": dataset,
        "subject": f"{dataset}/{subject}",
        "session": session,
        "fs": fs,
        "label_fs": label_fs,
        "segments": [[0, half], [half, len(frames)]],
        "gesture_names": ["rest", "fist", "pinch"],
        "gesture_families": ["rest", "hand_posture", "precision"],
    }
    (folder / "meta.json").write_text(json.dumps(meta))
    return frames


def femg_config(root, **dataset):
    params = yaml.safe_load((ROOT / "configs" / "femg.yaml").read_text())
    params["dataset"].update(raw_dir=str(root), **dataset)
    return from_dict(params)


def test_femg_reader(tmp_path):
    from src.dataset import load_recordings

    rng = np.random.default_rng(0)
    write_femg_session(tmp_path, "ds_a", "s01", "A1", rng)
    write_femg_session(tmp_path, "ds_b", "s01", "A1", rng)
    recs = list(load_recordings(femg_config(tmp_path)))
    assert len(recs) == 4  # 2 sessions x 2 segments
    assert {r.names["subject"] for r in recs} == {"ds_a/s01", "ds_b/s01"}
    assert recs[0].meta["subject"] != recs[2].meta["subject"]
    first, second = recs[0], recs[1]
    assert first.signal.shape[0] == 8 and first.label_names == ["rest", "fist", "pinch"]
    assert (first.labels[:40] == -1).all() and set(first.labels[40:]) == {0, 1, 2}
    assert first.signal.shape[1] + second.signal.shape[1] == 2 * (20 + 4 * 2 * 200)
    assert second.trial.max() == 4  # repetitions continue across the segments of a session

    only_b = list(load_recordings(femg_config(tmp_path, where={"dataset": ["ds_b"]})))
    assert {r.names["dataset"] for r in only_b} == {"ds_b"}
    fam = next(iter(load_recordings(femg_config(tmp_path, label_level="family"))))
    assert fam.label_names == ["rest", "hand_posture", "precision"]
    stim = next(iter(load_recordings(femg_config(tmp_path, label_level="stimulus"))))
    assert set(stim.labels) == {-1, 0, 11, 12} and stim.label_names is None


def test_femg_build_drops_unlabeled_and_holds_out_last_reps(tmp_path):
    rng = np.random.default_rng(1)
    raw = tmp_path / "forearm"
    for s in ("s01", "s02"):
        write_femg_session(raw, "ds_a", s, "A1", rng)
    params = yaml.safe_load((ROOT / "configs" / "femg.yaml").read_text())
    params["dataset"].update(raw_dir=str(raw), out_dir=str(tmp_path / "features"))
    params["evaluation"].update(out_dir=str(tmp_path / "runs"), calibration_seconds=[0.5])
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))
    env = {"PYTHONPATH": str(ROOT)}
    for script in ("build_features.py", "train.py"):
        subprocess.run([sys.executable, ROOT / "scripts" / script], cwd=tmp_path, env=env, check=True)
    info = json.loads((tmp_path / "features" / "info.json").read_text())
    columns, y = np.load(tmp_path / "features" / "columns.npy"), np.load(tmp_path / "features" / "labels.npy")
    trial, last = columns[:, info["columns"].index("trial")], columns[:, info["columns"].index("trial_from_end")]
    assert y.min() >= 0 and set(np.unique(y)) == {0, 1, 2}
    assert np.array_equal(np.isin(trial, [3, 4]), np.isin(last, [1, 2]))
    assert info["meta_names"]["subject"] == {"0": "ds_a/s01", "1": "ds_a/s02"}
    metrics = json.loads((tmp_path / "runs" / "metrics.json").read_text())
    assert metrics["test"]["balanced_accuracy"] > 0.8


def test_repetitions():
    from src.labeling import repetitions

    assert repetitions(np.zeros(3, dtype=int)).tolist() == [1, 1, 1]
    labels = np.array([0, 1, 1, 0, 2, 0, 1, 0, 2, 2])
    assert repetitions(labels).tolist() == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
