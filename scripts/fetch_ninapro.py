"""Download NinaPro DB2 and convert it to the recording format of the README (one .npz per subject and exercise).

EMG: 12 Delsys electrodes at 2 kHz, converted from V to µV. Labels: the relabeled stimulus (restimulus), 0 = rest,
1-17 exercise 1, 18-40 exercise 2, 41-49 exercise 3. Trial: the repetition (1-6) of the last movement onset, so the
rest after a movement belongs to its repetition (the rest before the first onset to repetition 1).

    python scripts/fetch_ninapro.py --subjects 1-10
    dvc add data/raw/ninapro_db2 && git add data/raw/ninapro_db2.dvc && dvc push
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

URL = "https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s{}.zip"
FS = 2000
N_MOVEMENTS = 49
LABEL_NAMES = ["rest"] + [f"m{i}" for i in range(1, N_MOVEMENTS + 1)]


def parse_list(text):
    values = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        values += range(int(lo), int(hi or lo) + 1)
    return values


def download(subject, exercises, cache_dir):
    """Zip -> cache_dir/DB2_s<subject>/S<subject>_E<e>_A1.mat (only the requested exercises)."""
    out = cache_dir / f"DB2_s{subject}"
    mats = [out / f"S{subject}_E{e}_A1.mat" for e in exercises]
    if all(m.exists() for m in mats):
        return mats
    out.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"DB2_s{subject}.zip"
    if not zip_path.exists():
        print(f"downloading {URL.format(subject)}")
        part = zip_path.with_suffix(".part")
        with urllib.request.urlopen(URL.format(subject)) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f)
        part.rename(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for m in mats:
            with z.open(f"DB2_s{subject}/{m.name}") as src, open(m, "wb") as dst:
                shutil.copyfileobj(src, dst)
    zip_path.unlink()
    return mats


def trial_ids(stimulus, repetition):
    """Repetition of the last movement onset for every sample."""
    onsets = np.flatnonzero(np.diff((stimulus > 0).astype(np.int8)) == 1) + 1
    trial = np.zeros(len(stimulus), dtype=np.int16)
    for start, stop in zip(onsets, [*onsets[1:], len(stimulus)], strict=True):
        trial[start:stop] = repetition[start]
    trial[: onsets[0]] = repetition[onsets[0]]
    return trial


def convert(subject, exercises, mats, out_dir):
    out = out_dir / f"s{subject}"
    out.mkdir(parents=True, exist_ok=True)
    for exercise, mat in zip(exercises, mats, strict=True):
        d = loadmat(mat, variable_names=["emg", "restimulus", "rerepetition"])
        length = min(len(d["emg"]), len(d["restimulus"]))
        stimulus, repetition = d["restimulus"][:length, 0], d["rerepetition"][:length, 0]
        np.savez(
            out / f"e{exercise}.npz",
            emg=(d["emg"][:length].T * 1e6).astype(np.float32),
            fs=FS,
            channel_names=[f"emg_{i + 1}" for i in range(d["emg"].shape[1])],
            labels=stimulus.astype(np.int16),
            label_names=LABEL_NAMES,
            trial=trial_ids(stimulus, repetition),
            meta_subject=subject,
            meta_exercise=exercise,
        )
        print(f"subject {subject} exercise {exercise}: {length / FS:.0f} s -> {out / f'e{exercise}.npz'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", default="1-10", help="e.g. 1-10 or 1,3,5 (DB2 has 40)")
    parser.add_argument("--exercises", default="1-2", help="1-3 (3: force patterns)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/ninapro_db2"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/external/ninapro_db2"))
    parser.add_argument("--keep-cache", action="store_true", help="keep the .mat files after conversion")
    args = parser.parse_args()
    exercises = parse_list(args.exercises)
    for subject in parse_list(args.subjects):
        mats = download(subject, exercises, args.cache_dir)
        convert(subject, exercises, mats, args.out_dir)
        if not args.keep_cache:
            shutil.rmtree(mats[0].parent)
