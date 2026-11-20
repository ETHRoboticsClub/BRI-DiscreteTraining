"""Download NinaPro DB2 or DB5 and convert it to the recording format of the README (one .npz per subject and
exercise).

DB2: 12 Delsys electrodes at 2 kHz, converted from V to µV. Labels: the relabeled stimulus (restimulus), 0 = rest,
1-17 exercise 1, 18-40 exercise 2, 41-49 exercise 3.
DB5: 16 Myo electrodes (two armbands) at 200 Hz, raw Myo units (int8 range, ±128, not µV). The files number the
movements per exercise; they are offset to 1-12 exercise 1, 13-29 exercise 2, 30-52 exercise 3.
Trial: the repetition (1-6) of the last movement onset, so the rest after a movement belongs to its repetition (the
rest before the first onset to repetition 1).

    python scripts/fetch_ninapro.py --subjects 1-10
    python scripts/fetch_ninapro.py --db 5 --subjects 1-10 --exercises 1
    dvc add data/raw/ninapro_db2 && git add data/raw/ninapro_db2.dvc && dvc push
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

DATABASES = {
    2: {
        "url": "https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s{}.zip",
        "folder": "DB2_s{}",  # folder of the .mat files inside the zip
        "fs": 2000,
        "scale": 1e6,  # V -> µV
        "offsets": {1: 0, 2: 0, 3: 0},  # restimulus is already numbered across exercises
        "n_movements": 49,
        "subjects": 40,
    },
    5: {
        "url": "https://ninapro.hevs.ch/files/DB5_Preproc/s{}.zip",
        "folder": "s{}",
        "fs": 200,
        "scale": 1.0,  # raw Myo units
        "offsets": {1: 0, 2: 12, 3: 29},  # restimulus restarts at 1 in every exercise
        "n_movements": 52,
        "subjects": 10,
    },
}


def parse_list(text):
    values = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        values += range(int(lo), int(hi or lo) + 1)
    return values


def download(db, subject, exercises, cache_dir):
    """Zip -> cache_dir/s<subject>/S<subject>_E<e>_A1.mat (only the requested exercises)."""
    spec = DATABASES[db]
    out = cache_dir / f"s{subject}"
    mats = [out / f"S{subject}_E{e}_A1.mat" for e in exercises]
    if all(m.exists() for m in mats):
        return mats
    out.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"s{subject}.zip"
    if not zip_path.exists():
        url = spec["url"].format(subject)
        print(f"downloading {url}")
        part = zip_path.with_suffix(".part")
        with urllib.request.urlopen(url) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f)
        part.rename(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for m in mats:
            with z.open(f"{spec['folder'].format(subject)}/{m.name}") as src, open(m, "wb") as dst:
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


def convert(db, subject, exercises, mats, out_dir):
    spec = DATABASES[db]
    label_names = ["rest"] + [f"m{i}" for i in range(1, spec["n_movements"] + 1)]
    out = out_dir / f"s{subject}"
    out.mkdir(parents=True, exist_ok=True)
    for exercise, mat in zip(exercises, mats, strict=True):
        d = loadmat(mat, variable_names=["emg", "restimulus", "rerepetition"])
        length = min(len(d["emg"]), len(d["restimulus"]))
        stimulus, repetition = d["restimulus"][:length, 0], d["rerepetition"][:length, 0]
        labels = np.where(stimulus > 0, stimulus + spec["offsets"][exercise], 0)
        np.savez(
            out / f"e{exercise}.npz",
            emg=(d["emg"][:length].T * spec["scale"]).astype(np.float32),
            fs=spec["fs"],
            channel_names=[f"emg_{i + 1}" for i in range(d["emg"].shape[1])],
            labels=labels.astype(np.int16),
            label_names=label_names,
            trial=trial_ids(stimulus, repetition),
            meta_subject=subject,
            meta_exercise=exercise,
        )
        print(f"subject {subject} exercise {exercise}: {length / spec['fs']:.0f} s -> {out / f'e{exercise}.npz'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=int, default=2, choices=sorted(DATABASES))
    parser.add_argument("--subjects", default="1-10", help="e.g. 1-10 or 1,3,5 (DB2 has 40, DB5 10)")
    parser.add_argument("--exercises", default="1-2", help="1-3 (DB2 3: force patterns)")
    parser.add_argument("--out-dir", type=Path, help="default data/raw/ninapro_db<db>")
    parser.add_argument("--cache-dir", type=Path, help="default data/external/ninapro_db<db>")
    parser.add_argument("--keep-cache", action="store_true", help="keep the .mat files after conversion")
    args = parser.parse_args()
    out_dir = args.out_dir or Path(f"data/raw/ninapro_db{args.db}")
    cache_dir = args.cache_dir or Path(f"data/external/ninapro_db{args.db}")
    exercises = parse_list(args.exercises)
    for subject in parse_list(args.subjects):
        mats = download(args.db, subject, exercises, cache_dir)
        convert(args.db, subject, exercises, mats, out_dir)
        if not args.keep_cache:
            shutil.rmtree(mats[0].parent)
