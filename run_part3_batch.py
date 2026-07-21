#!/usr/bin/env python
"""
run_part3_batch.py  --  Batch-run part3a-d (GPU training) across many buckets ON RC.

WHY: each part3 notebook hardcodes BRAND_BUCKET; running 11 buckets x 4 variants by
hand = 44 manual edits. This loops them, injects BRAND_BUCKET, executes on the GPU,
SKIPS variants already done (so a disconnect/crash resumes cleanly), and keeps going
past a failure.

HOW TO RUN (on Explorer):
    # 1) get a dedicated GPU + survive disconnects
    tmux new -s part3
    srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=8:00:00 --pty bash
    conda activate shoes
    cd /home/yin.yuhu                      # project root (has data_preparation/ + code/)

    # 2) one-time: make sure the env can drive notebooks
    python -c "import nbformat, nbconvert" || pip install nbformat nbconvert ipykernel
    #    if ExecutePreprocessor can't find the kernel, register it once:
    #    python -m ipykernel install --user --name shoes
    #    then run with:  PART3_KERNEL=shoes python run_part3_batch.py

    # 3) run
    python run_part3_batch.py
    #    (detach tmux with Ctrl-b d; reattach: tmux attach -t part3)

Edit BUCKETS below if you want a different set. Set PART3_KERNEL env var if the
default 'python3' kernel does not point at the shoes env.
"""
import os
import re
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent
PREP = ROOT / "data_preparation"
DATA = ROOT / "data"
KERNEL = os.environ.get("PART3_KERNEL", "python3")
TIMEOUT = int(os.environ.get("PART3_TIMEOUT", "10800"))  # 3h per notebook, plenty

BUCKETS = [
    "women_asics", "women_reebok", "women_heydude", "women_dr_scholl_s_shoes",
    "men_asics", "men_reebok", "men_heydude", "men_olukai",
    "women_adidas", "women_rest", "men_rest",
]

# notebook -> the predictions subfolder it fills (used to skip already-done variants)
NOTEBOOKS = [
    ("00_part3a_train_txtimg_lag1.ipynb",     "txtimg/lag1"),
    ("00_part3b_train_txtimg_time_ind.ipynb", "txtimg/time_indipendent"),
    ("00_part3c_train_txt_lag1.ipynb",        "txt/lag1"),
    ("00_part3d_train_txt_time_ind.ipynb",    "txt/time_indipendent"),
]


def set_bucket(nb, bucket):
    pat = re.compile(r"^BRAND_BUCKET\s*=\s*['\"][^'\"]*['\"]", re.M)
    for c in nb.cells:
        if c.cell_type == "code" and pat.search(c.source):
            c.source = pat.sub(f"BRAND_BUCKET = '{bucket}'", c.source)


def already_done(bucket, subfolder):
    d = DATA / bucket / "predictions" / subfolder
    return d.is_dir() and len(list(d.glob("*.zip"))) >= 6  # 6 zips per variant


def run_one(nb_name, bucket):
    nb = nbformat.read(PREP / nb_name, as_version=4)
    set_bucket(nb, bucket)
    ep = ExecutePreprocessor(timeout=TIMEOUT, kernel_name=KERNEL)
    ep.preprocess(nb, {"metadata": {"path": str(PREP)}})
    log = DATA / bucket / "_run_logs"
    log.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, log / nb_name)


def main():
    total = len(BUCKETS) * len(NOTEBOOKS)
    done = skipped = failed = 0
    failures = []
    print(f"part3 batch: {len(BUCKETS)} buckets x {len(NOTEBOOKS)} variants = {total} runs")
    print(f"kernel={KERNEL}  timeout={TIMEOUT}s\n")
    for b in BUCKETS:
        for nb_name, subfolder in NOTEBOOKS:
            tag = f"{b}/{subfolder}"
            if already_done(b, subfolder):
                print(f"[skip] {tag} (already has >=6 zips)")
                skipped += 1
                continue
            print(f"[run ] {tag} ...", flush=True)
            t0 = time.time()
            try:
                run_one(nb_name, b)
                done += 1
                print(f"       done {time.time()-t0:.0f}s")
            except Exception as e:  # noqa: BLE001
                failed += 1
                failures.append(f"{tag}: {type(e).__name__}: {str(e)[-300:]}")
                print(f"       FAILED: {type(e).__name__} (continuing)")
    print("\n" + "=" * 60)
    print(f"done={done}  skipped={skipped}  failed={failed}  / {total}")
    for f in failures:
        print("  FAIL:", f)
    print("=" * 60)


if __name__ == "__main__":
    main()