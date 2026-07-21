#!/usr/bin/env python
"""
run_part3_lora_batch.py  --  Batch-run the LoRA part3 (txtimg) across all *_lora
buckets ON RC, for the LoRA-vs-full-FT A/B test.

Mirrors run_part3_batch.py but uses the two LoRA notebooks (txtimg only) and the
*_lora buckets. Injects BRAND_BUCKET, runs on GPU, SKIPS variants already done
(resumable), continues past failures.

RUN (Explorer, GPU srun + tmux + conda activate shoes):
    pip install "peft==0.13.2" --no-deps          # if not already; --no-deps keeps transformers 4.44.2
    cd /home/yin.yuhu
    PART3_KERNEL=shoes python run_part3_lora_batch.py
"""
import os, re, time
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent
PREP = ROOT / "data_preparation"
DATA = ROOT / "data"
KERNEL = os.environ.get("PART3_KERNEL", "python3")
TIMEOUT = int(os.environ.get("PART3_TIMEOUT", "10800"))

BUCKETS = [
    "women_nike_lora", "women_asics_lora", "women_reebok_lora", "women_heydude_lora",
    "women_dr_scholl_s_shoes_lora", "women_adidas_lora", "women_rest_lora",
    "men_top2_nike_lora", "men_asics_lora", "men_reebok_lora", "men_heydude_lora",
    "men_olukai_lora", "men_rest_lora",
    "men_adidas_lora",  # txtimg already done -> auto-skipped, only its txt runs
]

NOTEBOOKS = [
    ("00_part3a_train_txtimg_lag1_LORA.ipynb",     "txtimg/lag1"),
    ("00_part3b_train_txtimg_time_ind_LORA.ipynb", "txtimg/time_indipendent"),
    ("00_part3c_train_txt_lag1_LORA.ipynb",        "txt/lag1"),
    ("00_part3d_train_txt_time_ind_LORA.ipynb",    "txt/time_indipendent"),
]


def set_bucket(nb, bucket):
    pat = re.compile(r"^BRAND_BUCKET\s*=\s*['\"][^'\"]*['\"]", re.M)
    for c in nb.cells:
        if c.cell_type == "code" and pat.search(c.source):
            c.source = pat.sub(f"BRAND_BUCKET = '{bucket}'", c.source)


def already_done(bucket, subfolder):
    d = DATA / bucket / "predictions" / subfolder
    return d.is_dir() and len(list(d.glob("*.zip"))) >= 6


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
    print(f"LoRA part3 batch: {len(BUCKETS)} buckets x {len(NOTEBOOKS)} = {total} runs")
    print(f"kernel={KERNEL}\n")
    for b in BUCKETS:
        for nb_name, subfolder in NOTEBOOKS:
            tag = f"{b}/{subfolder}"
            if already_done(b, subfolder):
                print(f"[skip] {tag}")
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
    print(f"done={done} skipped={skipped} failed={failed} / {total}")
    for f in failures:
        print("  FAIL:", f)
    print("=" * 60)


if __name__ == "__main__":
    main()
