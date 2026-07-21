#!/usr/bin/env python3
"""
run_code_pipeline.py  --  Phase 3 (code/) analysis driver for the
Amazon Shoes demand-analysis pipeline.

WHAT THIS DOES
  Runs the code/ notebooks (Phase 3 in pipeline_documentation) for one or
  more brand buckets, in dependency order, end to end. It is a thin driver:
  it does NOT change the analysis logic. For each bucket it

    1. regenerates code/utils/paths_config.yaml from that bucket's actual
       prediction .zip files (the committed yaml is a stale single-bucket
       leftover and must be rebuilt per bucket -- filenames carry per-run
       dates/epochs/losses that differ across buckets), then
    2. injects BRAND_BUCKET into a *temporary copy* of each notebook (source
       notebooks are left untouched) and executes it, writing the executed
       copy to data/<bucket>/_run_logs/ as a log.

  estimate_vartheta_shoes runs ONCE (it is category-level / shared), before
  the per-bucket 04_* notebooks.

WHAT IT WILL SKIP (and tell you about)
  * Buckets whose predictions/ are incomplete (need all 4 variants:
    txt/txtimg x lag1/time_indipendent). Finish part3a-d + download first.
  * 02_cluster_centroid* notebooks when the bucket has no image parquet
    (data/<bucket>/amzn_shoes_monthly_avg_long_ffill_28_7_images/). Those
    are qualitative collages; run part5 + download the image parquet to
    enable them, or leave skipped.

USAGE
  python run_code_pipeline.py                  # all complete buckets, default steps
  python run_code_pipeline.py --buckets women_other,men_top2_nike
  python run_code_pipeline.py --steps 01_1,01_2,04_evaluation_final_all
  python run_code_pipeline.py --skip-vartheta  # vartheta already estimated
  python run_code_pipeline.py --dry-run        # print plan, run nothing
  python run_code_pipeline.py --continue-on-error
  python run_code_pipeline.py --list           # list step keys and exit

Requires: jupyter (nbformat, nbconvert) -- already present since the
notebooks run in Jupyter. No GPU; all code/ steps are CPU.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "data"
YAML_PATH = CODE_DIR / "utils" / "paths_config.yaml"

ALL_BUCKETS = [
    "women_top1_dream_pairs",
    "women_top2_idifu",
    "women_other",
    "men_top1_bruno_marc",
    "men_top2_nike",
    "men_other",
]

# Per-bucket steps, in dependency order. (step_key, notebook_filename, needs_yaml)
PER_BUCKET_STEPS: list[tuple[str, str, bool]] = [
    ("01_1", "01_1_create_dataset_txt_img.ipynb", True),
    ("01_2", "01_2_create_dataset_txt.ipynb", True),
    ("02", "02_cluster_centroid_products.ipynb", False),
    ("02_random20", "02_cluster_centroid_products_random20.ipynb", False),
    ("03_1", "03_1_predictive_performance_txt_img.ipynb", False),
    ("03_2", "03_2_predictive_performance_txt.ipynb", False),
    ("04_evaluation_final_all", "04_evaluation_final_all.ipynb", False),
    ("04_evaluation_iv_final", "04_evaluation_iv_final.ipynb", False),
    ("04_scenario1", "04_scenario1_flats_only.ipynb", False),
    ("04_scenario2", "04_scenario2_excl_top500.ipynb", False),
    ("04_scenario3", "04_scenario3_excl_top500_flats.ipynb", False),
]

# Steps that need the image parquet (skipped if it is missing for the bucket)
IMAGE_PARQUET_STEPS = {"02", "02_random20"}
IMAGE_PARQUET_DIR = "amzn_shoes_monthly_avg_long_ffill_28_7_images"

VARTHETA_NB = "estimate_vartheta_shoes.ipynb"

# Maps on-disk prediction subfolder -> (modality, lag-key used in the yaml).
# Note the on-disk folder is misspelled "time_indipendent"; the yaml key is the
# correctly-spelled "time_independent" that load_pred_and_emb() looks up.
PRED_SUBFOLDERS = [
    ("txt", "time_indipendent", "txt", "time_independent"),
    ("txt", "lag1", "txt", "lag1"),
    ("txtimg", "time_indipendent", "txtimg", "time_independent"),
    ("txtimg", "lag1", "txtimg", "lag1"),
]


# ---------------------------------------------------------------------------
# Prediction discovery / yaml generation
# ---------------------------------------------------------------------------


def _classify(fname: str):
    """Return ('level', dim, split) | ('diff', None, split) | None for a zip."""
    if fname.startswith("pred_diff"):
        split = "train" if fname.endswith("_train.zip") else "val"
        return ("diff", None, split)
    m = re.match(r"(train|val)_pred_model.*dim=(\d+)_", fname)
    if m:
        return ("level", int(m.group(2)), m.group(1))
    return None


def discover_predictions(bucket: str) -> dict:
    """Glob a bucket's predictions/ and resolve the slots load_pred_and_emb needs.

    Returns a nested dict ready to dump as paths_config.yaml, with ABSOLUTE
    paths. Raises with a clear message if a required slot is missing or
    ambiguous (duplicate).
    """
    pred_root = DATA_DIR / bucket / "predictions"
    cfg: dict = {"txt": {}, "txtimg": {}, "diff_txt": {}, "diff_txtimg": {}}
    problems: list[str] = []

    for modality, disk_lag, _mod_key, yaml_lag in PRED_SUBFOLDERS:
        d = pred_root / modality / disk_lag
        slots: dict[tuple, list[Path]] = {}
        if d.is_dir():
            for f in sorted(d.glob("*.zip")):
                c = _classify(f.name)
                if c is None:
                    continue
                slots.setdefault(c, []).append(f)

        # level dims 128/256 (train+val)
        cfg[modality].setdefault(yaml_lag, {})
        for dim in (128, 256):
            cfg[modality][yaml_lag][dim] = {}
            for split in ("train", "val"):
                files = slots.get(("level", dim, split), [])
                if len(files) == 1:
                    cfg[modality][yaml_lag][dim][split] = str(files[0])
                elif len(files) == 0:
                    problems.append(f"missing {modality}/{disk_lag} level dim={dim} {split}")
                else:
                    problems.append(
                        f"DUPLICATE {modality}/{disk_lag} level dim={dim} {split}: "
                        + ", ".join(f.name for f in files)
                    )

        # diff (train+val)
        diff_key = "diff_txt" if modality == "txt" else "diff_txtimg"
        cfg[diff_key].setdefault(yaml_lag, {})
        for split in ("train", "val"):
            files = slots.get(("diff", None, split), [])
            if len(files) == 1:
                cfg[diff_key][yaml_lag][split] = str(files[0])
            elif len(files) == 0:
                problems.append(f"missing {modality}/{disk_lag} diff {split}")
            else:
                problems.append(
                    f"DUPLICATE {modality}/{disk_lag} diff {split}: "
                    + ", ".join(f.name for f in files)
                )

    return cfg, problems


def write_yaml(cfg: dict) -> None:
    YAML_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))


# ---------------------------------------------------------------------------
# Notebook execution
# ---------------------------------------------------------------------------


def set_bucket_in_notebook(nb, bucket: str) -> bool:
    """Rewrite the BRAND_BUCKET line in the bucket-config cell. Returns True
    if a line was changed (False for shared notebooks like vartheta)."""
    changed = False
    pat = re.compile(r"^BRAND_BUCKET\s*=\s*['\"][^'\"]*['\"]", re.M)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "BRAND_BUCKET" in cell.source and pat.search(cell.source):
            cell.source = pat.sub(f"BRAND_BUCKET = '{bucket}'", cell.source)
            changed = True
    return changed


def set_txt_only_in_notebook(nb) -> bool:
    """Flip `txt_only = False` -> `txt_only = True` so the notebook loads the
    txt-only analysis dataset, and redirect its OUTPUT_DIR to a `<name>_txt`
    folder so the txt run does not clobber the txtimg results. Returns True if
    anything was changed."""
    changed = False
    txt_pat = re.compile(r"^txt_only\s*=\s*False\b", re.M)
    # OUTPUT_DIR = DATA_ROOT / 'output' / '<name>'  ->  '<name>_txt'
    out_pat = re.compile(r"(OUTPUT_DIR\s*=\s*DATA_ROOT\s*/\s*'output'\s*/\s*')([^']+)'")
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        new = cell.source
        if txt_pat.search(new):
            new = txt_pat.sub("txt_only = True", new)
        if out_pat.search(new):
            new = out_pat.sub(lambda m: f"{m.group(1)}{m.group(2)}_txt'", new)
        if new != cell.source:
            cell.source = new
            changed = True
    return changed


def run_notebook(nb_name: str, bucket: str | None, timeout: int, kernel: str,
                 txt_only: bool = False, allow_errors: bool = False) -> None:
    """Execute a notebook copy with BRAND_BUCKET injected, CWD=code/, and save
    the executed copy under data/<bucket>/_run_logs/ (or code/_run_logs for
    shared notebooks). With txt_only=True the notebook is switched to the
    txt-only dataset and its output/log are suffixed `_txt` to keep both
    modalities side by side."""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    src = CODE_DIR / nb_name
    nb = nbformat.read(src, as_version=4)
    if bucket is not None:
        set_bucket_in_notebook(nb, bucket)
    # Only flip the modality switch on the 04 evaluation notebooks; 01_1/03_1 also
    # contain `txt_only = False` but there it selects the txtimg pipeline itself.
    if txt_only and nb_name.startswith("04"):
        set_txt_only_in_notebook(nb)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel, allow_errors=allow_errors)
    # CWD must be code/ so `from utils...` and "utils/paths_config.yaml" resolve.
    ep.preprocess(nb, {"metadata": {"path": str(CODE_DIR)}})

    log_root = (DATA_DIR / bucket / "_run_logs") if bucket else (CODE_DIR / "_run_logs")
    log_root.mkdir(parents=True, exist_ok=True)
    log_name = nb_name.replace(".ipynb", "_txt.ipynb") if txt_only else nb_name
    nbformat.write(nb, log_root / log_name)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def bucket_predictions_complete(bucket: str) -> tuple[bool, list[str]]:
    _cfg, problems = discover_predictions(bucket)
    return (len(problems) == 0), problems


def bucket_has_image_parquet(bucket: str) -> bool:
    d = DATA_DIR / bucket / IMAGE_PARQUET_DIR
    return d.is_dir() and any(d.glob("*.parquet"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def default_complete_buckets() -> list[str]:
    return [b for b in ALL_BUCKETS if bucket_predictions_complete(b)[0]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run code/ Phase-3 notebooks per bucket.")
    ap.add_argument("--buckets", default="", help="comma list; default = all complete buckets")
    ap.add_argument("--steps", default="", help="comma list of step keys; default = all")
    ap.add_argument("--skip-vartheta", action="store_true", help="skip estimate_vartheta_shoes")
    ap.add_argument("--timeout", type=int, default=3600, help="per-notebook timeout (s)")
    ap.add_argument("--kernel", default="shoes-venv",
                    help="Jupyter kernel name to execute with (must point at the "
                         "project venv that has code/requirements.txt installed)")
    ap.add_argument("--force", action="store_true",
                    help="run the requested steps even if a bucket's predictions are "
                         "incomplete (e.g. txtimg-only steps 03_1/04 when txt/lag1 is missing)")
    ap.add_argument("--allow-errors", action="store_true",
                    help="keep executing a notebook past a failing cell and still save the "
                         "log (e.g. small buckets that hit the missing-cluster KeyError in 04 "
                         "after the ATE elasticity is already computed)")
    ap.add_argument("--txt-only", action="store_true",
                    help="Run 04 notebooks on the txt-only dataset (txt_only=True), "
                         "writing to output/<name>_txt/ and _run_logs/<nb>_txt.ipynb "
                         "so txt and txtimg results sit side by side. Requires 01_2 "
                         "to have produced dataset_txt_only_True_*.")
    ap.add_argument("--continue-on-error", action="store_true", help="keep going after a failure")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, execute nothing")
    ap.add_argument("--list", action="store_true", help="list step keys and exit")
    args = ap.parse_args()

    if args.list:
        print("Step keys (per bucket, in order):")
        for key, nb, _ in PER_BUCKET_STEPS:
            print(f"  {key:28s} {nb}")
        print(f"\nShared (once): vartheta -> {VARTHETA_NB}")
        return 0

    # Resolve buckets
    if args.buckets:
        buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    else:
        buckets = default_complete_buckets()

    # Resolve steps
    step_keys = [s.strip() for s in args.steps.split(",") if s.strip()] if args.steps else None
    steps = [s for s in PER_BUCKET_STEPS if step_keys is None or s[0] in step_keys]
    if step_keys:
        unknown = set(step_keys) - {s[0] for s in PER_BUCKET_STEPS}
        if unknown:
            print(f"ERROR: unknown step keys: {sorted(unknown)}", file=sys.stderr)
            return 2

    print("=" * 70)
    print("CODE PIPELINE PLAN")
    print("=" * 70)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Buckets      : {buckets}")
    print(f"Steps        : {[s[0] for s in steps]}")
    print(f"Vartheta     : {'skip' if args.skip_vartheta else 'run once'}")
    print()

    # Pre-flight: report bucket / step status
    runnable: dict[str, list[tuple[str, str, bool]]] = {}
    for b in buckets:
        ok, problems = bucket_predictions_complete(b)
        if not ok and not args.force:
            print(f"[SKIP bucket] {b}: predictions incomplete (use --force to run anyway) ->")
            for p in problems:
                print(f"      - {p}")
            continue
        if not ok and args.force:
            print(f"[FORCE] {b}: predictions incomplete, running requested steps anyway:")
            for p in problems:
                print(f"      - {p}")
        bsteps = []
        for key, nb, needs_yaml in steps:
            if key in IMAGE_PARQUET_STEPS and not bucket_has_image_parquet(b):
                print(f"[SKIP step ] {b}/{key}: no image parquet ({IMAGE_PARQUET_DIR}/)")
                continue
            bsteps.append((key, nb, needs_yaml))
        runnable[b] = bsteps

    if not runnable:
        print("\nNothing runnable. Finish part3a-d + download for at least one bucket.")
        return 1

    if args.dry_run:
        print("\n--dry-run: would execute:")
        if not args.skip_vartheta:
            print(f"  [once] {VARTHETA_NB}")
        for b, bsteps in runnable.items():
            for key, nb, _ in bsteps:
                print(f"  [{b}] {key} -> {nb}")
        return 0

    failures: list[str] = []

    # Shared vartheta (once, before any 04_*)
    if not args.skip_vartheta:
        print("\n" + "-" * 70)
        print(f"[once] vartheta -> {VARTHETA_NB}")
        t0 = time.time()
        try:
            run_notebook(VARTHETA_NB, None, args.timeout, args.kernel,
                         allow_errors=args.allow_errors)
            print(f"  done in {time.time()-t0:.0f}s")
        except Exception as e:  # noqa: BLE001
            failures.append(f"vartheta: {e}")
            print(f"  FAILED: {e}")
            if not args.continue_on_error:
                _report(failures)
                return 1

    # Per-bucket steps
    for b, bsteps in runnable.items():
        yaml_done = False
        for key, nb, needs_yaml in bsteps:
            if needs_yaml and not yaml_done:
                cfg, problems = discover_predictions(b)
                if problems and not args.force:
                    failures.append(f"{b}: yaml gen: {problems}")
                    print(f"[{b}] yaml gen FAILED: {problems}")
                    if not args.continue_on_error:
                        _report(failures)
                        return 1
                    continue
                if problems and args.force:
                    print(f"[{b}] yaml gen: partial (missing some variants, writing "
                          f"what's available) -> {problems}")
                write_yaml(cfg)
                yaml_done = True
                print(f"\n[{b}] regenerated {YAML_PATH.relative_to(PROJECT_ROOT)}")

            print("-" * 70)
            print(f"[{b}] {key} -> {nb}")
            t0 = time.time()
            try:
                run_notebook(nb, b, args.timeout, args.kernel, txt_only=args.txt_only,
                             allow_errors=args.allow_errors)
                print(f"  done in {time.time()-t0:.0f}s "
                      f"(log: data/{b}/_run_logs/{nb})")
            except Exception as e:  # noqa: BLE001
                failures.append(f"{b}/{key}: {e}")
                print(f"  FAILED: {e}")
                if not args.continue_on_error:
                    _report(failures)
                    return 1

    _report(failures)
    return 1 if failures else 0


def _report(failures: list[str]) -> None:
    print("\n" + "=" * 70)
    if failures:
        print(f"FINISHED WITH {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("ALL STEPS SUCCEEDED")
    print("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())