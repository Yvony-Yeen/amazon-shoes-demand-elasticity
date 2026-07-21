"""
Converts all Google Colab notebooks in this project to run locally
(PyCharm Jupyter or nbconvert --execute).

Changes applied to every notebook:
  1. Delete cells that ONLY do drive mounting / colab imports
  2. Rewrite the config / path-setup cell with a robust _find_project_root() helper
  3. Remove /content/ intermediate-save patterns (save directly to target path)
  4. Strip --target=/content/pkgs and sys.path.insert('/content/pkgs') from pip cells

Run from project root:
    python _localize_notebooks.py
"""

import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ─── path-setup snippet injected at the top of every config cell ──────────
PATH_SETUP = '''\
from pathlib import Path

def _find_project_root() -> Path:
    """Walk up from cwd until a directory containing both
    'data_preparation/' and 'code/' is found.
    Works whether Jupyter/PyCharm sets cwd to project root or notebook dir."""
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / 'data_preparation').exists() and (p / 'code').exists():
            return p
    raise RuntimeError(
        'Cannot find project root.\\n'
        'Expected a directory containing both \\"data_preparation/\\" and \\"code/\\".\\n'
        f'Searched from: {Path.cwd()}'
    )

PROJECT_ROOT = _find_project_root()
DATA_ROOT    = PROJECT_ROOT / 'data'
DATA_DIR     = DATA_ROOT / 'amzn_shoes_monthly_diffs_ffill_fixed_splits'
IMG_DIR      = DATA_ROOT / 'images'
PRED_DIR     = DATA_ROOT / 'predictions'
KEY_DIR      = PROJECT_ROOT / 'code'
'''

# ─── helpers ──────────────────────────────────────────────────────────────

def src(cell):
    """Return cell source as a single string."""
    s = cell.get('source', '')
    return s if isinstance(s, str) else ''.join(s)

def set_src(cell, text):
    cell['source'] = text

def is_colab_only(cell):
    """True for cells whose ONLY job is to import / mount Google Drive."""
    s = src(cell).strip()
    if cell['cell_type'] != 'code':
        return False
    colab_lines = {
        l.strip() for l in s.splitlines()
        if l.strip() and not l.strip().startswith('#')
    }
    # Must consist exclusively of colab-drive lines (and optional print)
    non_colab = [
        l for l in colab_lines
        if not any(kw in l for kw in [
            'from google.colab', 'drive.mount', "print('✅ Drive mounted')",
            'print("✅ Drive mounted")',
        ])
    ]
    return len(non_colab) == 0 and len(colab_lines) > 0

def is_mount_drive_header(cell):
    """True for markdown cells that are just '## ① Mount Drive'."""
    if cell['cell_type'] != 'markdown':
        return False
    s = src(cell).strip()
    return bool(re.match(r'^##\s*[①②]?\s*Mount Drive\s*$', s, re.I))

def rewrite_config(s, nb_path):
    """
    Replace the Colab ROOT / SPLIT_DIR / KEY_DIR / etc. assignments with
    the portable PATH_SETUP block, keeping the rest of the cell unchanged.
    """
    lines = s.splitlines(keepends=True)
    out = []
    path_block_inserted = False

    # Lines we want to DROP from the original config
    drop_patterns = [
        r'from google\.colab import drive',
        r'drive\.mount\(',
        r"ROOT\s*=\s*['\"].*demand_modeling",
        r"ROOT\s*=\s*['\"].*Amazon Project",
        r"ROOT\s*=\s*['\"].*Northeastern",
        r"INPUT_FILE\s*=\s*['\"].*/content/drive/",
        r"DATA_DIR\s*=\s*ROOT\s*\+",
        r"DATA_DIR\s*=\s*['\"].*/content/",
        r"SPLIT_DIR\s*=\s*",
        r"IMG_DIR\s*=\s*(ROOT|DATA_DIR)\s*\+",
        r"PRED_DIR\s*=\s*",
        r"KEY_DIR\s*=\s*ROOT\s*\+",
        r"CLEAN_FILE\s*=\s*DATA_DIR\s*\+",
        r"OUT_DIR\s*=\s*ROOT\s*\+",
        r"OUT_DIR\s*=\s*['\"].*/content/",
        r"os\.makedirs\(KEY_DIR",
        r"os\.makedirs\(ROOT\s*\+\s*['\"]output",
    ]
    drop_re = re.compile('|'.join(drop_patterns))

    # Also drop standalone os.makedirs that still reference the old paths
    for line in lines:
        stripped = line.strip()
        if drop_re.search(stripped):
            # Insert the path block before the first dropped line
            if not path_block_inserted:
                out.append(PATH_SETUP + '\n')
                path_block_inserted = True
            continue  # drop this line
        out.append(line)

    if not path_block_inserted:
        # Prepend if no matches found (rare)
        out.insert(0, PATH_SETUP + '\n')

    # Fix INPUT_FILE for 00_part1 (points to raw combined parquet)
    result = ''.join(out)
    result = re.sub(
        r"INPUT_FILE\s*=\s*['\"][^'\"]+amazon_shoes_8_combined\.parquet['\"]",
        "INPUT_FILE = DATA_ROOT / 'amazon_shoes_8_combined.parquet'",
        result,
    )

    # Fix remaining string-concat path references → Path division
    result = re.sub(r"ROOT\s*\+\s*['\"]data/images/['\"]", "DATA_ROOT / 'images'", result)
    result = re.sub(r"ROOT\s*\+\s*['\"]data/predictions/['\"]", "DATA_ROOT / 'predictions'", result)
    result = re.sub(r"ROOT\s*\+\s*['\"]code/['\"]", "PROJECT_ROOT / 'code'", result)
    result = re.sub(r"ROOT\s*\+\s*['\"]output/['\"]", "PROJECT_ROOT / 'output'", result)
    result = re.sub(r"ROOT\s*\+\s*['\"]data/['\"]", "DATA_ROOT / ", result)

    # Long image-parquet OUT_DIR name
    result = re.sub(
        r"DATA_DIR\s*\+\s*['\"]amzn_shoes_monthly_avg_long_ffill_28_7_images/['\"]",
        "DATA_ROOT / 'amzn_shoes_monthly_avg_long_ffill_28_7_images'",
        result,
    )
    result = re.sub(
        r"OUT_DIR\s*=\s*DATA_ROOT\s*/\s*",
        "OUT_DIR = DATA_ROOT / ",
        result,
    )

    # Ensure os.makedirs calls use str() for compatibility
    result = re.sub(
        r"os\.makedirs\(([A-Z_]+)\s*,",
        lambda m: f"os.makedirs({m.group(1)}, ",
        result,
    )

    return result


def fix_pip_cell(s):
    """Strip /content/pkgs isolation logic; keep pip installs and gensim bypass."""
    lines = s.splitlines(keepends=True)
    out = []
    drop = [
        r"os\.makedirs\(['\"]\/content\/pkgs",
        r"--target=\/content\/pkgs",
        r"sys\.path\.insert\(0.*\/content\/pkgs",
        r"if\s+['\"]\/content\/pkgs['\"]",
        r"if\s*'\/content\/pkgs'\s*not\s*in\s*sys\.path",
    ]
    drop_re = re.compile('|'.join(drop))
    for line in lines:
        if drop_re.search(line):
            continue
        # Remove the --target flag from pip install lines inline
        line = re.sub(r'\s*--target=\S+', '', line)
        out.append(line)
    return ''.join(out)


def fix_content_saves(s):
    """
    Convert Colab two-step save pattern to single direct save:
      LOCAL_X = '/content/something'
      df.to_parquet(LOCAL_X, ...)
      shutil.copy(LOCAL_X, DEST)
    →
      DEST.parent.mkdir(parents=True, exist_ok=True)
      df.to_parquet(DEST, ...)
    """
    # Find LOCAL_X = '/content/...' assignments
    local_vars = {}  # var_name → dest_name guess from shutil.copy
    lines = s.splitlines(keepends=True)

    # Pass 1: collect LOCAL_X definitions and their copy destinations
    copy_map = {}  # LOCAL_VAR → DEST expression from shutil.copy
    for line in lines:
        m = re.match(r"\s*(\w+)\s*=\s*['\"]\/content\/([^'\"]+)['\"]", line)
        if m:
            local_vars[m.group(1)] = m.group(2)
        m2 = re.match(r"\s*shutil\.copy\((\w+)\s*,\s*(.+)\)", line.rstrip())
        if m2:
            copy_map[m2.group(1)] = m2.group(2).strip().rstrip(')')

    if not local_vars:
        return s  # nothing to fix

    out = []
    mkdir_inserted = set()
    for line in lines:
        stripped = line.strip()

        # Drop LOCAL_X = '/content/...' assignments
        m = re.match(r"\s*(\w+)\s*=\s*['\"]\/content\/[^'\"]+['\"]", line)
        if m and m.group(1) in local_vars:
            continue

        # Replace shutil.copy(LOCAL_X, DEST) → skip (we already saved directly)
        m2 = re.match(r"\s*shutil\.copy\((\w+)\s*,\s*(.+)\)", stripped)
        if m2 and m2.group(1) in local_vars:
            continue

        # Replace .to_parquet(LOCAL_X, ...) with .to_parquet(DEST, ...)
        # Replace .to_csv(LOCAL_X, ...) with .to_csv(DEST, ...)
        for var, dest_expr in copy_map.items():
            if var in line and var in local_vars:
                if dest_expr not in mkdir_inserted:
                    out.append(f'    {dest_expr}.parent.mkdir(parents=True, exist_ok=True)\n')
                    mkdir_inserted.add(dest_expr)
                line = line.replace(var, dest_expr)

        # Also fix os.path.getsize(LOCAL_X) → os.path.getsize(DEST)
        for var, dest_expr in copy_map.items():
            if var in line and var in local_vars:
                line = line.replace(var, dest_expr)

        out.append(line)

    return ''.join(out)


def fix_path_string_ops(s):
    """
    Replace string-concatenation path ops with Path division:
      SPLIT_DIR + 'train-...'  → DATA_DIR / 'train-...'
      IMG_DIR + f'{asin}.jpg'  → IMG_DIR / f'{asin}.jpg'
      PRED_DIR + subfolder + '/' + fname  → PRED_DIR / subfolder / fname
      os.path.exists(IMG_DIR + ...)  → (IMG_DIR / ...).exists()
      os.makedirs(PRED_DIR + subdir, ...)  → (PRED_DIR / subdir).mkdir(...)
    """
    # data_dir string concatenation
    s = re.sub(r"DATA_DIR\s*\+\s*'([^']+)'", r"DATA_DIR / '\1'", s)
    s = re.sub(r'DATA_DIR\s*\+\s*"([^"]+)"', r'DATA_DIR / "\1"', s)
    s = re.sub(r"SPLIT_DIR\s*\+\s*'([^']+)'", r"DATA_DIR / '\1'", s)
    s = re.sub(r'SPLIT_DIR\s*\+\s*"([^"]+)"', r'DATA_DIR / "\1"', s)
    s = re.sub(r"IMG_DIR\s*\+\s*f'([^']+)'", r"IMG_DIR / f'\1'", s)
    s = re.sub(r'IMG_DIR\s*\+\s*f"([^"]+)"', r'IMG_DIR / f"\1"', s)
    s = re.sub(r"IMG_DIR\s*\+\s*'([^']+)'", r"IMG_DIR / '\1'", s)
    s = re.sub(r"OUT_DIR\s*\+\s*'([^']+)'", r"OUT_DIR / '\1'", s)
    s = re.sub(r'OUT_DIR\s*\+\s*"([^"]+)"', r'OUT_DIR / "\1"', s)
    s = re.sub(r"KEY_DIR\s*\+\s*'([^']+)'", r"KEY_DIR / '\1'", s)
    s = re.sub(r'KEY_DIR\s*\+\s*"([^"]+)"', r'KEY_DIR / "\1"', s)
    s = re.sub(r"PRED_DIR\s*\+\s*'([^']+)'", r"PRED_DIR / '\1'", s)
    s = re.sub(r'PRED_DIR\s*\+\s*"([^"]+)"', r'PRED_DIR / "\1"', s)

    # os.path.exists(PATH + ...) → (PATH / ...).exists()
    s = re.sub(
        r'os\.path\.exists\(IMG_DIR\s*/\s*f\'([^\']+)\'\)',
        r"(IMG_DIR / f'\1').exists()",
        s,
    )
    s = re.sub(
        r'os\.path\.exists\(IMG_DIR\s*\+\s*f\'([^\']+)\'\)',
        r"(IMG_DIR / f'\1').exists()",
        s,
    )
    s = re.sub(
        r'os\.path\.exists\(IMG_DIR\s*\+\s*f"([^"]+)"\)',
        r'(IMG_DIR / f"\1").exists()',
        s,
    )

    # os.path.getsize(path) still works with Path objects — no change needed

    # str(Path) where needed: pd.read_parquet / to_parquet / to_csv accept Path directly
    # os.makedirs(Path, ...) also accepts Path directly

    return s


def process_notebook(nb_path: Path):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []
    for cell in nb['cells']:
        s = src(cell)

        # ── 1. Delete drive-mount-only cells ─────────────────────────────
        if is_colab_only(cell) or is_mount_drive_header(cell):
            print(f'  DELETE: {cell["cell_type"]} | {s.strip()[:60]}')
            continue

        # ── 2. Rewrite config cell ────────────────────────────────────────
        if cell['cell_type'] == 'code' and (
            "ROOT" in s and (
                "demand_modeling_data_women_8_v2" in s
                or "/content/drive/" in s
                or "Northeastern University" in s
            )
        ):
            new_s = rewrite_config(s, nb_path)
            new_s = fix_path_string_ops(new_s)
            set_src(cell, new_s)
            print(f'  REWRITE CONFIG: {cell.get("id","?")}')

        # ── 3. Fix pip install cell ───────────────────────────────────────
        elif cell['cell_type'] == 'code' and '/content/pkgs' in s:
            new_s = fix_pip_cell(s)
            set_src(cell, new_s)
            print(f'  FIX PIP: {cell.get("id","?")}')

        # ── 4. Fix intermediate /content/ saves ───────────────────────────
        elif cell['cell_type'] == 'code' and '/content/' in s and (
            'shutil.copy' in s or "= '/content/" in s or '= "/content/' in s
        ):
            new_s = fix_content_saves(s)
            new_s = fix_path_string_ops(new_s)
            set_src(cell, new_s)
            print(f'  FIX SAVES: {cell.get("id","?")}')

        # ── 5. Fix remaining string-path ops in all other code cells ──────
        elif cell['cell_type'] == 'code' and any(
            pat in s for pat in ['IMG_DIR +', 'PRED_DIR +', 'DATA_DIR +',
                                  'SPLIT_DIR +', 'KEY_DIR +', 'OUT_DIR +']
        ):
            set_src(cell, fix_path_string_ops(s))

        new_cells.append(cell)

    nb['cells'] = new_cells

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f'  → saved {nb_path.name}')


# ─── main ──────────────────────────────────────────────────────────────────

notebooks = sorted(PROJECT_ROOT.rglob('*.ipynb'))
# Skip the localize script itself if accidentally named .ipynb
notebooks = [p for p in notebooks if '_localize' not in p.name]

print(f'Found {len(notebooks)} notebooks under {PROJECT_ROOT}\n')
errors = []
for nb_path in notebooks:
    print(f'\n[{nb_path.relative_to(PROJECT_ROOT)}]')
    try:
        process_notebook(nb_path)
    except Exception as e:
        print(f'  ERROR: {e}')
        errors.append((nb_path, e))

print(f'\n{"="*60}')
if errors:
    print(f'ERRORS in {len(errors)} notebooks:')
    for p, e in errors:
        print(f'  {p.name}: {e}')
else:
    print('All notebooks processed successfully.')
