# Amazon Shoes Demand Analysis

Replication of Bach et al. (2025) *"Adventures in Demand Analysis Using AI"* applied to **Amazon Shoes (Size 8)**, Jan 2025 – Mar 2026, split by **gender × brand tier**.

**Team:** Billy, Yuhui, Ian, Jyoti| Advisor: Guy, Ruidong Ma

---

## 0. What changed since the original paper / first version

This pipeline differs from the original Bach et al. paper and our first iteration on two important axes:

1. **Brand-bucket isolation.** Instead of pooling all shoes together, the raw dataset is first split into **6 independent buckets** (gender × brand tier). Every embedding / training step runs separately per bucket — mixing categories pollutes embedding weights and produces uninterpretable elasticities for any single subgroup.

   ```
   Women × top1  →  women_top1_dream_pairs
   Women × top2  →  women_top2_idifu
   Women × other →  women_other
   Men   × top1  →  men_top1_bruno_marc
   Men   × top2  →  men_top2_nike
   Men   × other →  men_other
   ```

2. **Stratified split happens upstream, not after embedding.** The original `00_part2b_embed_cluster_split.ipynb` (RoBERTa + BEiT + SAINT embedding → PCA → KMeans → cluster-stratified split) has been **deleted**. The new `brand_split/01_stratified_split.ipynb` does a cheap stratified 50/50 split on `subcat_aggregated × SALES_RANK quartile` per bucket — no GPU, no embedding model needed. The assignment is decided **before** any data_preparation step runs, then materialized into per-bucket `train/val` parquets at the tail of `00_part1`.

The shoes-specific Pareto shape parameter ϑ (used to convert rank-elasticity to demand-elasticity) is estimated **once** on the full dataset and **shared across all 6 buckets** — ϑ is a category-level structural parameter, not bucket-specific.

---

## 1. Setup

```bash
# Clone & enter
git clone <repo-url>
cd Amazon_demand_analysis_seperate

# Python env (3.10+ recommended)
pip install -r code/requirements.txt
# part2 also needs:  pip install pyarrow pandas requests Pillow tqdm
# part3a/b/c/d also need: pip install torch transformers pytorch-widedeep torchmetrics
```

**Input file required at `data/amazon_shoes_8_combined.parquet`** (raw Keepa export, 1.0M rows × 26 cols, 20 MB). Not in the repo — ask Billy.

---

## 2. Pipeline overview

All paths below assume project root is the repo root. Every notebook is parameterized by `BRAND_BUCKET` (except where noted).

| # | Notebook | Compute | Per-bucket? | Produces |
|---|---|---|---|---|
| **brand_split (run once total)** |||||
| 0 | `brand_split/00_brand_filter.ipynb` | CPU | No (writes all 6) | `data/<bucket>/amazon_shoes_8_combined.parquet` × 6 |
| 1 | `brand_split/01_stratified_split.ipynb` | CPU | No (writes all 6) | `data/<bucket>/keys/asin_split.csv` × 6 |
| **data_preparation (run for EACH of 6 buckets)** |||||
| 2 | `data_preparation/00_part1_data_preparation.ipynb` | CPU | Yes | `all_products_clean.parquet`, `train/validation parquets`, `main_*_keys.csv` |
| 3 | `data_preparation/00_part2_download_images.ipynb` | CPU + network | Yes (but images shared) | `data/images/{ASIN}.jpg` (shared across buckets) |
| 4 | `data_preparation/00_part5_create_image_parquet.ipynb` | CPU | Yes | 16-shard image+feature parquet under `data/<bucket>/amzn_shoes_monthly_avg_long_ffill_28_7_images/` |
| 5a | `data_preparation/00_part3a_train_txtimg_lag1.ipynb` | **GPU** | Yes | Multimodal txt+img embedding zips → `data/<bucket>/predictions/txtimg/lag1/` |
| 5b | `data_preparation/00_part3b_train_txtimg_time_ind.ipynb` | **GPU** | Yes | `data/<bucket>/predictions/txtimg/time_indipendent/` |
| 5c | `data_preparation/00_part3c_train_txt_lag1.ipynb` | **GPU** | Yes | `data/<bucket>/predictions/txt/lag1/` |
| 5d | `data_preparation/00_part3d_train_txt_time_ind.ipynb` | **GPU** | Yes | `data/<bucket>/predictions/txt/time_indipendent/` |
| 6 | `data_preparation/00_part4_verify.ipynb` | CPU | Yes (optional) | Sanity check report |
| **analysis — code/ (run for EACH of 6 buckets)** |||||
| 7a | `code/01_1_create_dataset_txt_img.ipynb` | CPU | Yes | `data/<bucket>/analysis_datasets/dataset_*_train.zip` + `_val.zip` |
| 7b | `code/01_2_create_dataset_txt.ipynb` | CPU | Yes | Same shape, text-only variant |
| 8 | `code/02_cluster_centroid_products.ipynb` | CPU | Yes | `data/<bucket>/output/cluster_centroid/*.png` |
| 8' | `code/02_cluster_centroid_products_random20.ipynb` | CPU | Yes (optional comparison) | `data/<bucket>/output/cluster_centroid_random20/*.png` |
| 9a | `code/03_1_predictive_performance_txt_img.ipynb` | CPU | Yes | `data/<bucket>/output/predictive_performance_txt_img/` |
| 9b | `code/03_2_predictive_performance_txt.ipynb` | CPU | Yes | `data/<bucket>/output/predictive_performance_txt/` |
| 10a | `code/04_evaluation_final_all.ipynb` | CPU | Yes | Primary DoubleML elasticity + ϑ conversion → `data/<bucket>/output/evaluation_final_all/` |
| 10b | `code/04_evaluation_iv_final.ipynb` | CPU | Yes (alternative method) | IV elasticity → `data/<bucket>/output/evaluation_iv/` |
| 10c | `code/04_scenario1_flats_only.ipynb` | CPU | Yes (subsample) | `data/<bucket>/output/scenario1_flats/` |
| 10d | `code/04_scenario2_excl_top500.ipynb` | CPU | Yes (subsample) | `data/<bucket>/output/scenario2_excl_top500/` |
| 10e | `code/04_scenario3_excl_top500_flats.ipynb` | CPU | Yes (subsample) | `data/<bucket>/output/scenario3_excl_top500_flats/` |
| **shared (run once total)** |||||
| 11 | `code/estimate_vartheta_shoes.ipynb` | CPU | **No** (ϑ is category-level) | `data/_shared/vartheta/vartheta_estimates_shoes.csv` |

**Only steps 5a–5d need a GPU.** Everything else runs on CPU.

---

## 3. How to run for each bucket

Every per-bucket notebook has a single line you edit, near the top:

```python
BRAND_BUCKET = 'women_top1_dream_pairs'   # ← change this
```

Valid values:
```
women_top1_dream_pairs   women_top2_idifu   women_other
men_top1_bruno_marc      men_top2_nike      men_other
```

### Standard run order for ONE bucket

1. Open the notebook
2. Edit `BRAND_BUCKET` to your target bucket
3. Run all cells
4. Verify outputs landed under `data/<bucket>/...` (paths in the table above)
5. Repeat for the next notebook in the pipeline

### Full pipeline — sequence to run

```
ONCE                          : brand_split/00 → brand_split/01
PER BUCKET (6 ×)              : part1 → part2 → part5 → part3a → part3b → part3c → part3d → part4
                                → code/01_1 → code/01_2 → code/02 → code/03_1 → code/03_2
                                → code/04_evaluation_final_all (and any scenario/IV variants you want)
ONCE                          : code/estimate_vartheta_shoes
```

### Useful optimizations

- **part2 (image download):** images live in the shared `data/images/` folder keyed by ASIN. If you run `women_other` (the biggest, ~4200 ASINs) first, subsequent buckets will skip already-downloaded images (the notebook checks `already_done` in cell 8). Total expected: ~6,500 unique JPEGs across all 6 buckets.
- **GPU steps (part3a–d):** these are the only stages that need a GPU. 4 notebooks × 6 buckets = 24 GPU runs. They can be staged independently of the CPU-only stages.

---

## 4. Final folder structure (after a complete run of all 6 buckets)

```
Amazon_demand_analysis_seperate/
├── data/
│   ├── amazon_shoes_8_combined.parquet              ← raw input (from Keepa)
│   ├── images/                                      ← {ASIN}.jpg, shared across buckets (~6.5k JPEGs)
│   ├── _shared/
│   │   └── vartheta/
│   │       └── vartheta_estimates_shoes.csv         ← ϑ (shared, category-level)
│   └── <bucket>/                                    ← 6 of these
│       ├── amazon_shoes_8_combined.parquet          ← bucket-filtered raw
│       ├── amzn_shoes_monthly_diffs_ffill_fixed_splits/
│       │   ├── all_products_clean.parquet
│       │   ├── train-00000-of-00001.parquet
│       │   └── validation-00000-of-00001.parquet
│       ├── amzn_shoes_monthly_avg_long_ffill_28_7_images/
│       │   └── train-000{0-3}-of-00004-part00{0-3}.parquet   ← 16 image parquets
│       ├── keys/
│       │   ├── asin_split.csv                       ← from brand_split/01
│       │   ├── main_train_keys.csv                  ← from part1 (ASIN, date pairs for train)
│       │   └── main_val_keys.csv                    ← from part1
│       ├── predictions/                             ← from part3a-d (GPU)
│       │   ├── txtimg/{lag1,time_indipendent}/
│       │   └── txt/{lag1,time_indipendent}/
│       ├── analysis_datasets/                       ← from code/01_*
│       │   ├── dataset_txt_only_True_*_train.zip
│       │   └── dataset_txt_only_True_*_val.zip
│       └── output/                                  ← from code/02-04
│           ├── cluster_centroid/
│           ├── cluster_centroid_random20/
│           ├── predictive_performance_txt_img/
│           ├── predictive_performance_txt/
│           ├── evaluation_final_all/                ← primary DoubleML results
│           ├── evaluation_iv/
│           ├── scenario1_flats/
│           ├── scenario2_excl_top500/
│           └── scenario3_excl_top500_flats/
├── brand_split/
│   ├── 00_brand_filter.ipynb
│   └── 01_stratified_split.ipynb
├── data_preparation/
│   ├── 00_part1_data_preparation.ipynb
│   ├── 00_part2_download_images.ipynb
│   ├── 00_part3a_train_txtimg_lag1.ipynb
│   ├── 00_part3b_train_txtimg_time_ind.ipynb
│   ├── 00_part3c_train_txt_lag1.ipynb
│   ├── 00_part3d_train_txt_time_ind.ipynb
│   ├── 00_part4_verify.ipynb
│   └── 00_part5_create_image_parquet.ipynb
└── code/
    ├── 01_1_create_dataset_txt_img.ipynb
    ├── 01_2_create_dataset_txt.ipynb
    ├── 02_cluster_centroid_products.ipynb
    ├── 02_cluster_centroid_products_random20.ipynb
    ├── 03_1_predictive_performance_txt_img.ipynb
    ├── 03_2_predictive_performance_txt.ipynb
    ├── 04_evaluation_final_all.ipynb
    ├── 04_evaluation_iv_final.ipynb
    ├── 04_scenario1_flats_only.ipynb
    ├── 04_scenario2_excl_top500.ipynb
    ├── 04_scenario3_excl_top500_flats.ipynb
    ├── estimate_vartheta_shoes.ipynb
    ├── requirements.txt
    └── utils/
```

---

## 5. Key design decisions

**Stratified split happens upstream, not after embedding.**
`brand_split/01_stratified_split.ipynb` runs once and produces `data/<bucket>/keys/asin_split.csv` for all 6 buckets in seconds. Stratification axis = `subcat_aggregated × SALES_RANK` quartile, computed within each bucket. Strata with < 4 ASINs merge into `OTHER_SMALL` to avoid degenerate splits. No GPU required, no leakage (the stratification keys are product attributes, not learned).

**Per-bucket embedding isolation.**
Embeddings + downstream model training are run **separately per bucket**. Pooling categories (women's pumps + men's running shoes + …) during embedding extraction produces weights that don't represent any single subgroup well. The 6 buckets are a hard isolation boundary; no bucket should ever read another bucket's predictions, embeddings, or model weights. The only intentionally shared artifacts are:
- `data/images/{ASIN}.jpg` (raw product photos, ASIN-unique)
- `data/_shared/vartheta/vartheta_estimates_shoes.csv` (category-level ϑ)

**Forward-fill before trim (`00_part1`).**
Jan/Feb 2025 data is kept as a fill seed before trimming to the Mar 2025 – Mar 2026 analysis window. This prevents leading NaN in the first weeks of the analysis window.

**BUYBOX_PRICE special handling (`00_part1` cell 18).**
For BUYBOX_PRICE only, we apply `ffill().bfill()` (other columns get `ffill()` only). Many products gained a Buy Box mid-window and have leading NaN that needs back-fill from the first known price. Drop-threshold for BUYBOX is 0.999 (only fully-missing series are dropped), vs 0.0 for other columns (any NaN drops the ASIN).

**ϑ is shared, not per-bucket.**
The Pareto shape parameter ϑ used to convert rank-elasticity → demand-elasticity is a structural parameter of the **category** (shoes), not a brand subsample. Estimate once via `code/estimate_vartheta_shoes.ipynb`, reuse for every bucket. Do not re-estimate per bucket.

---

## 7. References

- Bach et al. (2025). *Adventures in Demand Analysis Using AI.*
- He & Hollenbeck (2020) — toys ϑ ≈ 0.5 baseline.
- See `CHANGES_20260526.md` for the full refactor log from the original single-bucket pipeline to the current 6-bucket isolated pipeline.
