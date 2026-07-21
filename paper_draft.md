# Multimodal Product Representations for Causal Price-Elasticity Estimation: Evidence from Amazon Footwear

> **Draft scaffold.** Content is grounded in the actual pipeline (multimodal
> encoders, LoRA, PCA/similarity controls, DML/PLIV, theta rank->demand
> conversion, embedding-space clustering). Items marked **[FILL]** need the
> author's input (citations, exact numbers to finalize). All demand
> elasticities use theta = 0.6 (multiplier 1/theta = 1.667) unless noted.

---

## Abstract

We estimate own-price demand elasticities for footwear on Amazon using a double
machine learning (DML) framework in which high-dimensional **multimodal product
representations** (text + image embeddings) serve as confounder controls.
Because observed demand is a censored *sales-rank* signal rather than units
sold, we map rank elasticities to demand elasticities via a category-specific
pass-through parameter theta (theta = 0.6, multiplier 1.667). We compare full
fine-tuning against **LoRA** adaptation of the encoders, and contrast
embedding-based controls with **PCA** and similarity-based representations; an
instrumental-variable specification (PLIV) using an embedding-derived competitor
price addresses residual price endogeneity. Across 14 brand buckets, estimated
elasticities are predominantly negative but noisy; only large-sample buckets
yield stable, significant estimates. LoRA corrects several economically
implausible (positive) signs but does not uniformly tighten confidence
intervals, and nuisance-model out-of-sample R^2 remains low, indicating that
small per-brand samples, not the representation choice, are the binding
constraint.

---

## 1. Introduction

Price elasticity is central to pricing and welfare analysis, but on
marketplaces two problems complicate causal estimation: (a) demand is observed
only as a *sales rank*, not as units sold; and (b) price is endogenous, being
correlated with unobserved product quality that shoppers perceive through images
and text descriptions. Modern vision-language encoders turn a product's photo
and description into a vector that proxies exactly those quality confounders;
feeding these embeddings into DML lets us control for "what the shopper sees."

**Research questions.**
1. What is the own-price demand elasticity of Amazon footwear, per brand?
2. Do multimodal embeddings, used as DML controls, yield credible causal
   estimates relative to lower-dimensional (PCA / similarity) alternatives?
3. Does parameter-efficient adaptation (LoRA) of the encoders change the
   estimates or their precision?

**Contributions.** (i) A rank->demand elasticity pipeline with an explicit,
category-calibrated theta; (ii) a like-for-like LoRA vs full fine-tuning
comparison on the same causal target across 14 brands; (iii) per-brand evidence
on when embedding-based DML is trustworthy, supported by nuisance-model
diagnostics.

---

## 2. Data

- **Source.** Amazon footwear (size 8), monthly panel; unit of observation =
  (ASIN x month).
- **Variables.** `BUYBOX_PRICE` (log), `SALES_RANK` (demand proxy), product
  text, product image, and tabular controls (subcategory, rating, review count,
  fulfillment/offer counts, NaN flags).
- **Brand buckets.** Data are partitioned into 14 brand buckets (men/women x
  brand). Each bucket is an **isolation boundary**: embeddings and models are
  never shared across buckets, preventing cross-category contamination of the
  learned representations.
- **Split.** Per-bucket stratified train/validation split on subcategory x
  sales-rank quartile (no GPU; deterministic).

Average buy-box prices across the 14 brands range from about $55
(women_heydude) to $119 (men_olukai). **[FILL: sample period, ASIN counts,
observations per bucket table.]**

---

## 3. Methods

### 3.1 Multimodal product encoders
A per-bucket encoder maps (text, image) to a 768-dimensional embedding; a
SAINT-style tabular encoder handles structured covariates; two output heads
serve the DML nuisance functions (outcome and treatment). Two modality settings
are trained to isolate the marginal value of images: **txtimg** (text + image)
and **txt** (text only).

### 3.2 Parameter-efficient adaptation (LoRA)
The pretrained text/image encoders are adapted either by **full fine-tuning**
or by **LoRA** (low-rank adapters, r = 8, alpha = 16) injected into the
attention query/value projections; LoRA trains a small fraction of parameters
and requires fp32 (fp16 diverged to NaN). The goal is to test whether cheaper,
lower-variance adaptation improves downstream causal estimates on small samples.

### 3.3 Alternative representations
Three control representations are compared as robustness checks: the full
**embeddings**, a **PCA** compression (5 components), and **similarity**
features (cosine similarity to 5 KMeans centroids). Agreement across
representations guards against artifacts of any single one.

### 3.4 Double machine learning (DML / PLR)
We use a partially linear model on first differences:
`Delta_Q = theta * Delta_P + g(X) + e`, with treatment equation
`Delta_P = m(X) + v`, and cross-fitted nuisance learners for `g` (outcome) and
`m` (treatment); `X` = embeddings + tabular controls. Reported specifications:
controls-only, embeddings-only, **embeddings + controls (preferred)**,
similarity + controls, PCA + controls.

**Diagnostics.** Out-of-sample nuisance R^2 (`R2_Y`, `R2_D`) measure whether the
representation actually predicts outcome and treatment; positive out-of-sample
R^2 is a precondition for a trustworthy theta.

### 3.5 Endogeneity: instrumental-variable DML (PLIV)
Prices respond to demand, biasing OLS. We instrument the own-price change
`Delta_P_bb_t` with **`Delta_neighbor_price_t`**, the first difference of the
mean log price of a product's ten nearest neighbors in multimodal-embedding
space. The exclusion restriction is that competitors' price movements shift a
product's own price (through competitive/cost channels) but are uncorrelated
with its idiosyncratic demand shock, conditional on controls. Relevance is
strong (first-stage **F = 34.76**). We estimate a partially-linear IV model
(`DoubleMLPLIV`, LightGBM nuisances) across five control sets; the tabular
controls-only specification is the headline (raw theta = **-0.975**, 90% CI
[-1.59, -0.36], p = 0.009), with embedding/PCA/similarity specs as robustness.
On the theta = 0.6 scale this is a demand elasticity of **-1.625**.

### 3.6 Rank-to-demand conversion (theta)
DML/PLIV yield a **rank** elasticity theta. Since units sold scale as
rank^(-1/theta), the **demand** elasticity = theta * (1/theta_hat). We adopt a
footwear-calibrated **theta_hat = 0.6** (multiplier 1.667). A separate
sensitivity notebook explores alternatives (theta ~ 0.586 review-proxy, 0.875
rank-distribution; the toys-paper baseline 0.5 does not apply to footwear).
*Consistency note: the main PLR pipeline uses 1.667 throughout; the IV notebook
must use the same 1.667 (not the legacy toys-baseline x2) so both report on one
scale.*

### 3.7 Heterogeneity (clustering / GATES / CATE)
Products are clustered in embedding space (5 KMeans clusters). Cluster-level
average treatment effects (GATES) and the CATE distribution describe how
elasticity varies within a brand; the IV variant refits PLIV per cluster where
the native `.gate()` is unavailable.

---

## 4. Results

- **Main estimates (Fig. 1-2, Table 1).** Per-brand demand elasticities for
  txtimg and txt (theta = 0.6). Estimates are predominantly negative, with most
  magnitudes in [-2, 0].
- **Reliability.** Only large-N buckets are stable and significant, e.g.
  **men_top2_nike** (txtimg -1.54, txt -1.10; both 90%-significant) and
  **women_dr_scholl** (about -2.3 to -3.5, significant). Small buckets
  (men_reebok N=66, women_nike N=121) show extreme point estimates with very
  wide confidence intervals.
- **Elasticity vs price (Fig. 1/2).** Across brands there is no systematic
  monotone relationship between average price and elasticity.
- **LoRA vs full fine-tuning.** LoRA flips several implausible positive signs
  toward negative (e.g. men_adidas txt +0.78 -> -0.08; women_adidas txtimg
  +0.75 -> -1.20) but does **not** uniformly narrow CIs: LoRA CI widths range
  from about 60% to 260% of the full-FT width, and the effect is larger under
  txt than txtimg.
- **Nuisance R^2.** The treatment-equation R^2 (`R2_D`) is negative for most
  buckets, i.e. the representations fail to residualize price out of sample.

**[FILL: Table 1 = the per-bucket comparison; Figures 1-2 = the two
`data/_shared/elasticity_vs_price_{txtimg,txt}.png` charts; IV Table = the
5-spec PLIV summary on the 1.667 scale.]**

---

## 5. Discussion

Embedding-based DML is **necessary but not sufficient**. Multimodal embeddings
do capture the visible quality confounders that bias naive regressions (adding
embeddings consistently pushes the price coefficient more negative), but with
roughly 60-1000 observations per brand the nuisance functions are
under-identified, so theta inherits high variance. The binding constraint is
**sample size / signal strength**, not the choice of representation: PCA,
similarity, and full embeddings agree qualitatively, and LoRA (a
variance-reduction device) improves signs without delivering precision.

---

## 6. Limitations

- The rank->units mapping hinges on theta; all demand elasticities scale
  linearly with 1/theta_hat.
- Per-brand isolation trades bias (no cross-category contamination) for variance
  (small N).
- Short monthly panel; the image parquet is used only for qualitative cluster
  visualization, not for the causal estimates.
- IV validity rests on the exclusion restriction for competitor prices, which is
  assumed rather than testable.

---

## 7. Conclusion

Multimodal embeddings provide an interpretable way to control for
shopper-visible quality in marketplace elasticity estimation, and LoRA offers a
cheaper, sign-stabilizing encoder adaptation. However, credible per-brand
elasticities require pooling or larger samples: our reliable estimates come
exclusively from high-volume brands, and diagnostic R^2 shows small-brand
estimates should be read as directional at best. **Answering RQ1-RQ3:**
elasticities are negative where identifiable; embeddings are as good as, but not
clearly better than, PCA/similarity for identification; and LoRA improves
economic plausibility without solving the small-sample precision problem.

---

## References (to complete)

- **[FILL]** DML: Chernozhukov et al. (2018), *Double/Debiased Machine
  Learning*.
- **[FILL]** DoubleML software / PLIV: Bach et al. (2022/2025).
- **[FILL]** Sales-rank to units (theta): He & Hollenbeck (2020).
- **[FILL]** LoRA: Hu et al. (2021).
- **[FILL]** SAINT (tabular): Somepalli et al. (2021).

---

### Logical loop (one line)
Price is endogenous and demand is only observed as rank -> use multimodal
embeddings inside DML to control for shopper-visible quality, and convert the
rank elasticity to a demand elasticity via theta -> find elasticities are mostly
negative but noisy, reliable only for large brands, with LoRA fixing signs but
not precision -> the bottleneck is sample size, not the representation method.