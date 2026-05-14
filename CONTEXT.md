# delinquency-forecast — Technical Context Document

> **Audience**: LLM agents onboarding to this codebase. This document describes what was built, why each decision was made, and the exact state of all artifacts. Read this before modifying any script.

---

## 1. Project Objective

Predict daily crime counts per H3 hexagonal cell (resolution 9, ~0.1 km²) across Jalisco, Mexico, with a confidence score that quantifies prediction reliability. The primary operational use case is spatial prioritization: rank all cells by predicted risk to concentrate patrol resources. The evaluation criterion is therefore ranking quality (lift, Gini), not point-estimate accuracy (MAPE).

**Geographic scope**: Jalisco, Mexico. Bounding box: lon ∈ [−106, −101], lat ∈ [18.5, 23.0]. Two sub-zones tracked throughout: **AMG** (Zona Metropolitana de Guadalajara) and **Interior**.

---

## 2. Data Source

**IIEG (Instituto de Información Estadística y Geográfica de Jalisco)**
- 12 regional files, one per administrative region
- 627,674 crime records, 2017-01 through 2025-12
- Key columns: `fecha`, `delito`, `x` (longitude), `y` (latitude), `colonia`, `municipio`, `clave_mun`, `zona_geografica`
- Coordinate coverage: 598,271 records have valid (x, y) → **95.3% geocoding rate**
- Date format inconsistency: Sierra de Amula region uses `DD/MM/YYYY`; all others use `YYYY-MM-DD` — handled in `build_iieg_unified.py`

Unified output: `data/processed/iieg_unified.parquet` (627,674 rows).

---

## 3. Spatial Index

**Decision: H3 resolution 9**

H3 res=9 cells have area ≈ 0.1 km², edge length ≈ 174 m. Each cell has exactly 6 ring-1 neighbors and 12 ring-2 neighbors (except boundary cells).

Rationale:
- Coarser resolutions (7, 8) lose spatial discrimination needed for patrol routing
- Finer resolutions (10, 11) exceed geocoding precision of IIEG data (~100–200 m typical GPS error), causing artificial sparsity
- res=9 matches the geocoding noise level: a misplaced crime lands in a neighboring cell rather than a distant one, which the target smoothing kernel (§7) explicitly corrects

Active cells (with ≥1 crime in training period): **20,632** out of all possible cells covering Jalisco.

---

## 4. Pipeline Architecture

Two-stage model. Each stage is a separate LightGBM model:

```
Stage 1 (monthly):  H3 × month  → λ_m  (expected crimes this month)
Stage 2 (daily):    H3 × day    → λ_d  (expected crimes this day)
                    uses log(λ_m / days_in_month) as primary offset feature
```

**Why two stages instead of one daily model?**

A single daily model on 67M rows with monthly-scale features (rolling_mean_12, hist_mean_mes) would require either: (a) repeating expensive monthly aggregations 30× per cell-month, or (b) losing those features entirely. The two-stage decomposition avoids this: Stage 1 summarizes all monthly signal into a single scalar (λ_m), which Stage 2 consumes as `log_lam_dia_base`. Stage 2 then only needs to learn the daily residual (day-of-week effect, near-repeat contagion, holiday effects) on top of the monthly baseline.

Feature importance from Stage 2 confirms the architecture is sound: `log_lam_dia_base` has gain 2,613,495, which is 5.6× the next feature. The daily model is genuinely learning a correction on top of Stage 1, not re-learning from scratch.

---

## 5. Pipeline Scripts (in execution order)

All scripts use `Path(__file__).parent.parent.parent` to anchor paths to the repo root, so they can be called from any working directory.

### 5.1 Ingestion

| Script | Input | Output |
|--------|-------|--------|
| `src/ingestion/download_iieg.py` | IIEG website | `data/raw/iieg/*.csv` |
| `src/ingestion/build_iieg_unified.py` | `data/raw/iieg/` | `data/processed/iieg_unified.parquet` |
| `src/ingestion/build_manzanas_cpv.py` | RESAGEBURB | `data/processed/manzanas_cpv.parquet` |

### 5.2 Monthly model

| Script | Input | Output |
|--------|-------|--------|
| `src/crime_monthly/build_crime_timeseries.py` | `iieg_unified.parquet` | `crime_timeseries.parquet` |
| `src/crime_monthly/build_features_temporal.py` | `crime_timeseries.parquet` | `features_temporal.parquet` |
| `src/crime_monthly/train_lgbm_poisson.py` | `features_temporal.parquet` | `models/lgbm_poisson_v1.pkl` |

### 5.3 Daily model

| Script | Input | Output |
|--------|-------|--------|
| `src/crime_daily/build_crime_timeseries_daily.py` | `iieg_unified.parquet` + `crime_timeseries.parquet` | `crime_timeseries_daily.parquet` |
| `src/crime_daily/build_features_daily.py` | `crime_timeseries_daily.parquet` + `lgbm_poisson_v1.pkl` | `features_daily.parquet` |
| `src/crime_daily/train_lgbm_daily.py` | `features_daily.parquet` + `lgbm_poisson_v1.pkl` | `models/lgbm_daily_v1.pkl` |

### 5.4 Evaluation and inference

| Script | Purpose |
|--------|---------|
| `src/crime_daily/hitrate_diagnostico.py` | Computes lift tables and Gini for 3 baselines on test 2025 |
| `src/predict.py` | Single-cell inference CLI and library interface |

---

## 6. Feature Engineering

### 6.1 Monthly features (`features_temporal.parquet`, 26 columns)

| Feature | Description |
|---------|-------------|
| `lag_1..12` | Lagged crime counts at 1, 2, 3, 6, 12 months |
| `rolling_mean_3/6/12` | Causal rolling mean (window starts 1 period before target) |
| `rolling_std_3` | Rolling standard deviation, 3-month window |
| `hist_mean_mes` | Historical mean for same calendar month (expanding, causal) |
| `hist_max` | Expanding historical maximum |
| `mes_sin`, `mes_cos` | Cyclic encoding of month: sin(2π·m/12), cos(2π·m/12) |
| `trend` | Months since 2017-01 (linear trend) |
| `trimestre` | Calendar quarter (1–4) |
| `es_fin_año` | Binary: month ∈ {11, 12} |
| `es_verano` | Binary: month ∈ {7, 8} |
| `zona_geografica`, `region`, `clave_mun`, `trimestre` | LabelEncoded categoricals |

Encoders are fitted on training data and stored in `lgbm_poisson_v1.pkl["encoders"]` for reuse by the daily pipeline and inference.

### 6.2 Daily features (`features_daily.parquet`, 25 columns)

| Feature | Description | Computation |
|---------|-------------|-------------|
| `log_lam_dia_base` | log(λ_m / days_in_month) — Stage 1 offset | Monthly model prediction |
| `dia_semana` | Day of week 0=Mon…6=Sun | pandas `.dayofweek` |
| `es_fin_semana` | Binary: dia_semana ≥ 5 | |
| `es_festivo` | Mexican public holiday | `holidays.Mexico()` |
| `mes` | Calendar month 1–12 | |
| `lag_1`, `lag_7`, `lag_14` | Crime count 1/7/14 days prior | Matrix shift on (n_h3 × n_dates) |
| `rolling_mean_7`, `rolling_mean_14` | Causal rolling mean, shifted 1 day | Vectorized cumsum |
| `rolling_std_7` | Causal rolling std with Bessel correction | Vectorized cumsum on x² |
| `zona_geografica`, `clave_mun` | Reused from monthly encoders | |
| `nr_ring1_1d/7d/14d` | Crime sum in ring-1 neighbors, past 1/7/14 days | Sparse A1 @ rolling_sum matrix |
| `nr_ring2_7d` | Crime sum in ring-2 neighbors, past 7 days | Sparse A2 @ rolling_sum matrix |
| `target_smoothed` | Smoothed training target (§7) | 0.60·conteo + (0.40/6)·A1@conteo |

**Memory note**: `features_daily.parquet` is 101 MB on disk. The `build_features_daily.py` script peaks at ~5–8 GB RAM during construction (67M rows × two sparse matrix multiplications).

**Vectorization approach**: All lags and rolling stats are computed by reshaping the panel into a (20,632 × 3,286) matrix and applying numpy cumsum operations. This replaces groupby+transform which would be prohibitively slow at 67M rows.

---

## 7. Near-Repeat Features

The near-repeat hypothesis states that a crime at location X elevates risk in nearby cells in subsequent days. This is a well-documented criminological phenomenon.

**Implementation**: Two sparse CSR adjacency matrices are built once over all 20,632 H3 cells:
- `A1`: ring-1 adjacency (68,168 connections, 6 neighbors per interior cell)
- `A2`: ring-2 adjacency (114,028 connections, 12 cells per interior cell)

Near-repeat features are computed as: `nr_feature = A @ rolling_sum_matrix` where the rolling sum matrix (n_h3 × n_dates) contains causal rolling crime totals. The sparse matrix multiply runs in seconds for the full panel.

Lift contribution of near-repeat features: `nr_ring1_14d` and `nr_ring2_7d` are the 2nd and 4th most important features by gain, explaining part of the +17% lift Stage 2 adds over Stage 1.

---

## 8. Target Smoothing (60/40 Kernel)

**Problem**: At H3 res=9, geocoding noise (±100–200 m) causes crimes to be recorded in a neighboring cell rather than the true cell. This creates artificial zero-inflation in training targets and noisy gradients.

**Solution**: Replace the raw `conteo` target with `target_smoothed`:

```
target_smoothed[i] = 0.60 × conteo[i] + (0.40 / 6) × Σ_{j ∈ ring1(i)} conteo[j]
```

Each crime at cell i contributes 60% to cell i's training target and 6.67% to each of the 6 ring-1 neighbors. Interior cells conserve mass exactly (60% + 6×6.67% = 100%). Boundary cells leak ~2.87% total mass (cells on the Jalisco border have fewer than 6 neighbors in the panel).

The model is **trained** on `target_smoothed` but **evaluated** on `conteo` (real crime counts). Early stopping also uses `target_smoothed` on the validation set (this is a known tradeoff: val loss is on smoothed targets, not real counts, but experiments showed it does not cause overfitting on the smoothing artifact).

---

## 9. Model Hyperparameters

### Stage 1 — Monthly (`lgbm_poisson_v1.pkl`)

```python
objective        = "poisson"   # selected dynamically: variance/mean = 1.31 < 2.0
metric           = "mae"
learning_rate    = 0.05
num_leaves       = 63
min_child_samples = 20
feature_fraction  = 0.8
bagging_fraction  = 0.8
bagging_freq      = 5
reg_alpha         = 0.1
reg_lambda        = 0.1
num_boost_round   = 1000       # with early_stopping(50)
best_iteration    = 568
seed              = 42
```

Objective selection: if variance/mean of nonzero training counts > 2.0, use `tweedie` with `power=1.5`; otherwise `poisson`. Current data: 1.31 → Poisson.

### Stage 2 — Daily (`lgbm_daily_v1.pkl`)

```python
objective         = "poisson"
metric            = "mae"
learning_rate     = 0.05
num_leaves        = 31          # smaller than Stage 1 to avoid overfitting on 0-inflated panel
min_child_samples = 100         # high floor for the 99.2%-zero panel
feature_fraction  = 0.8
bagging_fraction  = 0.8
bagging_freq      = 5
reg_alpha         = 0.1
reg_lambda        = 0.5
num_boost_round   = 500         # with early_stopping(50)
best_iteration    = 500         # did not trigger early stopping — could benefit from more rounds
seed              = 42
```

**Active cell filter**: Only H3 cells with ≥ 10 crimes in the training set (≤2023) are included in the training DataFrame. This filters 20,632 → 6,907 cells and reduces training rows from 52.7M to 17.7M. The remaining 13,725 cells (67%) are structurally near-empty and would dominate the gradient with zero-count updates.

---

## 10. Temporal Split

Consistent across all scripts:

| Set | Period | Rows (monthly) | Rows (daily) |
|-----|--------|----------------|--------------|
| Train | ≤ 2023 | 1,733,088 | 52,735,392 |
| Val | 2024 | 247,584 | 7,551,312 |
| Test | 2025 | 247,584 | 7,510,048 |

Val is used for early stopping and model selection. Test is held out and only touched in `hitrate_diagnostico.py`. The val early stopping uses `target_smoothed`; all reported metrics use `conteo`.

---

## 11. Evaluation Results (Test 2025)

30,909 real crimes across 7,510,048 cell-day observations (0.41% non-zero rate).

### Why not MAPE?

MAPE on the daily model is ~97%. This is misleading: when the true count is 0 and the prediction is 0.01, MAPE is undefined and excluded; when the true count is 1 and the prediction is 0.3, MAPE is 70%. With 99.2% zeros, MAPE measures noise. The operationally relevant question is: "do high-prediction cells actually contain crimes?" — which is what lift and Gini measure.

### Lift and Gini (ranking quality)

| Model | Gini | Top 0.1% | Top 1% | Top 5% | Top 10% |
|-------|------|----------|--------|--------|---------|
| Random baseline | 0.052 | 0.2× | 0.2× | 0.2× | 0.2× |
| Stage 1 only (λ_m/days) | 0.729 | 27.5× | 12.9× | 7.4× | 5.7× |
| Stage 1 + Stage 2 | **0.750** | **31.7×** | **15.1×** | **8.8×** | **6.2×** |

**Interpretation**: The top 5% of cell-day predictions (by Stage 1+2 score) capture 44.1% of all real crimes in 2025. Stage 2 adds a consistent +17% lift over Stage 1 alone across all cutoffs (ratio E1+2/E1 ≈ 1.15–1.19×).

### Stage 2 training metrics (val 2024)

| Zone | MAE | RMSE |
|------|-----|------|
| Global | 0.0161 | 0.0868 |
| AMG | 0.0306 | 0.1250 |
| Interior | 0.0076 | 0.0532 |

### Stage 2 feature importance (gain, top 10)

| Feature | Gain |
|---------|------|
| log_lam_dia_base | 2,613,495 |
| nr_ring1_14d | 341,253 |
| dia_semana | 297,160 |
| nr_ring2_7d | 130,821 |
| es_fin_semana | 65,440 |
| rolling_mean_14 | 48,374 |
| es_festivo | 29,667 |
| nr_ring1_7d | 19,963 |
| clave_mun | 18,212 |
| zona_geografica | 17,346 |

---

## 12. Inference Interface (`src/predict.py`)

Two modes based on date format:
- `YYYY-MM` → monthly prediction (Stage 1 only)
- `YYYY-MM-DD` → daily prediction (Stage 1 + Stage 2)

Location input: either `--lat`/`--lon` (converted to H3 via `h3.latlng_to_cell(lat, lon, 9)`) or `--h3` (direct H3 index).

### Output fields

```python
{
  "granularidad":          "diaria" | "mensual",
  "fecha":                 str,
  "h3_index":              str,           # H3 res=9 index
  "municipio":             str,
  "zona_geografica":       str,
  "prediccion":            float,         # λ (expected crimes)
  "rango_probable":        (int, int),    # 80% Poisson CI
  "rango_amplio_95pct":    (int, int),    # 95% Poisson CI
  "nivel_riesgo":          "bajo"|"medio"|"alto",
  "confiabilidad_score":   int,           # 0–100
  "confiabilidad_nivel":   "CONFIABLE"|"APROXIMADA"|"ESPECULATIVA",
  "confiabilidad_nota":    str,
  "cv_historico":          float | None,
  "meses_historial":       int,
  "brecha_temporal_dias":  int,           # days extrapolated beyond training end (2023-12-31)
  # daily only:
  "dia_semana":            str,
  "es_festivo":            bool,
  "prediccion_mensual":    float,
}
```

### Confidence score logic

Starts at 100, deductions applied additively:

| Condition | Deduction |
|-----------|-----------|
| 0 months history | −50 |
| < 6 months history | −30 |
| < 24 months history | −10 |
| Zero rate > 90% | −25 |
| Zero rate > 80% | −10 |
| Extrapolation > 24 months | −20 |
| Extrapolation > 12 months | −10 |
| Extrapolation > 6 months | −5 |
| CV > 2.0 | −10 |

Levels: ≥65 → CONFIABLE, ≥35 → APROXIMADA, <35 → ESPECULATIVA.

### Risk level logic

```python
if p66 == 0:          # low-crime cell: all historical months are zero
    p33, p66 = 1.0, 3.0   # use global reasonable defaults to avoid all-ALTO bug
if lam <= p33: "bajo"
elif lam <= p66: "medio"
else: "alto"
```

The `p66 == 0` guard was added to fix a bug where any λ > 0 in a near-empty cell was classified as "alto" (p33 = p66 = 0 when all historical months are zero).

---

## 13. Artifact Files

```
models/
  lgbm_poisson_v1.pkl    # Stage 1 — keys: model, encoders, feature_cols, objective, overdispersion_ratio
  lgbm_daily_v1.pkl      # Stage 2 — keys: model, encoders, feature_cols, objective, min_crimes_train_filter

data/processed/
  iieg_unified.parquet           # 627,674 rows — unified raw microdata with h3_9 column
  crime_timeseries.parquet       # 2,228,256 rows — H3 × month panel (2017-01 to 2025-12)
  features_temporal.parquet      # 2,228,256 rows — monthly features for Stage 1
  crime_timeseries_daily.parquet # 67,796,752 rows — H3 × day panel (2017-01-01 to 2025-12-30)
  features_daily.parquet         # 67,796,752 rows — daily features + target_smoothed
```

`models/test/` contains copies of both `.pkl` files produced during a pipeline validation run. These are functionally identical to `models/` (same data, same params, same random seed).

---

## 14. Known Limitations

**Target smoothing mass loss (2.87%)**: The 60/40 kernel distributes 40% of each crime to 6 ring-1 neighbors. Cells on the Jalisco border have fewer than 6 neighbors in the panel, so a fraction of the mass distributes outside the panel and is lost. This is intentional (border cells genuinely have missing neighbor data) but means `target_smoothed.sum() < conteo.sum()`.

**Daily Stage 2 did not trigger early stopping**: `best_iteration = 500 = num_boost_round`. The model may benefit from more boosting rounds. This is a low-priority refinement since Gini is already 0.75.

**No lag features for near-zero cells in inference**: `predict_daily` loads the H3 cell's full daily history from `crime_timeseries_daily.parquet` using a Parquet filter push-down. Near-repeat features load neighboring cells. This is correct but slow for batch inference (individual parquet reads per cell). Batch prediction should use the pre-built `features_daily.parquet` directly.

**Encoders handle unseen labels with fallback**: If a new zona_geografica or clave_mun appears in future IIEG data, `build_features_daily.py` maps it to the first class of the encoder. This is a silent fallback — a data validation step should flag new values explicitly.

**Training cutoff is hardcoded**: `max_train_date = pd.Timestamp("2023-12-31")` is hardcoded in `predict.py` for the temporal gap calculation. If the model is retrained with newer data, this must be updated manually.

---

## 15. Conda Environment

Environment name: `geomod` (Python 3.11). Key packages:
- `lightgbm` — gradient boosted trees
- `h3` — H3 spatial indexing (`h3.latlng_to_cell`, `h3.grid_disk`, `h3.grid_ring`)
- `scipy.sparse` — CSR sparse matrices for near-repeat computation
- `holidays` — Mexican public holiday calendar
- `pandas`, `numpy`, `scikit-learn`

Install: `conda env create -f environment.yml` (file not yet exported — see pending work).

---

## 16. Stage 3 — Crime Composition Model (`lgbm_e3_v1.pkl`)

Stage 3 answers a different question from Stages 1 and 2: not *how many* crimes, but *what kind*. Given that a crime occurs in cell H3 on day D, what category is it most likely to be?

### 16.1 Four crime categories

| Category | Crimes included |
|---|---|
| `alto_impacto` | Homicidio doloso, Feminicidio, Violación |
| `violencia_personal` | Violencia familiar, Lesiones dolosas, Abuso sexual infantil |
| `robo_confrontacion` | Robo a persona, Robo a negocio, Robo a cuentahabientes, Robo a bancos |
| `robo_patrimonial` | Robo a vehículos, Robo de motocicleta, Robo a int. de vehículos, Robo de autopartes, Robo a casa habitación, Robo a carga pesada |

Crimes not matching any category are excluded from training (none were found in the 2017–2025 data).

### 16.2 Model design

- **Algorithm**: LightGBM `multiclass` with softmax output, 4 classes
- **Training unit**: individual incident (one row per crime in `iieg_unified.parquet` with valid H3), not a cell-day panel
- **Output**: probability vector [P(alto_impacto), P(violencia_personal), P(robo_confrontacion), P(robo_patrimonial)], sums to 1
- **Class imbalance**: `alto_impacto` represents 3.3% of incidents. Addressed with square-root sample weights (`np.sqrt(compute_sample_weight("balanced", y_train))`), which is a middle ground between unweighted (recall ≈ 1%) and fully balanced (recall ≈ 42% but accuracy drops to 45%)

### 16.3 Training split (incident-level)

| Set | Period | Incidents |
|---|---|---|
| Train | ≤ 2023 | 511,735 |
| Val | 2024 | 55,627 |
| Test | 2025 | 30,909 |

### 16.4 Features (51 total)

**Inherited from Stage 2** (same cell, same day):
- `log_lam_dia_base`, `dia_semana`, `es_fin_semana`, `es_festivo`, `mes`
- `lag_1`, `lag_7`, `lag_14`, `rolling_mean_7`, `rolling_mean_14`, `rolling_std_7`
- `nr_ring1_1d`, `nr_ring1_7d`, `nr_ring1_14d`, `nr_ring2_7d`
- `zona_geografica`, `clave_mun`
- `lambda_diario` — Stage 2 prediction for that cell-day (most important feature)

**POIs from DENUE** (static per H3 cell, 9 categories):
- `poi_bancos`, `poi_bares`, `poi_escuelas`, `poi_salud`, `poi_conveniencia`
- `poi_hoteles`, `poi_gasolineras`, `poi_mercados`, `poi_farmacias`
- `poi_total` (sum of all POI counts)
- Binary flags (`poi_*_flag`) were removed after analysis showed gain < 5K — redundant with counts

**Infrastructure from INEGI INV 2020** (13 proportions per H3 cell):
- `inv_banqueta_pct`, `inv_alumpub_pct`, `inv_drenajep_pct`, `inv_transcol_pct`
- `inv_semapeat_pct`, `inv_paratran_pct`, `inv_rampas_pct`, `inv_pasopeat_pct`
- `inv_ciclovia_pct`, `inv_puestosf_pct`, `inv_puestoam_pct`, `inv_arboles_pct`, `inv_guarnici_pct`
- `inv_n_segmentos` (street segment count as coverage proxy)

**Category-specific near-repeat features** (8 features):
- `nr_cat_{alto,viol,conf,patr}_ring1_{7,14}d`: crime count in ring-1 neighbors in the past 7/14 days, broken down by crime category
- Hypothesis: crime contagion is species-specific — a violent incident yesterday is a better predictor of today's violence than a vehicle theft
- In practice, these features did not appear in the top 20 by gain, suggesting the global near-repeat features already capture most of the spatial contagion signal

### 16.5 Hyperparameters

```python
objective         = "multiclass"
num_class         = 4
metric            = "multi_logloss"
num_leaves        = 63
learning_rate     = 0.05
min_child_samples = 50
feature_fraction  = 0.8
bagging_fraction  = 0.8
bagging_freq      = 5
reg_alpha         = 0.1
reg_lambda        = 0.1
num_boost_round   = 500    # early stopping patience = 50; best_iteration = 490
```

### 16.6 Post-training calibration and threshold optimization

**Isotonic calibration**: A separate `IsotonicRegression` is fitted one-vs-rest for each of the 4 classes on the validation set (2024). This significantly improves probability calibration measured by Expected Calibration Error (ECE):

| Class | ECE before | ECE after |
|---|---|---|
| alto_impacto | 0.060 | 0.0015 |
| violencia_personal | 0.044 | 0.0028 |
| robo_confrontacion | 0.051 | 0.0062 |
| robo_patrimonial | 0.089 | 0.0095 |

**Threshold optimization**: Per-class multipliers are optimized via Nelder-Mead on the calibrated validation probabilities to maximize macro-F1. The multipliers are applied at inference time as `(probs * thresholds).argmax(axis=1)`.

### 16.7 Evaluation results (Test 2025)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| alto_impacto | — | 12% | 0.12 | 1,012 |
| violencia_personal | — | — | 0.37 | 5,465 |
| robo_confrontacion | — | — | 0.38 | 8,162 |
| robo_patrimonial | — | — | 0.67 | 16,270 |
| **macro avg** | — | — | **0.38** | 30,909 |

Accuracy: **55%** | Log-loss (calibrated): **1.017**

Confusion matrix — TEST 2025:

```
                      alto_imp  viol_pers  robo_conf  robo_patr
alto_impacto  (1,012)      119        145        162        586
viol_personal (5,465)      104      2,045      1,497      1,819
robo_confron. (8,162)      267      1,533      2,921      3,441
robo_patrimon(16,270)      513      1,921      2,684     11,152
```

Top features by gain: `lambda_diario` (407K), `log_lam_dia_base` (167K), `poi_conveniencia` (149K), `zona_geografica` (92K), `inv_n_segmentos` (83K).

### 16.8 Auxiliary data processing scripts

| Script | Input | Output |
|---|---|---|
| `src/crime_composition/01_build_denue_h3.py` | DENUE 2025 ZIP | `data/processed/denue_h3.parquet` (13,009 cells × 20 cols) |
| `src/crime_composition/02_build_inv_h3.py` | INEGI INV 2020 national ZIP | `data/processed/inegi_inv_h3.parquet` (14,372 cells × 15 cols) |
| `src/crime_composition/03_train_e3.py` | all of the above + E2 model | `models/lgbm_e3_v1.pkl` |

**DENUE processing**: POI establishments matched by SCIAN code prefix (e.g., `"522"` = banks, `"7224"` = bars). State 14 (Jalisco) only. Output is count of establishments per H3 cell per category.

**INV processing**: National shapefile extracted for state 14. Street segment LINESTRING geometries reprojected from ITRF2008 Lambert Conformal Conic to WGS84, centroid mapped to H3 res=9. Infrastructure columns (binary: 1=available, 3=not available) aggregated as proportions per cell.

### 16.9 Artifact

```
models/lgbm_e3_v1.pkl   # 13.9 MB
  keys: model, feature_cols, classes, label_encoder, calibrators, thresholds

data/processed/
  denue_h3.parquet       # 13,009 rows × 20 cols (9 POI counts + poi_total + 9 flags + h3_9)
  inegi_inv_h3.parquet   # 14,372 rows × 15 cols (13 infrastructure pct + inv_n_segmentos + h3_9)
```

---

## 17. Interactive Visualization (Gradio App)

The three model stages are exposed through a Gradio web application (`app.py`) with a Folium interactive map.

### 17.1 Inputs

- **Date**: any date from 2017-01-01 onwards. Dates within the historical range (≤ 2025-12-30) use real features from `features_daily.parquet`. Dates beyond that use synthetic features generated from the historical monthly average of the same calendar month in the most recent reference year (2024).
- **Search**: municipality name, neighborhood (geocoded via Nominatim), or direct H3 index. Defaults to Guadalajara.

### 17.2 Map output

Each active H3 cell is rendered as a hexagonal polygon colored on a 5-tone traffic-light palette (green → red) proportional to `lambda_diario` (percentile 10 = min, percentile 95 = max). Hovering a cell shows:
- λ_diario value and risk level (bajo / medio / alto)
- Most probable crime category (Stage 3 prediction)
- Proportional bar of the four category probabilities
- Reliability score (0–100)

Up to 1,500 cells are rendered per query (top by λ_diario) to maintain browser performance.

### 17.3 Reliability score for future dates

For historical dates, reliability reflects data coverage (POI availability, infrastructure data, crime history depth). For future dates, the primary uncertainty source is that lag features and near-repeat counts are replaced with historical monthly averages — not actual recent observations. The penalty is therefore based on distance from the last real data point (2025-12-30), not from today:

| Distance from last data | Penalty | Typical score |
|---|---|---|
| 0 (historical) | 0 pts | ~99/100 |
| 1–31 days beyond data | −25 pts | ~65–75/100 |
| 32–180 days beyond data | −35 pts | ~55–65/100 |
| > 180 days beyond data | −45 pts | ~45–55/100 |

A banner is displayed when the selected date is outside the historical range, showing how many days beyond the last data point the prediction is and indicating whether it is a near-term forecast or a speculative projection.
