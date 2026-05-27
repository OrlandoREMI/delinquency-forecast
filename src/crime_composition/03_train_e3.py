"""
Entrena Etapa 3: modelo de composición de riesgo por incidente.
LightGBM multiclass (4 categorías de delito).

Mejoras sobre v1:
  - Near-repeats por categoría: nr_cat_{X}_ring1_{7,14}d
  - Calibración isotónica por clase (ajustada en VAL 2024)
  - Sin flags binarios de DENUE (gain < 5K, redundantes con conteos)
  - sqrt weights + optimización de umbrales

Output: models/lgbm_e3_v1.pkl
"""
import pickle
from pathlib import Path

import h3
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).parent.parent.parent

IIEG_PATH  = ROOT / "data/processed/iieg_unified.parquet"
FEAT_PATH  = ROOT / "data/processed/features_daily.parquet"
DENUE_PATH = ROOT / "data/processed/denue_h3.parquet"
INV_PATH   = ROOT / "data/processed/inegi_inv_h3.parquet"
E2_PKL     = ROOT / "models/lgbm_daily_v1.pkl"
OUT_PKL    = ROOT / "models/lgbm_e3_v1.pkl"

DELITO_CATEGORIA = {
    "Homicidio doloso":              "alto_impacto",
    "Feminicidio":                   "alto_impacto",
    "Violación":                     "alto_impacto",
    "Violencia familiar":            "violencia_personal",
    "Lesiones dolosas":              "violencia_personal",
    "Abuso sexual infantil":         "violencia_personal",
    "Robo a persona":                "robo_confrontacion",
    "Robo a negocio":                "robo_confrontacion",
    "Robo a cuentahabientes":        "robo_confrontacion",
    "Robo a bancos":                 "robo_confrontacion",
    "Robo a vehículos particulares": "robo_patrimonial",
    "Robo de motocicleta":           "robo_patrimonial",
    "Robo a int de vehículos":       "robo_patrimonial",
    "Robo de autopartes":            "robo_patrimonial",
    "Robo a casa habitación":        "robo_patrimonial",
    "Robo casa habitación":          "robo_patrimonial",
    "Robo a carga pesada":           "robo_patrimonial",
}

CLASES     = ["alto_impacto", "violencia_personal", "robo_confrontacion", "robo_patrimonial"]
CAT_SHORTS = ["alto", "viol", "conf", "patr"]

E2_FEATURE_COLS = [
    "log_lam_dia_base", "dia_semana", "es_fin_semana", "es_festivo", "mes",
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    "zona_geografica", "clave_mun",
    "nr_ring1_1d", "nr_ring1_7d", "nr_ring1_14d", "nr_ring2_7d",
]

TRAIN_END = pd.Timestamp("2023-12-31")
VAL_END   = pd.Timestamp("2024-12-31")


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_incidents() -> pd.DataFrame:
    print("Cargando iieg_unified...")
    df = pd.read_parquet(IIEG_PATH, columns=["h3_9", "fecha", "delito"])
    df = df.dropna(subset=["h3_9", "fecha"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["categoria"] = df["delito"].map(DELITO_CATEGORIA)
    n_total = len(df)
    df = df.dropna(subset=["categoria"]).reset_index(drop=True)
    print(f"  {n_total:,} incidentes → {len(df):,} con categoría asignada")
    return df[["h3_9", "fecha", "categoria"]]


def merge_daily_features(incidents: pd.DataFrame) -> pd.DataFrame:
    """Lee features_daily año por año y hace merge incremental para no tener 67M filas en RAM."""
    print("Mergeando con features_daily (año por año)...")
    needed = ["h3_9", "fecha"] + E2_FEATURE_COLS
    incident_cells = set(incidents["h3_9"])

    chunks = []
    for year in range(2017, 2026):
        print(f"  Año {year}...", end=" ", flush=True)
        chunk = pd.read_parquet(FEAT_PATH, columns=needed,
                                filters=[("año", "=", year)])
        chunk = chunk[chunk["h3_9"].isin(incident_cells)]
        chunk["fecha"] = pd.to_datetime(chunk["fecha"])

        year_inc = incidents[incidents["fecha"].dt.year == year]
        merged   = year_inc.merge(chunk, on=["h3_9", "fecha"], how="left")
        chunks.append(merged)
        del chunk
        print(f"{len(merged):,}")

    df = pd.concat(chunks, ignore_index=True)
    n_sin = df[E2_FEATURE_COLS[0]].isna().sum()
    print(f"  Sin features (fecha fuera de rango): {n_sin:,}")
    return df.dropna(subset=[E2_FEATURE_COLS[0]])


# ---------------------------------------------------------------------------
# Near-repeats por categoría
# ---------------------------------------------------------------------------

def build_cat_nearrepeat(incidents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada incidente calcula near-repeats de vecinos ring-1 por categoría de delito:
      nr_cat_{X}_ring1_7d   — suma de crímenes de tipo X en vecinos ring-1, últimos 7 días
      nr_cat_{X}_ring1_14d  — ídem 14 días

    Hipótesis: el delito contagia su propia especie. Las features globales (nr_ring1_14d)
    no distinguen si el calor vecinal viene de violencia o de robo patrimonial.
    """
    print("Construyendo near-repeats por categoría...")

    all_cells = sorted(incidents_df["h3_9"].unique())
    cell_idx  = {c: i for i, c in enumerate(all_cells)}
    n_cells   = len(all_cells)

    date_range = pd.date_range(incidents_df["fecha"].min(), incidents_df["fecha"].max())
    date_to_i  = {d.date(): i for i, d in enumerate(date_range)}
    n_dates    = len(date_range)

    # Adjacencia ring-1 sobre celdas con incidentes (celdas sin incidentes aportan 0)
    print(f"  Adjacencia A1 sobre {n_cells:,} celdas...", end=" ", flush=True)
    rows, cols = [], []
    for i, cell in enumerate(all_cells):
        for nb in h3.grid_ring(cell, 1):
            j = cell_idx.get(nb)
            if j is not None:
                rows.append(i); cols.append(j)
    A1 = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                    shape=(n_cells, n_cells))
    print(f"{len(rows):,} conexiones")

    # Índices de celda/fecha para todos los incidentes
    inc_ci = incidents_df["h3_9"].map(cell_idx).values.astype("float32")
    inc_di = incidents_df["fecha"].dt.date.map(date_to_i)
    inc_di = pd.array(inc_di, dtype="Int64").to_numpy(na_value=-1, dtype=float)
    inc_ci = inc_ci.astype(int)
    inc_di = inc_di.astype(int)
    valid  = (inc_ci >= 0) & (inc_di >= 0)

    def causal_roll(count_mat: np.ndarray, W: int) -> np.ndarray:
        """Suma de los W días anteriores (sin incluir el día actual)."""
        cs = count_mat.cumsum(axis=1)
        r  = np.zeros_like(cs, dtype=np.float32)
        r[:, 1:] = cs[:, :-1]
        if W + 1 < n_dates:
            r[:, W + 1:] -= cs[:, : -W - 1]
        return r

    result = {}
    for cat, short in zip(CLASES, CAT_SHORTS):
        print(f"  Categoría '{short}'...", end=" ", flush=True)
        cat_mask  = (incidents_df["categoria"] == cat).values
        ci_cat    = inc_ci[cat_mask]
        di_cat    = inc_di[cat_mask]
        valid_cat = (ci_cat >= 0) & (di_cat >= 0)

        count_mat = np.zeros((n_cells, n_dates), dtype=np.float32)
        np.add.at(count_mat, (ci_cat[valid_cat], di_cat[valid_cat]), 1)

        for W, label in [(7, "7d"), (14, "14d")]:
            roll = causal_roll(count_mat, W)
            nr   = (A1 @ roll).astype(np.float32)
            feat = np.zeros(len(incidents_df), dtype=np.float32)
            feat[valid] = nr[inc_ci[valid], inc_di[valid]]
            result[f"nr_cat_{short}_ring1_{label}"] = feat
            del roll, nr

        del count_mat
        print("ok")

    return pd.DataFrame(result, index=incidents_df.index)


# ---------------------------------------------------------------------------
# Calibración isotónica
# ---------------------------------------------------------------------------

def fit_calibrators(y: np.ndarray, probs: np.ndarray) -> list:
    cals = []
    for i in range(4):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(probs[:, i], (y == i).astype(float))
        cals.append(ir)
    return cals


def apply_calibration(probs: np.ndarray, calibrators: list) -> np.ndarray:
    cal = np.column_stack([c.predict(probs[:, i]) for i, c in enumerate(calibrators)])
    cal = np.clip(cal, 1e-7, 1.0)
    return cal / cal.sum(axis=1, keepdims=True)


def compute_ece(y: np.ndarray, probs: np.ndarray, class_idx: int, n_bins: int = 10) -> float:
    p  = probs[:, class_idx]
    yb = (y == class_idx).astype(float)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1],
                      np.linspace(0, 1, n_bins + 1)[1:]):
        mask = (p >= lo) & (p <= hi) if hi == 1.0 else (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(y) * abs(yb[mask].mean() - p[mask].mean())
    return ece


# ---------------------------------------------------------------------------
# Helpers de evaluación
# ---------------------------------------------------------------------------

def apply_thresholds(probs: np.ndarray, mults: np.ndarray) -> np.ndarray:
    return (probs * mults).argmax(axis=1)


def print_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    header = f"{'':22}" + "".join(f"{c[:14]:>15}" for c in CLASES)
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for i, cls in enumerate(CLASES):
        row = f"{cls[:21]:22}" + "".join(f"{cm[i, j]:>15,}" for j in range(4))
        print(row)
    acc = np.diag(cm).sum() / cm.sum()
    print(f"\n  Accuracy: {acc:.3f}  |  Total: {cm.sum():,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Incidentes
    incidents = load_incidents()

    # 2. Near-repeats por categoría (solo necesita iieg_unified)
    cat_nr = build_cat_nearrepeat(incidents)

    # 3. Merge con features_daily
    df = merge_daily_features(incidents)

    # Unir near-repeats de categoría
    df = df.join(cat_nr.loc[df.index], how="left")
    cat_nr_cols = cat_nr.columns.tolist()

    # 4. Lambda diario de E2
    print("Prediciendo lambda_diario con E2...")
    e2    = pickle.load(open(E2_PKL, "rb"))
    X_e2  = df[E2_FEATURE_COLS].values.astype(float)
    df["lambda_diario"] = e2["model"].predict(X_e2)

    # 5. Features estáticas: DENUE (sin flags) e INV
    print("Cargando DENUE e INV...")
    denue     = pd.read_parquet(DENUE_PATH)
    denue_cols = [c for c in denue.columns if c != "h3_9" and not c.endswith("_flag")]
    denue      = denue[["h3_9"] + denue_cols]

    inv     = pd.read_parquet(INV_PATH)
    inv_cols = [c for c in inv.columns if c != "h3_9"]

    df = df.merge(denue, on="h3_9", how="left")
    df = df.merge(inv,   on="h3_9", how="left")
    df[denue_cols + inv_cols] = df[denue_cols + inv_cols].fillna(0)

    # 6. Feature cols de E3
    e3_feature_cols = E2_FEATURE_COLS + ["lambda_diario"] + cat_nr_cols + denue_cols + inv_cols

    # 7. Encode target
    le = LabelEncoder()
    le.fit(CLASES)
    df["target"] = le.transform(df["categoria"])

    # Splits
    train = df[df["fecha"] <= TRAIN_END]
    val   = df[(df["fecha"] > TRAIN_END) & (df["fecha"] <= VAL_END)]
    test  = df[df["fecha"] > VAL_END]
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    X_train = train[e3_feature_cols].values.astype(float)
    y_train = train["target"].values
    X_val   = val[e3_feature_cols].values.astype(float)
    y_val   = val["target"].values
    X_test  = test[e3_feature_cols].values.astype(float)
    y_test  = test["target"].values

    # 8. Entrenamiento
    print("\nEntrenando LightGBM multiclass (E3)...")
    params = {
        "objective":         "multiclass",
        "num_class":         4,
        "metric":            "multi_logloss",
        "learning_rate":     0.05,
        "num_leaves":        63,
        "min_child_samples": 30,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "seed":              42,
        "verbosity":         -1,
    }
    sample_weights = np.sqrt(compute_sample_weight("balanced", y_train))
    dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights,
                         feature_name=e3_feature_cols)
    dval   = lgb.Dataset(X_val, label=y_val,
                         feature_name=e3_feature_cols, reference=dtrain)

    booster = lgb.train(
        params, dtrain, num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)],
    )

    probs_val  = booster.predict(X_val)
    probs_test = booster.predict(X_test)

    # --- Evaluación base (sin calibración) ---
    print("\n--- BASE (sqrt weights, umbral uniforme) — TEST 2025 ---")
    print(classification_report(y_test, probs_test.argmax(axis=1), target_names=CLASES))
    print(f"Log-loss test (raw): {log_loss(y_test, probs_test):.4f}")

    # --- 9. Calibración isotónica ---
    print("\nCalibrando probabilidades (isotonic, ajuste en VAL 2024)...")
    calibrators = fit_calibrators(y_val, probs_val)

    print("ECE por clase — VAL 2024:")
    print(f"  {'Clase':<25} {'Antes':>8} {'Después':>9}")
    probs_val_cal = apply_calibration(probs_val, calibrators)
    for i, cls in enumerate(CLASES):
        ece_before = compute_ece(y_val, probs_val, i)
        ece_after  = compute_ece(y_val, probs_val_cal, i)
        print(f"  {cls:<25} {ece_before:>8.4f} {ece_after:>9.4f}")

    print(f"\nLog-loss VAL — antes calibración: {log_loss(y_val, probs_val):.4f}")
    print(f"Log-loss VAL — después:           {log_loss(y_val, probs_val_cal):.4f}")

    probs_test_cal = apply_calibration(probs_test, calibrators)

    # --- 10. Optimización de umbrales sobre VAL calibrado ---
    print("\nOptimizando multiplicadores de clase sobre VAL calibrado...")

    def neg_macro_f1(log_mults):
        mults = np.exp(log_mults)
        preds = apply_thresholds(probs_val_cal, mults)
        return -f1_score(y_val, preds, average="macro")

    best_score, best_mults = np.inf, np.ones(4)
    starts = [
        np.zeros(4),
        np.log([3.0, 1.5, 1.0, 0.7]),
        np.log([5.0, 2.0, 1.0, 0.6]),
        np.log([2.0, 1.5, 1.2, 0.8]),
    ]
    for x0 in starts:
        res = minimize(neg_macro_f1, x0, method="Nelder-Mead",
                       options={"maxiter": 800, "xatol": 1e-4, "fatol": 1e-4})
        if res.fun < best_score:
            best_score = res.fun
            best_mults = np.exp(res.x)

    best_mults = best_mults / best_mults.max()
    print("  Multiplicadores óptimos:")
    for cls, m in zip(CLASES, best_mults):
        print(f"    {cls:<25} {m:.4f}")
    print(f"  Macro-F1 VAL antes umbral:  {f1_score(y_val, probs_val_cal.argmax(axis=1), average='macro'):.4f}")
    print(f"  Macro-F1 VAL después umbral: {-best_score:.4f}")

    # --- Evaluación final ---
    preds_test = apply_thresholds(probs_test_cal, best_mults)
    preds_val  = apply_thresholds(probs_val_cal,  best_mults)

    print("\n=== EVALUACIÓN FINAL (sqrt weights + calibración + umbrales) ===")
    print("\n--- TEST 2025 ---")
    print(classification_report(y_test, preds_test, target_names=CLASES))
    print(f"Log-loss test (calibrado): {log_loss(y_test, probs_test_cal):.4f}")
    print_confusion_matrix(y_test, preds_test, "Matriz de confusión — TEST 2025")

    print("\n--- VAL 2024 ---")
    print(classification_report(y_val, preds_val, target_names=CLASES))
    print_confusion_matrix(y_val, preds_val, "Matriz de confusión — VAL 2024")

    # Feature importance (top 20)
    imp = pd.Series(booster.feature_importance("gain"), index=e3_feature_cols)
    print("\nTop 20 features por gain:")
    print(imp.sort_values(ascending=False).head(20).to_string())

    # --- Guardar ---
    artifact = {
        "model":        booster,
        "feature_cols": e3_feature_cols,
        "classes":      CLASES,
        "label_encoder": le,
        "calibrators":  calibrators,
        "thresholds":   best_mults,
    }
    with open(OUT_PKL, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nModelo guardado en {OUT_PKL}")


if __name__ == "__main__":
    main()
