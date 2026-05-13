"""
Genera features para el modelo LightGBM diario (H3 × día).

Features clave:
  - log_lam_dia_base : log(λ_mensual / días_del_mes) — offset del modelo mensual
  - dia_semana, es_fin_semana, es_festivo, mes
  - lag_1, lag_7, lag_14 : conteos en días previos
  - rolling_mean_7, rolling_mean_14, rolling_std_7
  - zona_geografica, clave_mun (codificados con encoders del modelo mensual)

Usa operaciones matriciales vectorizadas para los lags y rolling stats:
el panel diario se reshapea a (n_h3, n_fechas) para operar con numpy puro
en lugar de groupby + transform, reduciendo el tiempo de cómputo ~10×.

Memoria estimada: 4–6 GB RAM durante la ejecución.
"""
import pickle
from pathlib import Path

import holidays as hol_lib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
MONTHLY_MODEL_PATH = ROOT / "models/lgbm_poisson_v1.pkl"
MONTHLY_FEATURES_PATH = ROOT / "data/processed/features_temporal.parquet"
DAILY_TS_PATH = ROOT / "data/processed/crime_timeseries_daily.parquet"
OUTPUT = ROOT / "data/processed/features_daily.parquet"

FEATURE_COLS_DAILY = [
    "log_lam_dia_base",
    "dia_semana", "es_fin_semana", "es_festivo",
    "mes",
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14",
    "rolling_std_7",
    "zona_geografica", "clave_mun",
]

CAT_ENCODE = ["zona_geografica", "region", "trimestre", "clave_mun"]


# ─── Festivos ────────────────────────────────────────────────────────────────

def build_holiday_set(years) -> set:
    mx = hol_lib.Mexico(years=list(years))
    return set(mx.keys())


# ─── Vectorized rolling helpers ───────────────────────────────────────────────

def causal_rolling_mean(matrix: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean causal (solo usa valores pasados) a lo largo del eje 1."""
    n, m = matrix.shape
    cs = np.zeros((n, m + 1), dtype=np.float32)
    cs[:, 1:] = np.cumsum(matrix.astype(np.float32), axis=1)
    j = np.arange(m)
    w = np.minimum(j + 1, window).astype(np.float32)
    start = np.maximum(0, j + 1 - window)
    return (cs[:, j + 1] - cs[:, start]) / w


def causal_rolling_std(matrix: np.ndarray, window: int, min_periods: int = 2) -> np.ndarray:
    """Rolling std causal con corrección de Bessel."""
    n, m = matrix.shape
    mat = matrix.astype(np.float64)
    cs  = np.zeros((n, m + 1), dtype=np.float64)
    cs[:, 1:]  = np.cumsum(mat, axis=1)
    cs2 = np.zeros((n, m + 1), dtype=np.float64)
    cs2[:, 1:] = np.cumsum(mat ** 2, axis=1)

    j = np.arange(m)
    w = np.minimum(j + 1, window).astype(np.float64)
    start = np.maximum(0, j + 1 - window)

    sum_x  = cs[:, j + 1]  - cs[:, start]
    sum_x2 = cs2[:, j + 1] - cs2[:, start]
    mean   = sum_x / w
    var    = np.maximum((sum_x2 / w) - mean ** 2, 0.0)
    std    = np.sqrt(var * w / np.maximum(w - 1, 1.0))
    std[:, :min_periods - 1] = 0.0
    return std.astype(np.float32)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Modelo mensual: calcular offset log(λ_m / días_mes) ──────────────────
    print("Cargando modelo mensual y calculando offset...")
    with open(MONTHLY_MODEL_PATH, "rb") as f:
        monthly_art = pickle.load(f)
    monthly_model   = monthly_art["model"]
    encoders        = monthly_art["encoders"]
    monthly_feat_cols = monthly_art["feature_cols"]

    mdf = pd.read_parquet(MONTHLY_FEATURES_PATH)
    for col in CAT_ENCODE:
        mdf[col] = encoders[col].transform(mdf[col].astype(str))

    lam_m = np.clip(monthly_model.predict(mdf[monthly_feat_cols].values), 1e-6, None)
    mdf["lambda_mensual"] = lam_m
    mdf["days_in_month"]  = pd.to_datetime(mdf["año_mes"]).dt.days_in_month
    mdf["log_lam_dia_base"] = np.log(mdf["lambda_mensual"] / mdf["days_in_month"])

    monthly_lookup = mdf[["h3_9", "año", "mes", "log_lam_dia_base"]].copy()

    # ── Serie temporal diaria ─────────────────────────────────────────────────
    print("Cargando crime_timeseries_daily.parquet...")
    panel = pd.read_parquet(DAILY_TS_PATH)
    panel = panel.sort_values(["h3_9", "fecha"]).reset_index(drop=True)
    print(f"  Shape: {panel.shape}")

    # ── Festivos ─────────────────────────────────────────────────────────────
    years = range(panel["fecha"].dt.year.min(), panel["fecha"].dt.year.max() + 1)
    holidays_set = build_holiday_set(years)
    panel["es_festivo"] = panel["fecha"].dt.date.map(
        lambda d: int(d in holidays_set)
    ).astype("int8")

    # ── Join offset mensual ───────────────────────────────────────────────────
    print("Uniendo offset mensual...")
    panel = panel.merge(monthly_lookup, on=["h3_9", "año", "mes"], how="left")
    panel["log_lam_dia_base"] = panel["log_lam_dia_base"].fillna(np.log(1e-6))

    # ── Codificar atributos estáticos (ya presentes en el panel desde el timeseries) ─
    # zona_geografica y clave_mun vienen como string/int originales; los codificamos
    # con los mismos encoders del modelo mensual para garantizar consistencia.
    known_zonas = set(encoders["zona_geografica"].classes_)
    fallback_zona = encoders["zona_geografica"].classes_[0]
    panel["zona_geografica"] = encoders["zona_geografica"].transform(
        panel["zona_geografica"].fillna(fallback_zona)
                                .map(lambda x: x if x in known_zonas else fallback_zona)
                                .astype(str)
    )

    known_clavemun = set(encoders["clave_mun"].classes_)
    fallback_clave = encoders["clave_mun"].classes_[0]
    panel["clave_mun"] = encoders["clave_mun"].transform(
        panel["clave_mun"].fillna(0)
                          .astype(int)
                          .astype(str)
                          .map(lambda x: x if x in known_clavemun else fallback_clave)
    )

    # ── Lags y rolling stats (operaciones matriciales) ────────────────────────
    all_h3_ordered = panel["h3_9"].unique()   # orden de sorted en build_timeseries
    n_h3    = len(all_h3_ordered)
    n_dates = len(panel) // n_h3
    assert len(panel) == n_h3 * n_dates, \
        f"Panel no es completo: {len(panel)} ≠ {n_h3} × {n_dates}"

    print(f"Calculando lags y rolling stats (matriz {n_h3:,} × {n_dates:,})...")
    conteo_mat = panel["conteo"].values.astype(np.float32).reshape(n_h3, n_dates)

    # Lags directos
    for lag, name in [(1, "lag_1"), (7, "lag_7"), (14, "lag_14")]:
        m = np.empty_like(conteo_mat)
        m[:, :lag] = np.nan
        m[:, lag:] = conteo_mat[:, :-lag]
        panel[name] = m.ravel()
        del m
        print(f"  {name} ✓")

    # Rolling sobre conteo desplazado un día (no incluye el día actual)
    shifted = np.empty_like(conteo_mat)
    shifted[:, 0]  = 0.0
    shifted[:, 1:] = conteo_mat[:, :-1]

    for w, name in [(7, "rolling_mean_7"), (14, "rolling_mean_14")]:
        panel[name] = causal_rolling_mean(shifted, w).ravel()
        print(f"  {name} ✓")

    panel["rolling_std_7"] = causal_rolling_std(shifted, 7).ravel()
    print("  rolling_std_7 ✓")

    del conteo_mat, shifted

    # ── Limpiar NaNs de lags ──────────────────────────────────────────────────
    lag_cols = [c for c in panel.columns if c.startswith(("lag_", "rolling_"))]
    panel[lag_cols] = panel[lag_cols].fillna(0)

    # ── Reporte ───────────────────────────────────────────────────────────────
    feature_check = [c for c in FEATURE_COLS_DAILY if c not in panel.columns]
    if feature_check:
        raise RuntimeError(f"Columnas faltantes: {feature_check}")

    print(f"\nShape final: {panel.shape}")
    print(f"Columnas: {list(panel.columns)}")

    train = panel[panel["año"] <= 2023]
    val   = panel[panel["año"] == 2024]
    test  = panel[panel["año"] >= 2025]
    print(f"\nSplit temporal:")
    print(f"  Train (≤2023): {len(train):>12,} filas")
    print(f"  Val   (2024):  {len(val):>12,} filas")
    print(f"  Test  (2025+): {len(test):>12,} filas")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT, index=False)
    print(f"\nGuardado en {OUTPUT}")


if __name__ == "__main__":
    main()
