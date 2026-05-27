"""
Inferencia batch para renderizado de mapa.

API pública:
    predict_batch(fecha, h3_indexes=None, municipio=None) -> pd.DataFrame
"""
import pickle
from pathlib import Path
from typing import Optional

import h3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent

E2_PKL     = ROOT / "models/lgbm_daily_v1.pkl"
E3_PKL     = ROOT / "models/lgbm_e3_v1.pkl"
FEAT_PATH  = ROOT / "data/processed/features_daily.parquet"
DENUE_PATH = ROOT / "data/processed/denue_h3.parquet"
INV_PATH   = ROOT / "data/processed/inegi_inv_h3.parquet"
IIEG_PATH  = ROOT / "data/processed/iieg_unified.parquet"

E2_FEATURE_COLS = [
    "log_lam_dia_base", "dia_semana", "es_fin_semana", "es_festivo", "mes",
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    "zona_geografica", "clave_mun",
    "nr_ring1_1d", "nr_ring1_7d", "nr_ring1_14d", "nr_ring2_7d",
]

FEAT_COLS_TO_LOAD = [
    "h3_9", "fecha", "municipio", "zona_geografica", "clave_mun",
] + [c for c in E2_FEATURE_COLS if c not in ("zona_geografica", "clave_mun")]

CLASES     = ["alto_impacto", "violencia_personal", "robo_confrontacion", "robo_patrimonial"]
CAT_SHORTS = ["alto", "viol", "conf", "patr"]

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

TODAY    = pd.Timestamp.now().normalize()
DATA_END = pd.Timestamp("2025-12-30")   # último día en features_daily.parquet

# ---------------------------------------------------------------------------
# Cache de modelos y datos estáticos
# ---------------------------------------------------------------------------
_E2, _E3 = None, None
_DENUE, _INV = None, None


def _ensure_loaded():
    global _E2, _E3, _DENUE, _INV
    if _E2 is None:
        _E2 = pickle.load(open(E2_PKL, "rb"))
        _E3 = pickle.load(open(E3_PKL, "rb"))
    if _DENUE is None:
        _DENUE = pd.read_parquet(DENUE_PATH)
        _INV   = pd.read_parquet(INV_PATH)


# ---------------------------------------------------------------------------
# Near-repeats por categoría (inferencia)
# ---------------------------------------------------------------------------

def _compute_cat_nearrepeat(df: pd.DataFrame, fecha: pd.Timestamp) -> pd.DataFrame:
    """
    Para cada celda en df, calcula near-repeats de vecinos ring-1 por categoría.
    Carga los últimos 14 días de iieg_unified para la fecha dada.
    """
    start = fecha - pd.Timedelta(15, "D")
    table = pq.read_table(
        IIEG_PATH,
        filters=[("fecha", ">=", start), ("fecha", "<", fecha)],
        columns=["h3_9", "fecha", "delito"],
    )
    recent = table.to_pandas()

    # Inicializar todas las features en 0
    zero_cols = {
        f"nr_cat_{s}_ring1_{w}d": 0.0
        for s in CAT_SHORTS for w in [7, 14]
    }

    if recent.empty:
        return df.assign(**zero_cols)

    recent["categoria"] = recent["delito"].map(DELITO_CATEGORIA)
    recent = recent.dropna(subset=["categoria"])
    recent["fecha"] = pd.to_datetime(recent["fecha"])

    # Vecinos ring-1 para cada celda del batch
    cells = df["h3_9"].tolist()
    ring1 = {cell: h3.grid_ring(cell, 1) for cell in cells}

    for short, cat in zip(CAT_SHORTS, CLASES):
        cat_inc = recent[recent["categoria"] == cat]
        for window in [7, 14]:
            cutoff   = fecha - pd.Timedelta(window, "D")
            cat_w    = cat_inc[cat_inc["fecha"] >= cutoff].groupby("h3_9").size()
            col_name = f"nr_cat_{short}_ring1_{window}d"
            df[col_name] = [
                float(sum(cat_w.get(nb, 0) for nb in ring1.get(cell, [])))
                for cell in cells
            ]

    return df


# ---------------------------------------------------------------------------
# Calibración
# ---------------------------------------------------------------------------

def _apply_calibration(probs: np.ndarray, calibrators: list) -> np.ndarray:
    cal = np.column_stack([c.predict(probs[:, i]) for i, c in enumerate(calibrators)])
    cal = np.clip(cal, 1e-7, 1.0)
    return cal / cal.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Features sintéticas para fechas futuras
# ---------------------------------------------------------------------------

def _load_future_features(
    fecha: pd.Timestamp,
    h3_indexes: Optional[list],
    municipio: Optional[str],
) -> pd.DataFrame:
    """
    Para fechas sin datos en features_daily, genera features usando el promedio
    del mismo mes en el año de referencia más reciente disponible.
    Las features calendáricas se sobreescriben con los valores reales de `fecha`.
    """
    target_month = fecha.month
    static_cols = ["municipio", "zona_geografica", "clave_mun"]
    avg_cols = [c for c in FEAT_COLS_TO_LOAD
                if c not in ["h3_9", "fecha", "municipio", "zona_geografica", "clave_mun",
                             "dia_semana", "es_fin_semana", "es_festivo", "mes"]]

    ref_df = pd.DataFrame()
    for ref_year in [2024, 2023, 2022]:
        start = pd.Timestamp(f"{ref_year}-{target_month:02d}-01")
        end   = start + pd.offsets.MonthEnd(1)
        try:
            table = pq.read_table(
                FEAT_PATH,
                columns=FEAT_COLS_TO_LOAD,
                filters=[("fecha", ">=", start), ("fecha", "<=", end)],
            )
            ref_df = table.to_pandas()
            if not ref_df.empty:
                break
        except Exception:
            continue

    if ref_df.empty:
        return pd.DataFrame()

    if h3_indexes is not None:
        ref_df = ref_df[ref_df["h3_9"].isin(set(h3_indexes))]
    elif municipio is not None:
        ref_df = ref_df[ref_df["municipio"].str.lower().str.contains(municipio.lower(), na=False)]

    if ref_df.empty:
        return pd.DataFrame()

    static_df = ref_df.groupby("h3_9")[static_cols].first().reset_index()
    avg_df    = ref_df.groupby("h3_9")[avg_cols].mean().reset_index()
    df = static_df.merge(avg_df, on="h3_9")

    df["fecha"]        = fecha
    df["dia_semana"]   = fecha.dayofweek
    df["es_fin_semana"] = int(fecha.dayofweek >= 5)
    df["es_festivo"]   = 0
    df["mes"]          = fecha.month

    return df


# ---------------------------------------------------------------------------
# Nivel de riesgo
# ---------------------------------------------------------------------------

def _nivel_riesgo(lam: np.ndarray, p33: float, p66: float) -> list[str]:
    return [
        "bajo" if v <= p33 else "medio" if v <= p66 else "alto"
        for v in lam
    ]


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def predict_batch(
    fecha: str,
    h3_indexes: Optional[list] = None,
    municipio: Optional[str] = None,
) -> pd.DataFrame:
    """
    Retorna predicciones E2 + E3 para una fecha dada.

    Args:
        fecha:      'YYYY-MM-DD'
        h3_indexes: lista de H3 indexes; si None, usa todos los activos del día.
        municipio:  filtra por nombre de municipio (ignorado si h3_indexes != None).

    Returns:
        DataFrame con columnas:
            h3_index, municipio, zona_geografica, lambda_diario,
            categoria_pred, p_alto_impacto, p_violencia_personal,
            p_robo_confrontacion, p_robo_patrimonial,
            nivel_riesgo, confiabilidad_score
    """
    _ensure_loaded()
    ts = pd.Timestamp(fecha)
    days_ahead = max(0, (ts - TODAY).days)

    # --- Features del día desde features_daily ---
    table = pq.read_table(FEAT_PATH, columns=FEAT_COLS_TO_LOAD,
                           filters=[("fecha", "=", ts)])
    df = table.to_pandas()

    # Para fechas futuras, generar features sintéticas del mismo mes histórico
    if df.empty and days_ahead > 0:
        df = _load_future_features(ts, h3_indexes, municipio)
        if df.empty:
            return pd.DataFrame()
    else:
        if h3_indexes is not None:
            df = df[df["h3_9"].isin(set(h3_indexes))]
        elif municipio is not None:
            df = df[df["municipio"].str.lower().str.contains(municipio.lower(), na=False)]

    if df.empty:
        return pd.DataFrame()

    # --- E2: lambda_diario ---
    X_e2 = df[E2_FEATURE_COLS].values.astype(float)
    df["lambda_diario"] = _E2["model"].predict(X_e2)

    # --- Near-repeats por categoría ---
    df = _compute_cat_nearrepeat(df, ts)

    # --- Features estáticas DENUE (sin flags) e INV ---
    poi_cols = [c for c in _DENUE.columns if c != "h3_9" and not c.endswith("_flag")]
    inv_cols = [c for c in _INV.columns   if c != "h3_9"]

    df = df.merge(_DENUE[["h3_9"] + poi_cols], on="h3_9", how="left")
    df = df.merge(_INV,                         on="h3_9", how="left")
    df[poi_cols + inv_cols] = df[poi_cols + inv_cols].fillna(0)

    # --- E3: composición ---
    e3_feat_cols = _E3["feature_cols"]
    # Asegurar que todas las features existan (rellena con 0 si faltan)
    for col in e3_feat_cols:
        if col not in df.columns:
            df[col] = 0.0

    X_e3  = df[e3_feat_cols].values.astype(float)
    probs = _E3["model"].predict(X_e3)

    # Calibración isotónica
    calibrators = _E3.get("calibrators")
    if calibrators is not None:
        probs = _apply_calibration(probs, calibrators)

    for i, cls in enumerate(CLASES):
        df[f"p_{cls}"] = probs[:, i]

    # Multiplicadores de clase (threshold tuning)
    mults = _E3.get("thresholds", np.ones(4))
    df["categoria_pred"] = [CLASES[i] for i in (probs * mults).argmax(axis=1)]

    # --- Nivel de riesgo ---
    lam = df["lambda_diario"].values
    pos = lam[lam > 0]
    p33, p66 = (np.percentile(pos, [33, 66]) if len(pos) > 0 else (1.0, 3.0))
    df["nivel_riesgo"] = _nivel_riesgo(lam, p33, p66)

    # --- Probabilidad Poisson de al menos un crimen ---
    # P(X >= 1) = 1 - e^(-λ), válido bajo supuesto Poisson del modelo E2.
    # Interpreta: probabilidad de que se reporte >= 1 crimen en esta celda hoy.
    df["prob_crimen"] = (1 - np.exp(-lam.clip(min=0))) * 100

    # --- Confiabilidad ---
    # La incertidumbre no viene de cuántos días faltan para la fecha objetivo,
    # sino de qué tan lejos está del último dato real (DATA_END = 2025-12-30).
    # Para fechas futuras los lags, rolling y near-repeats son promedios históricos
    # del mismo mes, no conteos reales — eso es la fuente principal de incertidumbre.
    days_from_data_end = max(0, (ts - DATA_END).days)
    if days_from_data_end == 0:
        conf_penalty = 0          # datos reales del parquet
    elif days_from_data_end <= 31:
        conf_penalty = 25         # sintético, muy cerca del fin de datos
    elif days_from_data_end <= 180:
        conf_penalty = 35         # sintético, meses después
    else:
        conf_penalty = 45         # sintético, largo plazo

    df["confiabilidad_score"] = (
        70
        + (df.get("poi_total", pd.Series(0, index=df.index)) > 0).astype(int) * 10
        + (df.get("inv_n_segmentos", pd.Series(0, index=df.index)) > 0).astype(int) * 10
        + (df["log_lam_dia_base"] > -5).astype(int) * 10
        - conf_penalty
    ).clip(0, 100)

    return df.rename(columns={"h3_9": "h3_index"})[[
        "h3_index", "municipio", "zona_geografica", "lambda_diario",
        "prob_crimen",
        "categoria_pred",
        "p_alto_impacto", "p_violencia_personal",
        "p_robo_confrontacion", "p_robo_patrimonial",
        "nivel_riesgo", "confiabilidad_score",
    ]]
