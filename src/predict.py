"""
Carga los modelos entrenados y genera predicciones de conteo de delitos
para una fecha y ubicación dadas.

Fechas mensuales  → solo modelo mensual (Etapa 1)
Fechas diarias    → modelo mensual como base + modelo diario (Etapa 2)

Uso:
    python predict.py --fecha 2025-06    --lat 20.6597 --lon -103.3496
    python predict.py --fecha 2025-06-15 --lat 20.6597 --lon -103.3496
    python predict.py --fecha 2025-06    --h3 8965c0a6fffffff
    python predict.py --fecha 2025-06-15 --h3 8965c0a6fffffff
"""
import argparse
import pickle
from datetime import date
from pathlib import Path

import h3
import holidays as hol_lib
import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).parent.parent

MODEL_MONTHLY_PATH = ROOT / "models/lgbm_poisson_v1.pkl"
MODEL_DAILY_PATH   = ROOT / "models/lgbm_daily_v1.pkl"
HISTORY_MONTHLY    = ROOT / "data/processed/features_temporal.parquet"
HISTORY_DAILY      = ROOT / "data/processed/crime_timeseries_daily.parquet"

JALISCO_BBOX = dict(lon_min=-106.0, lon_max=-101.0, lat_min=18.5, lat_max=23.0)

_MX_HOLIDAYS: set | None = None


def _get_holidays() -> set:
    global _MX_HOLIDAYS
    if _MX_HOLIDAYS is None:
        years = range(2017, 2031)
        mx = hol_lib.Mexico(years=list(years))
        _MX_HOLIDAYS = set(mx.keys())
    return _MX_HOLIDAYS


# ─── Utilidades ──────────────────────────────────────────────────────────────

def validate_coords(lat: float, lon: float):
    bb = JALISCO_BBOX
    if not (bb["lat_min"] <= lat <= bb["lat_max"]):
        raise ValueError(f"Latitud {lat} fuera del rango de Jalisco ({bb['lat_min']}–{bb['lat_max']})")
    if not (bb["lon_min"] <= lon <= bb["lon_max"]):
        raise ValueError(f"Longitud {lon} fuera del rango de Jalisco ({bb['lon_min']}–{bb['lon_max']})")


def resolve_h3(lat, lon, h3_index) -> str:
    if h3_index is not None:
        return h3_index
    if lat is None or lon is None:
        raise ValueError("Se requiere lat+lon o h3_index")
    validate_coords(lat, lon)
    return h3.latlng_to_cell(lat, lon, 9)


def load_monthly_artifacts():
    with open(MODEL_MONTHLY_PATH, "rb") as f:
        return pickle.load(f)


def load_daily_artifacts():
    with open(MODEL_DAILY_PATH, "rb") as f:
        return pickle.load(f)


# ─── Feature vectors ─────────────────────────────────────────────────────────

def build_monthly_feature_vector(
    h3_index: str,
    target_period: pd.Period,
    history: pd.DataFrame,
    encoders: dict,
    feature_cols: list,
) -> np.ndarray:
    cell_hist = history[history["h3_9"] == h3_index].sort_values("año_mes")
    known = cell_hist[cell_hist["año_mes"] < target_period.to_timestamp()]

    def get_lag(n):
        lag_period = target_period - n
        row = known[known["año_mes"] == lag_period.to_timestamp()]
        return float(row["conteo"].iloc[0]) if not row.empty else 0.0

    def rolling_mean(window):
        recent = known.tail(window)["conteo"]
        return recent.mean() if len(recent) > 0 else 0.0

    def rolling_std(window):
        recent = known.tail(window)["conteo"]
        return recent.std() if len(recent) > 1 else 0.0

    if not cell_hist.empty:
        row_static = cell_hist.iloc[0]
        zona     = encoders["zona_geografica"].transform([str(row_static["zona_geografica"])])[0]
        region   = encoders["region"].transform([str(row_static["region"])])[0]
        clave_mun = encoders["clave_mun"].transform([str(int(row_static["clave_mun"]))])[0]
    else:
        zona = region = clave_mun = 0

    mes = target_period.month
    año = target_period.year
    min_year, min_month = 2017, 1
    trend    = (año - min_year) * 12 + (mes - min_month)
    trimestre = encoders["trimestre"].transform([str((mes - 1) // 3 + 1)])[0]

    same_month   = known[pd.DatetimeIndex(known["año_mes"]).month == mes]["conteo"]
    hist_mean_mes = same_month.mean() if len(same_month) > 0 else 0.0
    hist_max      = known["conteo"].max() if len(known) > 0 else 0.0

    features = {
        "año": año, "mes": mes,
        "mes_sin": np.sin(2 * np.pi * mes / 12),
        "mes_cos": np.cos(2 * np.pi * mes / 12),
        "trend": trend, "trimestre": trimestre,
        "es_fin_año": int(mes in [11, 12]),
        "es_verano":  int(mes in [7, 8]),
        "zona_geografica": zona, "region": region, "clave_mun": clave_mun,
        "lag_1":  get_lag(1),  "lag_2":  get_lag(2),
        "lag_3":  get_lag(3),  "lag_6":  get_lag(6), "lag_12": get_lag(12),
        "rolling_mean_3":  rolling_mean(3),
        "rolling_mean_6":  rolling_mean(6),
        "rolling_mean_12": rolling_mean(12),
        "rolling_std_3":   rolling_std(3),
        "hist_mean_mes": hist_mean_mes,
        "hist_max":      hist_max,
    }
    return np.array([features[c] for c in feature_cols]).reshape(1, -1)


def _near_repeat_sums(
    h3_index: str,
    target_date: pd.Timestamp,
) -> dict:
    """Carga el historial de celdas vecinas y calcula sumas near-repeat."""
    ring1 = [nb for nb in h3.grid_disk(h3_index, 1) if nb != h3_index]
    ring2 = list(h3.grid_ring(h3_index, 2))
    all_nb = ring1 + ring2

    cutoff = target_date - pd.Timedelta(days=14)
    try:
        nb_hist = pd.read_parquet(
            HISTORY_DAILY,
            filters=[("h3_9", "in", all_nb)],
            columns=["h3_9", "fecha", "conteo"],
        )
        nb_hist["fecha"] = pd.to_datetime(nb_hist["fecha"])
        nb_hist = nb_hist[
            (nb_hist["fecha"] >= cutoff) & (nb_hist["fecha"] < target_date)
        ]
    except Exception:
        nb_hist = pd.DataFrame(columns=["h3_9", "fecha", "conteo"])

    def nb_sum(cells, days):
        cutoff_d = target_date - pd.Timedelta(days=days)
        s = nb_hist[nb_hist["h3_9"].isin(cells) & (nb_hist["fecha"] >= cutoff_d)]
        return float(s["conteo"].sum())

    return {
        "nr_ring1_1d":  nb_sum(ring1, 1),
        "nr_ring1_7d":  nb_sum(ring1, 7),
        "nr_ring1_14d": nb_sum(ring1, 14),
        "nr_ring2_7d":  nb_sum(ring2, 7),
    }


def build_daily_feature_vector(
    h3_index: str,
    target_date: pd.Timestamp,
    monthly_lam: float,
    daily_history: pd.DataFrame,
    encoders: dict,
    feature_cols: list,
) -> np.ndarray:
    days_in_month = target_date.days_in_month
    log_lam_dia_base = np.log(max(monthly_lam / days_in_month, 1e-6))

    dia_semana    = target_date.dayofweek
    es_fin_semana = int(dia_semana >= 5)
    es_festivo    = int(target_date.date() in _get_holidays())
    mes = target_date.month

    known = daily_history[daily_history["fecha"] < target_date].sort_values("fecha")

    def get_daily_lag(n):
        target = target_date - pd.Timedelta(days=n)
        row = known[known["fecha"] == target]
        return float(row["conteo"].iloc[0]) if not row.empty else 0.0

    def rolling_daily_mean(window):
        recent = known.tail(window)["conteo"]
        return float(recent.mean()) if len(recent) > 0 else 0.0

    def rolling_daily_std(window):
        recent = known.tail(window)["conteo"]
        return float(recent.std()) if len(recent) > 1 else 0.0

    if not daily_history.empty and "zona_geografica" in daily_history.columns:
        zona_raw  = str(daily_history["zona_geografica"].mode().iloc[0])
        clave_raw = str(int(daily_history["clave_mun"].mode().iloc[0]))
        try:
            zona      = encoders["zona_geografica"].transform([zona_raw])[0]
            clave_mun = encoders["clave_mun"].transform([clave_raw])[0]
        except Exception:
            zona = clave_mun = 0
    else:
        zona = clave_mun = 0

    nr = _near_repeat_sums(h3_index, target_date)

    features = {
        "log_lam_dia_base": log_lam_dia_base,
        "dia_semana":    dia_semana,
        "es_fin_semana": es_fin_semana,
        "es_festivo":    es_festivo,
        "mes":           mes,
        "lag_1":  get_daily_lag(1),
        "lag_7":  get_daily_lag(7),
        "lag_14": get_daily_lag(14),
        "rolling_mean_7":  rolling_daily_mean(7),
        "rolling_mean_14": rolling_daily_mean(14),
        "rolling_std_7":   rolling_daily_std(7),
        "zona_geografica": zona,
        "clave_mun":       clave_mun,
        **nr,
    }
    return np.array([features[c] for c in feature_cols]).reshape(1, -1)


# ─── Niveles de riesgo y confianza ────────────────────────────────────────────

def compute_risk_level(lam: float, p33: float, p66: float) -> str:
    if p66 == 0:
        p33, p66 = 1.0, 3.0
    if lam <= p33:
        return "bajo"
    elif lam <= p66:
        return "medio"
    return "alto"


def compute_confidence_score(
    n_months: int,
    zero_rate: float,
    brecha_meses: int,
    cv: float | None,
) -> tuple[int, str, str]:
    """
    Devuelve (score 0-100, nivel, nota) para usuarios no técnicos.
    Penaliza falta de historial, celdas silenciosas, extrapolación temporal y volatilidad.
    """
    score = 100

    if n_months == 0:
        score -= 50
    elif n_months < 6:
        score -= 30
    elif n_months < 24:
        score -= 10

    if zero_rate > 0.90:
        score -= 25
    elif zero_rate > 0.80:
        score -= 10

    if brecha_meses > 24:
        score -= 20
    elif brecha_meses > 12:
        score -= 10
    elif brecha_meses > 6:
        score -= 5

    if cv is not None and cv > 2.0:
        score -= 10

    score = max(0, min(100, score))

    if score >= 65:
        nivel = "CONFIABLE"
        nota  = f"Basada en {n_months} meses de historial para esta zona"
    elif score >= 35:
        nivel = "APROXIMADA"
        nota  = f"Datos limitados — interpretar el rango con cautela"
    else:
        nivel = "ESPECULATIVA"
        nota  = "Muy pocos antecedentes — usar solo como referencia"

    if brecha_meses > 12:
        nota += f" · {brecha_meses} meses de extrapolación"

    return score, nivel, nota


# ─── Predicción mensual ───────────────────────────────────────────────────────

def predict_monthly(
    fecha: str,
    lat: float = None,
    lon: float = None,
    h3_index: str = None,
) -> dict:
    h3_index = resolve_h3(lat, lon, h3_index)
    target_period = pd.Period(fecha, freq="M")

    art = load_monthly_artifacts()
    model, encoders, feature_cols = art["model"], art["encoders"], art["feature_cols"]

    history = pd.read_parquet(HISTORY_MONTHLY, columns=[
        "h3_9", "año_mes", "conteo", "zona_geografica",
        "region", "clave_mun", "municipio",
    ])

    cell_hist = history[history["h3_9"] == h3_index].copy()
    cell_hist["año_mes"] = pd.to_datetime(cell_hist["año_mes"])

    X = build_monthly_feature_vector(h3_index, target_period, cell_hist, encoders, feature_cols)
    lam = float(np.clip(model.predict(X)[0], 0, None))

    ci_80 = (int(poisson.ppf(0.10, lam)), int(poisson.ppf(0.90, lam)))
    ci_95 = (int(poisson.ppf(0.025, lam)), int(poisson.ppf(0.975, lam)))

    known = cell_hist[cell_hist["año_mes"] < target_period.to_timestamp()]
    n_months  = len(known)
    zero_rate = (known["conteo"] == 0).mean() if n_months > 0 else 1.0
    hist_cv   = (
        float(known["conteo"].std() / known["conteo"].mean())
        if n_months > 1 and known["conteo"].mean() > 0 else None
    )

    # Brecha temporal: meses entre último dato de entrenamiento y fecha solicitada
    max_train_date = pd.Timestamp("2023-12-01")
    brecha_meses = max(
        0,
        (target_period.to_timestamp() - max_train_date).days // 30,
    )

    p33 = known["conteo"].quantile(0.33) if n_months >= 6 else 1.0
    p66 = known["conteo"].quantile(0.66) if n_months >= 6 else 3.0
    risk = compute_risk_level(lam, p33, p66)

    conf_score, conf_nivel, conf_nota = compute_confidence_score(
        n_months, zero_rate, brecha_meses, hist_cv
    )

    municipio = cell_hist["municipio"].mode().iloc[0] if not cell_hist.empty else "Desconocido"
    zona      = cell_hist["zona_geografica"].mode().iloc[0] if not cell_hist.empty else "Desconocido"

    return {
        "granularidad":       "mensual",
        "fecha":              str(target_period),
        "h3_index":           h3_index,
        "municipio":          municipio,
        "zona_geografica":    zona,
        "prediccion":         round(lam, 2),
        "rango_probable":     ci_80,
        "rango_amplio_95pct": ci_95,
        "nivel_riesgo":       risk,
        "confiabilidad_score": conf_score,
        "confiabilidad_nivel": conf_nivel,
        "confiabilidad_nota":  conf_nota,
        "cv_historico":       round(hist_cv, 2) if hist_cv is not None else None,
        "meses_historial":    n_months,
        "brecha_temporal_meses": brecha_meses,
        "_cell_hist":         cell_hist,  # interno, para reutilizar en predict_daily
        "_h3_index":          h3_index,
    }


# ─── Predicción diaria ────────────────────────────────────────────────────────

def predict_daily(
    fecha: str,
    lat: float = None,
    lon: float = None,
    h3_index: str = None,
) -> dict:
    target_date = pd.Timestamp(fecha)
    target_month = target_date.strftime("%Y-%m")

    # Etapa 1: predicción mensual base
    monthly = predict_monthly(target_month, lat, lon, h3_index)
    lam_m   = monthly["prediccion"]
    h3_index = monthly["_h3_index"]

    # Etapa 2: predicción diaria
    daily_art = load_daily_artifacts()
    daily_model, encoders, daily_feat_cols = (
        daily_art["model"], daily_art["encoders"], daily_art["feature_cols"]
    )

    daily_history = pd.read_parquet(
        HISTORY_DAILY,
        filters=[("h3_9", "==", h3_index)],
    )
    daily_history["fecha"] = pd.to_datetime(daily_history["fecha"])

    X_day = build_daily_feature_vector(
        h3_index, target_date, lam_m, daily_history, encoders, daily_feat_cols
    )
    lam_day = float(np.clip(daily_model.predict(X_day)[0], 0, None))

    ci_80_day = (int(poisson.ppf(0.10, lam_day)), int(poisson.ppf(0.90, lam_day)))
    ci_95_day = (int(poisson.ppf(0.025, lam_day)), int(poisson.ppf(0.975, lam_day)))

    # Brecha temporal diaria: días entre último dato de entrenamiento y fecha solicitada
    max_train_date = pd.Timestamp("2023-12-31")
    brecha_dias = max(0, (target_date - max_train_date).days)

    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    es_festivo  = int(target_date.date() in _get_holidays())

    return {
        "granularidad":        "diaria",
        "fecha":               fecha,
        "dia_semana":          dias_semana[target_date.dayofweek],
        "es_festivo":          bool(es_festivo),
        "h3_index":            h3_index,
        "municipio":           monthly["municipio"],
        "zona_geografica":     monthly["zona_geografica"],
        "prediccion":          round(lam_day, 3),
        "rango_probable":      ci_80_day,
        "rango_amplio_95pct":  ci_95_day,
        "prediccion_mensual":  monthly["prediccion"],
        "nivel_riesgo":        monthly["nivel_riesgo"],
        "confiabilidad_score": monthly["confiabilidad_score"],
        "confiabilidad_nivel": monthly["confiabilidad_nivel"],
        "confiabilidad_nota":  monthly["confiabilidad_nota"],
        "cv_historico":        monthly["cv_historico"],
        "meses_historial":     monthly["meses_historial"],
        "brecha_temporal_dias": brecha_dias,
    }


# ─── Predicción unificada ─────────────────────────────────────────────────────

def predict_single(
    fecha: str,
    lat: float = None,
    lon: float = None,
    h3_index: str = None,
) -> dict:
    if len(fecha) == 7:       # YYYY-MM
        r = predict_monthly(fecha, lat, lon, h3_index)
        r.pop("_cell_hist", None)
        r.pop("_h3_index", None)
        return r
    elif len(fecha) == 10:    # YYYY-MM-DD
        return predict_daily(fecha, lat, lon, h3_index)
    else:
        raise ValueError(f"Formato de fecha no reconocido: '{fecha}'. Use YYYY-MM o YYYY-MM-DD")


# ─── Impresión ────────────────────────────────────────────────────────────────

def _confiabilidad_bar(score: int) -> str:
    filled = round(score / 20)   # 0-5 bloques
    return "█" * filled + "░" * (5 - filled)


def print_result(r: dict):
    es_diario = r["granularidad"] == "diaria"
    titulo = "PRONÓSTICO DIARIO" if es_diario else "PRONÓSTICO MENSUAL"

    print("\n" + "=" * 58)
    print(f"  {titulo} DE DELINCUENCIA")
    print("=" * 58)
    print(f"  Fecha:        {r['fecha']}", end="")
    if es_diario:
        festivo = " (festivo)" if r["es_festivo"] else ""
        print(f"  [{r['dia_semana']}{festivo}]")
    else:
        print()
    print(f"  H3 (res=9):   {r['h3_index']}")
    print(f"  Municipio:    {r['municipio']}")
    print(f"  Zona:         {r['zona_geografica']}")
    print("-" * 58)

    unidad = "delitos ese día" if es_diario else "delitos ese mes"
    print(f"  Delitos esperados:  {r['prediccion']:.2f}  ({unidad})")
    print(f"  Rango probable:     entre {r['rango_probable'][0]} y {r['rango_probable'][1]}")
    print(f"  Rango amplio 95%:   entre {r['rango_amplio_95pct'][0]} y {r['rango_amplio_95pct'][1]}")

    if es_diario:
        print(f"  Base mensual:       {r['prediccion_mensual']:.2f} delitos/mes")

    print("-" * 58)
    print(f"  Nivel de riesgo:    {r['nivel_riesgo'].upper()}")
    bar = _confiabilidad_bar(r['confiabilidad_score'])
    print(f"  Fiabilidad:         {bar} {r['confiabilidad_nivel']}  ({r['confiabilidad_score']}/100)")
    print(f"  {r['confiabilidad_nota']}")

    if r.get("cv_historico") is not None:
        print(f"  Variabilidad hist.: {r['cv_historico']:.2f}")

    brecha_key = "brecha_temporal_dias" if es_diario else "brecha_temporal_meses"
    unidad_brecha = "días" if es_diario else "meses"
    brecha = r.get(brecha_key, 0)
    if brecha > 0:
        print(f"  Extrapolación:      {brecha} {unidad_brecha} fuera del período de entrenamiento")

    print(f"  Meses de historial: {r['meses_historial']}")
    print("=" * 58 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predicción de delincuencia por H3 × fecha (mensual o diaria)"
    )
    parser.add_argument("--fecha", required=True,
                        help="Fecha a predecir: YYYY-MM (mensual) o YYYY-MM-DD (diaria)")
    parser.add_argument("--lat",  type=float, help="Latitud")
    parser.add_argument("--lon",  type=float, help="Longitud")
    parser.add_argument("--h3",   dest="h3_index", help="Índice H3 resolución 9")
    args = parser.parse_args()

    resultado = predict_single(
        fecha=args.fecha,
        lat=args.lat,
        lon=args.lon,
        h3_index=args.h3_index,
    )
    print_result(resultado)


if __name__ == "__main__":
    main()
