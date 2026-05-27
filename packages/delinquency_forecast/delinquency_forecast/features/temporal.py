"""
Construcción vectorizada de features temporales para E1 (mensual) y E2 (diario).
Maneja fechas futuras usando promedios históricos del mismo mes como fallback.
"""
import numpy as np
import pandas as pd


def build_e1_features(
    h3_indexes: list[str],
    target_period: pd.Period,
    history: pd.DataFrame,
    encoders: dict,
) -> pd.DataFrame:
    """
    Devuelve DataFrame con las 22 features de E1, indexado por h3_index.
    history: columnas [h3_index, año_mes, conteo, zona_geografica, region, clave_mun, municipio]
    """
    hist = history.copy()
    hist["period"] = pd.PeriodIndex(pd.to_datetime(hist["año_mes"]), freq="M")

    pivot = (
        hist.pivot_table(index="h3_index", columns="period", values="conteo", aggfunc="sum")
        .reindex(h3_indexes)
        .fillna(0.0)
    )

    known_before = sorted(p for p in pivot.columns if p < target_period)

    same_month_avg: dict[int, pd.Series] = {}
    for m in range(1, 13):
        cols = [p for p in known_before if p.month == m]
        same_month_avg[m] = pivot[cols].mean(axis=1) if cols else pd.Series(0.0, index=pivot.index)

    idx = pd.Index(h3_indexes)

    def lag(n: int) -> np.ndarray:
        lp = target_period - n
        if lp in pivot.columns:
            return pivot[lp].reindex(idx).fillna(0.0).values
        return same_month_avg.get(lp.month, pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0).values

    def rolling_mean(n: int) -> np.ndarray:
        cols = known_before[-n:] if known_before else []
        return pivot[cols].mean(axis=1).reindex(idx).fillna(0.0).values if cols else np.zeros(len(h3_indexes))

    def rolling_std(n: int) -> np.ndarray:
        cols = known_before[-n:] if known_before else []
        if len(cols) > 1:
            return pivot[cols].std(axis=1).reindex(idx).fillna(0.0).values
        return np.zeros(len(h3_indexes))

    static = (
        hist.groupby("h3_index")
        .agg(
            zona_geografica=("zona_geografica", lambda x: x.mode().iloc[0] if len(x) else ""),
            region=("region", lambda x: x.mode().iloc[0] if len(x) else ""),
            clave_mun=("clave_mun", lambda x: str(int(x.mode().iloc[0])) if len(x) else "0"),
            municipio=("municipio", lambda x: x.mode().iloc[0] if len(x) else ""),
        )
        .reindex(h3_indexes)
        .fillna({"zona_geografica": "", "region": "", "clave_mun": "0", "municipio": ""})
    )

    def encode(col: str, vals: pd.Series) -> np.ndarray:
        enc = encoders.get(col)
        if enc is None:
            return np.zeros(len(vals), dtype=int)
        try:
            return enc.transform(vals.astype(str).values).astype(int)
        except Exception:
            return np.zeros(len(vals), dtype=int)

    mes = target_period.month
    año = target_period.year
    tri = pd.Series(str((mes - 1) // 3 + 1), index=idx).values
    tri_arr = pd.Series(str((mes - 1) // 3 + 1), index=idx)

    hist_mean_mes = same_month_avg.get(mes, pd.Series(0.0, index=pivot.index)).reindex(idx).fillna(0.0).values
    hist_max = pivot.max(axis=1).reindex(idx).fillna(0.0).values

    df = pd.DataFrame(
        {
            "h3_index":          h3_indexes,
            "municipio":         static["municipio"].values,
            "zona_geografica_str": static["zona_geografica"].values,
            "clave_mun":         encode("clave_mun", static["clave_mun"]),
            "zona_geografica":   encode("zona_geografica", static["zona_geografica"]),
            "region":            encode("region", static["region"]),
            "año":               año,
            "mes":               mes,
            "mes_sin":           np.sin(2 * np.pi * mes / 12),
            "mes_cos":           np.cos(2 * np.pi * mes / 12),
            "trend":             (año - 2017) * 12 + (mes - 1),
            "trimestre":         encode("trimestre", tri_arr),
            "es_fin_año":        int(mes in [11, 12]),
            "es_verano":         int(mes in [7, 8]),
            "lag_1":             lag(1),
            "lag_2":             lag(2),
            "lag_3":             lag(3),
            "lag_6":             lag(6),
            "lag_12":            lag(12),
            "rolling_mean_3":    rolling_mean(3),
            "rolling_mean_6":    rolling_mean(6),
            "rolling_mean_12":   rolling_mean(12),
            "rolling_std_3":     rolling_std(3),
            "hist_mean_mes":     hist_mean_mes,
            "hist_max":          hist_max,
        }
    ).set_index("h3_index")

    return df


def build_e2_temporal(
    h3_indexes: list[str],
    target_date: pd.Timestamp,
    history_daily: pd.DataFrame,
    encoders: dict,
) -> pd.DataFrame:
    """
    Devuelve DataFrame con lags diarios, rolling stats y categoricals de E2.
    history_daily: columnas [h3_index, fecha, conteo]
    No incluye log_lam_dia_base ni near-repeat (se añaden externamente).
    """
    hist = history_daily.copy()
    hist["fecha"] = pd.to_datetime(hist["fecha"])
    hist = hist[hist["fecha"] < target_date].copy()

    pivot = (
        hist.pivot_table(index="h3_index", columns="fecha", values="conteo", aggfunc="sum")
        .reindex(h3_indexes)
        .fillna(0.0)
    )

    dates_before = sorted(pivot.columns)

    def get_day(d: pd.Timestamp) -> np.ndarray:
        if d in pivot.columns:
            return pivot[d].values
        return np.zeros(len(h3_indexes))

    def roll_mean(n: int) -> np.ndarray:
        cols = dates_before[-n:] if dates_before else []
        return pivot[cols].mean(axis=1).values if cols else np.zeros(len(h3_indexes))

    def roll_std(n: int) -> np.ndarray:
        cols = dates_before[-n:] if dates_before else []
        if len(cols) > 1:
            return pivot[cols].std(axis=1).values
        return np.zeros(len(h3_indexes))

    # zona_geografica y clave_mun vienen del historial mensual (E1);
    # aquí se inicializan en 0 — el builder los sobreescribe desde E1.
    idx = pd.Index(h3_indexes)
    df = pd.DataFrame(
        {
            "h3_index":       h3_indexes,
            "lag_1":          get_day(target_date - pd.Timedelta(1, "D")),
            "lag_7":          get_day(target_date - pd.Timedelta(7, "D")),
            "lag_14":         get_day(target_date - pd.Timedelta(14, "D")),
            "rolling_mean_7":  roll_mean(7),
            "rolling_mean_14": roll_mean(14),
            "rolling_std_7":   roll_std(7),
        }
    ).set_index("h3_index")

    return df
