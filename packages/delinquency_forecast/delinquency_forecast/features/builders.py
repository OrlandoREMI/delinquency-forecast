"""
Orquesta la construcción de features para los tres stages.
Detecta fechas futuras y aplica fallback con promedios históricos del mismo mes.
"""
import numpy as np
import pandas as pd

from .calendar import date_features
from .temporal import build_e1_features, build_e2_temporal
from .nearrepeat import build_nearrepeat_global, build_nearrepeat_by_category
from ..schemas import POI_COLS, INV_COLS


class FeatureBuilder:
    def __init__(self, e1_encoders: dict, e2_encoders: dict):
        self._e1_enc = e1_encoders
        self._e2_enc = e2_encoders

    def build_e1(
        self,
        h3_indexes: list[str],
        target_date: pd.Timestamp,
        history_monthly: pd.DataFrame,
    ) -> pd.DataFrame:
        """DataFrame con las 22 features de E1, indexado por h3_index."""
        target_period = pd.Period(target_date, freq="M")
        return build_e1_features(h3_indexes, target_period, history_monthly, self._e1_enc)

    def build_e2(
        self,
        h3_indexes: list[str],
        target_date: pd.Timestamp,
        lambda_monthly: np.ndarray,
        history_daily: pd.DataFrame,
        e1_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        DataFrame con las 17 features de E2, indexado por h3_index.
        Requiere e1_df para extraer zona_geografica, clave_mun y municipio.
        lambda_monthly: array con λ_mensual por celda (orden = h3_indexes).
        """
        cal = date_features(target_date)
        log_lam = np.log(np.maximum(lambda_monthly / target_date.days_in_month, 1e-6))

        temporal = build_e2_temporal(h3_indexes, target_date, history_daily, self._e2_enc)
        nr = build_nearrepeat_global(h3_indexes, target_date, history_daily)

        df = temporal.join(nr)
        df["log_lam_dia_base"] = log_lam
        df["dia_semana"]    = cal["dia_semana"]
        df["es_fin_semana"] = cal["es_fin_semana"]
        df["es_festivo"]    = cal["es_festivo"]
        df["mes"]           = cal["mes"]

        # categoricals codificados: provienen de E1 (mismo encoder)
        df["zona_geografica"] = e1_df["zona_geografica"].values
        df["clave_mun"]       = e1_df["clave_mun"].values

        # metadata para el output final
        df["municipio"]          = e1_df["municipio"].values
        df["zona_geografica_str"] = e1_df["zona_geografica_str"].values

        return df

    def build_e3(
        self,
        e2_df: pd.DataFrame,
        lambda_daily: np.ndarray,
        history_daily: pd.DataFrame,
        target_date: pd.Timestamp,
        denue: pd.DataFrame,
        inegi_inv: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        DataFrame con las 50 features de E3, indexado por h3_index.
        Extiende e2_df con lambda_diario, near-repeat por categoría, DENUE e INEGI.
        """
        h3_indexes = e2_df.index.tolist()
        df = e2_df.copy()
        df["lambda_diario"] = lambda_daily

        nr_cat = build_nearrepeat_by_category(h3_indexes, target_date, history_daily)
        df = df.join(nr_cat)

        poi = denue.set_index("h3_index")[POI_COLS].reindex(h3_indexes).fillna(0.0)
        inv = inegi_inv.set_index("h3_index")[INV_COLS].reindex(h3_indexes).fillna(0.0)
        df = df.join(poi).join(inv)

        return df

    def confidence_score(
        self,
        e2_df: pd.DataFrame,
        target_date: pd.Timestamp,
        data_end: pd.Timestamp,
    ) -> np.ndarray:
        days_from_end = max(0, (target_date - data_end).days)
        if days_from_end == 0:
            penalty = 0
        elif days_from_end <= 31:
            penalty = 25
        elif days_from_end <= 180:
            penalty = 35
        else:
            penalty = 45

        has_poi = (e2_df.get("poi_total", pd.Series(0, index=e2_df.index)) > 0).astype(int)
        has_inv = (e2_df.get("inv_n_segmentos", pd.Series(0, index=e2_df.index)) > 0).astype(int)
        active  = (e2_df["log_lam_dia_base"] > -5).astype(int)

        return (70 + has_poi * 10 + has_inv * 10 + active * 10 - penalty).clip(0, 100).values
