"""
API pública del pipeline de predicción.

Dos paths:
  predict()               — on-the-fly: recibe datos crudos del backend, computa features E1→E2→E3
  predict_from_features() — rápido: recibe features precomputadas, solo corre E2→E3
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .stages.e1_monthly import E1Stage
from .stages.e2_daily import E2Stage
from .stages.e3_composition import E3Stage
from .features.builders import FeatureBuilder
from .schemas import CLASES, OUTPUT_COLS

_ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


class DelinquencyPipeline:
    def __init__(self, artifacts_dir: Optional[Path] = None):
        d = Path(artifacts_dir) if artifacts_dir else _ARTIFACTS_DIR
        self._e1 = E1Stage.load(d / "lgbm_poisson_v1.pkl")
        self._e2 = E2Stage.load(d / "lgbm_daily_v1.pkl")
        self._e3 = E3Stage.load(d / "lgbm_e3_v1.pkl")

    @classmethod
    def load(cls, artifacts_dir: Optional[Path] = None) -> "DelinquencyPipeline":
        return cls(artifacts_dir)

    # ------------------------------------------------------------------
    # Path 1: on-the-fly  (E1 → features → E2 → E3)
    # ------------------------------------------------------------------

    def predict(
        self,
        fecha: str,
        crime_history_monthly: pd.DataFrame,
        crime_history_daily: pd.DataFrame,
        denue: pd.DataFrame,
        inegi_inv: pd.DataFrame,
        h3_indexes: Optional[list[str]] = None,
        data_end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Predicción completa para una fecha dada.

        Args:
            fecha:                  'YYYY-MM-DD'
            crime_history_monthly:  historial mensual (ver schemas.py)
            crime_history_daily:    historial diario con >= 30 días previos a fecha
                                    columnas: [h3_index, fecha, conteo, categoria?]
            denue:                  POIs por H3 (tabla estática)
            inegi_inv:              infraestructura por H3 (tabla estática)
            h3_indexes:             celdas a predecir; None = todas en crime_history_monthly
            data_end:               último día con datos reales ('YYYY-MM-DD').
                                    Si None, se infiere de crime_history_monthly.
        Returns:
            DataFrame con columnas definidas en schemas.OUTPUT_COLS
        """
        ts = pd.Timestamp(fecha)

        if h3_indexes is None:
            h3_indexes = crime_history_monthly["h3_index"].unique().tolist()

        _data_end = (
            pd.Timestamp(data_end)
            if data_end
            else pd.Timestamp(pd.to_datetime(crime_history_monthly["año_mes"]).max())
        )

        builder = FeatureBuilder(self._e1.encoders, self._e2.encoders)

        e1_df = builder.build_e1(h3_indexes, ts, crime_history_monthly)
        lambda_monthly = self._e1.predict(e1_df)

        e2_df = builder.build_e2(h3_indexes, ts, lambda_monthly, crime_history_daily, e1_df)
        lambda_daily = self._e2.predict(e2_df)

        e3_df = builder.build_e3(e2_df, lambda_daily, crime_history_daily, ts, denue, inegi_inv)
        probs = self._e3.predict_proba(e3_df)

        return self._format_output(h3_indexes, e2_df, lambda_daily, probs, ts, _data_end, builder, e3_df)

    # ------------------------------------------------------------------
    # Path 2: desde features precomputadas  (E2 → E3)
    # ------------------------------------------------------------------

    def predict_from_features(
        self,
        fecha: str,
        features_df: pd.DataFrame,
        data_end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Inferencia rápida sobre features ya precomputadas por el backend.

        features_df debe contener todas las columnas de E2 más
        poi_*, inv_*, nr_cat_*_ring1_*d y metadata (h3_index, municipio,
        zona_geografica como string en 'zona_geografica_str').

        La columna 'lambda_diario' se computa aquí desde E2; no debe incluirse en features_df.
        """
        ts = pd.Timestamp(fecha)

        df = features_df.copy()
        if df.index.name != "h3_index":
            df = df.set_index("h3_index")

        h3_indexes = df.index.tolist()

        lambda_daily = self._e2.predict(df)
        df["lambda_diario"] = lambda_daily

        probs = self._e3.predict_proba(df)

        _data_end = pd.Timestamp(data_end) if data_end else ts
        builder = FeatureBuilder(self._e1.encoders, self._e2.encoders)

        return self._format_output(h3_indexes, df, lambda_daily, probs, ts, _data_end, builder, df)

    # ------------------------------------------------------------------
    # Formateo de salida
    # ------------------------------------------------------------------

    def _format_output(
        self,
        h3_indexes: list[str],
        e2_df: pd.DataFrame,
        lambda_daily: np.ndarray,
        probs: np.ndarray,
        ts: pd.Timestamp,
        data_end: pd.Timestamp,
        builder: FeatureBuilder,
        e3_df: pd.DataFrame,
    ) -> pd.DataFrame:
        out = pd.DataFrame({"h3_index": h3_indexes})
        out["municipio"]       = e2_df["municipio"].values if "municipio" in e2_df.columns else ""
        out["zona_geografica"] = e2_df["zona_geografica_str"].values if "zona_geografica_str" in e2_df.columns else ""
        out["lambda_diario"]   = lambda_daily
        out["prob_crimen"]     = (1 - np.exp(-np.clip(lambda_daily, 0, None))) * 100

        for i, cls in enumerate(CLASES):
            out[f"p_{cls}"] = probs[:, i]

        out["categoria_pred"] = self._e3.predict_classes(probs)
        out["nivel_riesgo"]   = _nivel_riesgo(lambda_daily)
        out["confiabilidad_score"] = builder.confidence_score(e3_df, ts, data_end)

        return out[OUTPUT_COLS]


def _nivel_riesgo(lam: np.ndarray) -> list[str]:
    pos = lam[lam > 0]
    if len(pos) == 0:
        p33, p66 = 1.0, 3.0
    else:
        p33, p66 = np.percentile(pos, [33, 66])
    return [
        "bajo" if v <= p33 else "medio" if v <= p66 else "alto"
        for v in lam
    ]
