"""
ParquetLoader — misma interfaz que PostgresLoader pero lee desde archivos parquet locales.

Archivos esperados en data_dir:
    features_temporal.parquet        → crime_monthly
    crime_timeseries_daily.parquet   → base del daily (sin categoria)
    iieg_unified.parquet             → fuente de categoria por incidente
    denue_h3.parquet                 → denue
    inegi_inv_h3.parquet             → inegi_inv
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from ..schemas import DELITO_CATEGORIA, POI_COLS

_MONTHLY_LOOKBACK_MONTHS = 24


class ParquetLoader:
    def __init__(self, data_dir: str | Path):
        self._dir = Path(data_dir)

    def load(
        self,
        fecha: str,
        municipio: Optional[str] = None,
        h3_indexes: Optional[list[str]] = None,
    ) -> dict:
        if municipio is None and h3_indexes is None:
            raise ValueError("Se requiere municipio o h3_indexes.")

        ts    = pd.Timestamp(fecha)
        start = ts - pd.Timedelta(days=_MONTHLY_LOOKBACK_MONTHS * 31)

        crime_monthly = self._load_monthly(ts, start, municipio, h3_indexes)
        crime_daily   = self._load_daily(ts, municipio, h3_indexes)
        denue         = self._load_denue()
        inegi_inv     = self._load_inegi_inv()
        data_end      = self._infer_data_end()

        return dict(
            crime_history_monthly=crime_monthly,
            crime_history_daily=crime_daily,
            denue=denue,
            inegi_inv=inegi_inv,
            data_end=str(data_end.date()) if data_end else None,
        )

    # ------------------------------------------------------------------

    def _load_monthly(self, ts, start, municipio, h3_indexes):
        filters = [("año_mes", ">=", start)]
        if municipio:
            filters.append(("municipio", "=", municipio))

        df = (
            pq.read_table(
                self._dir / "features_temporal.parquet",
                filters=filters,
                columns=["h3_9", "año_mes", "conteo", "municipio",
                         "zona_geografica", "region", "clave_mun"],
            )
            .to_pandas()
            .rename(columns={"h3_9": "h3_index"})
        )

        if h3_indexes:
            df = df[df["h3_index"].isin(set(h3_indexes))]

        return df

    def _load_daily(self, ts, municipio, h3_indexes):
        nr_start = ts - pd.Timedelta(days=30)

        # Conteos diarios por celda
        filters = [("fecha", ">=", nr_start), ("fecha", "<", ts)]
        if municipio:
            filters.append(("municipio", "=", municipio))

        daily = (
            pq.read_table(
                self._dir / "crime_timeseries_daily.parquet",
                filters=filters,
                columns=["h3_9", "fecha", "conteo"],
            )
            .to_pandas()
            .rename(columns={"h3_9": "h3_index"})
        )

        if h3_indexes:
            daily = daily[daily["h3_index"].isin(set(h3_indexes))]

        # Categoria desde iieg_unified (incidentes individuales)
        iieg_path = self._dir / "iieg_unified.parquet"
        if iieg_path.exists():
            incidents = (
                pq.read_table(
                    iieg_path,
                    filters=[("fecha", ">=", nr_start), ("fecha", "<", ts)],
                    columns=["h3_9", "fecha", "delito"],
                )
                .to_pandas()
                .rename(columns={"h3_9": "h3_index"})
            )
            incidents["categoria"] = incidents["delito"].map(DELITO_CATEGORIA)
            incidents = incidents.dropna(subset=["categoria"])

            cat_counts = (
                incidents.groupby(["h3_index", "fecha", "categoria"])
                .size()
                .reset_index(name="conteo_cat")
            )
            daily = daily.merge(
                cat_counts[["h3_index", "fecha", "categoria"]].drop_duplicates(),
                on=["h3_index", "fecha"],
                how="left",
            )

        return daily

    def _load_denue(self):
        raw = pd.read_parquet(self._dir / "denue_h3.parquet")
        return (
            raw.rename(columns={"h3_9": "h3_index"})
            .drop(columns=[c for c in raw.columns if c.endswith("_flag")])
        )

    def _load_inegi_inv(self):
        return pd.read_parquet(self._dir / "inegi_inv_h3.parquet").rename(
            columns={"h3_9": "h3_index"}
        )

    def _infer_data_end(self) -> Optional[pd.Timestamp]:
        try:
            meta = pq.read_metadata(self._dir / "crime_timeseries_daily.parquet")
            # Lee solo la última fila de estadísticas — evita cargar el archivo completo
            df = pq.read_table(
                self._dir / "crime_timeseries_daily.parquet",
                columns=["fecha"],
            ).to_pandas()
            return pd.Timestamp(df["fecha"].max())
        except Exception:
            return None
