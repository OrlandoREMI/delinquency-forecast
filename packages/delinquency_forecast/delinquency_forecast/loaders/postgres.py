"""
PostgresLoader — carga los datos desde PostgreSQL (o cualquier DB compatible via SQLAlchemy).

Tablas esperadas en la base de datos:
    crime_monthly   — h3_index, año_mes, conteo, municipio, zona_geografica, region, clave_mun
    crime_daily     — h3_index, fecha, conteo, municipio, categoria (nullable)
    denue           — h3_index, poi_bancos, ..., poi_total
    inegi_inv       — h3_index, inv_banqueta_pct, ..., inv_n_segmentos
"""
from __future__ import annotations

from typing import Optional
import pandas as pd
from sqlalchemy import text


def _iso(ts) -> str:
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


_DEFAULT_TABLES = {
    "crime_monthly": "crime_monthly",
    "crime_daily":   "crime_daily",
    "denue":         "denue",
    "inegi_inv":     "inegi_inv",
}

_MONTHLY_LOOKBACK_MONTHS = 24


class PostgresLoader:
    def __init__(self, conn, tables: Optional[dict] = None):
        """
        conn:   SQLAlchemy Connection o Engine.
        tables: nombres de tablas alternativos, e.g. {"crime_monthly": "crimen_mensual"}.
        """
        self._conn   = conn
        self._tables = {**_DEFAULT_TABLES, **(tables or {})}

    def load(
        self,
        fecha: str,
        municipio: Optional[str] = None,
        h3_indexes: Optional[list[str]] = None,
    ) -> dict:
        """
        Retorna un dict listo para desempacar en pipeline.predict(**loader.load(...)).

        Se debe pasar municipio o h3_indexes (no ambos a la vez).
        data_end se infiere del máximo fecha en crime_daily para el filtro aplicado.
        """
        if municipio is None and h3_indexes is None:
            raise ValueError("Se requiere municipio o h3_indexes.")

        ts    = pd.Timestamp(fecha)
        start = ts - pd.Timedelta(days=_MONTHLY_LOOKBACK_MONTHS * 31)

        t = self._tables
        filter_col  = "municipio" if municipio else "h3_index"
        filter_val  = municipio   if municipio else None
        filter_list = h3_indexes  if h3_indexes else None

        crime_monthly = self._query_monthly(t["crime_monthly"], filter_col, filter_val, filter_list, start)
        crime_daily   = self._query_daily(t["crime_daily"],     filter_col, filter_val, filter_list, ts)
        denue         = self._query_static(t["denue"])
        inegi_inv     = self._query_static(t["inegi_inv"])
        data_end      = self._infer_data_end(t["crime_daily"])

        return dict(
            crime_history_monthly=crime_monthly,
            crime_history_daily=crime_daily,
            denue=denue,
            inegi_inv=inegi_inv,
            data_end=str(data_end.date()) if data_end else None,
        )

    # ------------------------------------------------------------------

    def _query_monthly(self, table, filter_col, filter_val, filter_list, start):
        start_s = _iso(start)
        if filter_val is not None:
            sql = text(f"""
                SELECT h3_index, año_mes, conteo, municipio, zona_geografica, region, clave_mun
                FROM {table}
                WHERE {filter_col} = :val AND año_mes >= :start
            """)
            return pd.read_sql(sql, self._conn, params={"val": filter_val, "start": start_s})
        else:
            sql = text(f"""
                SELECT h3_index, año_mes, conteo, municipio, zona_geografica, region, clave_mun
                FROM {table}
                WHERE h3_index IN :cells AND año_mes >= :start
            """)
            return pd.read_sql(sql, self._conn, params={"cells": tuple(filter_list), "start": start_s})

    def _query_daily(self, table, filter_col, filter_val, filter_list, ts):
        nr_start = ts - pd.Timedelta(days=30)
        if filter_val is not None:
            sql = text(f"""
                SELECT h3_index, fecha, conteo, categoria
                FROM {table}
                WHERE {filter_col} = :val AND fecha >= :start AND fecha < :end
            """)
            return pd.read_sql(sql, self._conn, params={"val": filter_val, "start": _iso(nr_start), "end": _iso(ts)})
        else:
            sql = text(f"""
                SELECT h3_index, fecha, conteo, categoria
                FROM {table}
                WHERE h3_index IN :cells AND fecha >= :start AND fecha < :end
            """)
            return pd.read_sql(sql, self._conn, params={"cells": tuple(filter_list), "start": _iso(nr_start), "end": _iso(ts)})

    def _query_static(self, table):
        return pd.read_sql(text(f"SELECT * FROM {table}"), self._conn)

    def _infer_data_end(self, table) -> Optional[pd.Timestamp]:
        row = pd.read_sql(text(f"SELECT MAX(fecha) AS max_fecha FROM {table}"), self._conn)
        val = row["max_fecha"].iloc[0]
        return pd.Timestamp(val) if val is not None else None
