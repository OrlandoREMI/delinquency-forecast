"""
Ejemplo de uso del pipeline con PostgresLoader.

Simula las tablas de PostgreSQL cargando los parquets en DuckDB en memoria.
"""
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from sqlalchemy import create_engine

from delinquency_forecast import DelinquencyPipeline
from delinquency_forecast.loaders import PostgresLoader
from delinquency_forecast.schemas import DELITO_CATEGORIA, POI_COLS, INV_COLS

ROOT  = Path(__file__).resolve().parents[2]
DATA  = ROOT / "data/processed"
FECHA = "2024-06-15"
MUN   = "Guadalajara"

ts          = pd.Timestamp(FECHA)
start_month = ts - pd.Timedelta(days=24 * 31)
start_day   = ts - pd.Timedelta(days=30)

# ---------------------------------------------------------------------------
# Los siguientes bloques de código cargan los parquet files
# ---------------------------------------------------------------------------

crime_monthly = (
    pq.read_table(DATA / "features_temporal.parquet",
                  filters=[("municipio", "=", MUN), ("año_mes", ">=", start_month)],
                  columns=["h3_9", "año_mes", "conteo", "municipio",
                           "zona_geografica", "region", "clave_mun"])
    .to_pandas()
    .rename(columns={"h3_9": "h3_index"})
)

crime_daily_base = (
    pq.read_table(DATA / "crime_timeseries_daily.parquet",
                  filters=[("municipio", "=", MUN),
                           ("fecha", ">=", start_day), ("fecha", "<", ts)],
                  columns=["h3_9", "fecha", "conteo", "municipio"])
    .to_pandas()
    .rename(columns={"h3_9": "h3_index"})
)

incidents = (
    pq.read_table(DATA / "iieg_unified.parquet",
                  filters=[("fecha", ">=", start_day), ("fecha", "<", ts)],
                  columns=["h3_9", "fecha", "delito"])
    .to_pandas()
    .rename(columns={"h3_9": "h3_index"})
)
incidents["categoria"] = incidents["delito"].map(DELITO_CATEGORIA)
incidents = incidents.dropna(subset=["categoria"])

crime_daily = crime_daily_base.merge(
    incidents[["h3_index", "fecha", "categoria"]].drop_duplicates(["h3_index", "fecha"]),
    on=["h3_index", "fecha"],
    how="left",
)

denue = (
    pd.read_parquet(DATA / "denue_h3.parquet")
    .rename(columns={"h3_9": "h3_index"})
    [["h3_index"] + POI_COLS]
)

inegi_inv = (
    pd.read_parquet(DATA / "inegi_inv_h3.parquet")
    .rename(columns={"h3_9": "h3_index"})
    [["h3_index"] + INV_COLS]
)

# ---------------------------------------------------------------------------
# Registrar tablas en SQLite en memoria para simular la DB
# ---------------------------------------------------------------------------
engine = create_engine("sqlite:///:memory:")
crime_monthly.to_sql("crime_monthly", engine, index=False, if_exists="replace")
crime_daily.to_sql("crime_daily",     engine, index=False, if_exists="replace")
denue.to_sql("denue",                 engine, index=False, if_exists="replace")
inegi_inv.to_sql("inegi_inv",         engine, index=False, if_exists="replace")

# ---------------------------------------------------------------------------
# Iniciar el pipeline y loader
# ---------------------------------------------------------------------------

pipeline = DelinquencyPipeline.load()
loader   = PostgresLoader(engine.connect())

inputs = loader.load(fecha=FECHA, municipio=MUN)
result = pipeline.predict(fecha=FECHA, **inputs)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
top10 = result.sort_values("lambda_diario", ascending=False).head(10)

print(f"Predicción — {MUN} · {FECHA}  (data_end: {inputs['data_end']})")
print()
print(top10[["h3_index", "lambda_diario", "categoria_pred",
             "nivel_riesgo", "confiabilidad_score"]].to_string(index=False))
print()
print(top10[["h3_index", "p_alto_impacto", "p_violencia_personal",
             "p_robo_confrontacion", "p_robo_patrimonial"]].to_string(index=False))
