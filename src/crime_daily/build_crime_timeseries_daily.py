"""
Agrega microdatos IIEG a serie temporal por H3 res=9 × día.
Genera panel completo (ceros incluidos) para todos los H3 activos × fechas del rango.

Memoria estimada: ~1.5 GB durante la construcción, ~300 MB en disco.
"""
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path(__file__).parent.parent.parent / "data/processed/iieg_unified.parquet"
MONTHLY_TS = Path(__file__).parent.parent.parent / "data/processed/crime_timeseries.parquet"
OUTPUT = Path(__file__).parent.parent.parent / "data/processed/crime_timeseries_daily.parquet"


def main():
    print("Cargando iieg_unified.parquet...")
    df = pd.read_parquet(INPUT, columns=["h3_9", "fecha"])
    df = df[df["h3_9"].notna()].copy()
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
    df = df[df["fecha"].notna()]
    print(f"  Registros con H3 y fecha válida: {len(df):,}")

    counts = (
        df.groupby(["h3_9", "fecha"])
        .size()
        .reset_index(name="conteo")
    )

    all_h3 = np.sort(counts["h3_9"].unique())
    date_min = df["fecha"].min()
    date_max = df["fecha"].max()
    all_dates = pd.date_range(start=date_min, end=date_max, freq="D")
    n_h3 = len(all_h3)
    n_dates = len(all_dates)

    print(f"\n  H3 únicos: {n_h3:,}")
    print(f"  Rango: {date_min.date()} → {date_max.date()} ({n_dates:,} días)")
    print(f"  Panel total: {n_h3 * n_dates:,} filas")
    print("  Construyendo por año...")

    # Atributos estáticos desde la serie mensual
    monthly = pd.read_parquet(
        MONTHLY_TS,
        columns=["h3_9", "zona_geografica", "clave_mun", "region", "municipio"],
    )
    static = monthly.groupby("h3_9").first().reset_index()

    counts_idx = counts.set_index(["h3_9", "fecha"])["conteo"]

    chunks = []
    for year in range(date_min.year, date_max.year + 1):
        year_start = max(date_min, pd.Timestamp(year, 1, 1))
        year_end = min(date_max, pd.Timestamp(year, 12, 31))
        year_dates = pd.date_range(year_start, year_end, freq="D")
        nd = len(year_dates)

        h3_arr = np.repeat(all_h3, nd)
        date_arr = np.tile(year_dates, n_h3)

        chunk = pd.DataFrame({"h3_9": h3_arr, "fecha": date_arr})
        chunk = chunk.merge(
            counts.rename(columns={"conteo": "conteo"}),
            on=["h3_9", "fecha"],
            how="left",
        )
        chunk["conteo"] = chunk["conteo"].fillna(0).astype("int16")

        chunk["año"] = year
        chunk["mes"] = chunk["fecha"].dt.month.astype("int8")
        chunk["dia"] = chunk["fecha"].dt.day.astype("int8")
        chunk["dia_semana"] = chunk["fecha"].dt.dayofweek.astype("int8")
        chunk["es_fin_semana"] = (chunk["dia_semana"] >= 5).astype("int8")

        chunks.append(chunk)
        print(f"  {year}: {len(chunk):,} filas")

    print("Concatenando años...")
    panel = pd.concat(chunks, ignore_index=True)
    del chunks

    panel = panel.merge(static, on="h3_9", how="left")
    panel = panel.sort_values(["h3_9", "fecha"]).reset_index(drop=True)

    zeros_pct = (panel["conteo"] == 0).mean() * 100
    print(f"\nShape final: {panel.shape}")
    print(f"  Filas conteo=0: {zeros_pct:.1f}%")
    print(f"  Conteo máximo:  {panel['conteo'].max()}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT, index=False)
    print(f"\nGuardado en {OUTPUT}")


if __name__ == "__main__":
    main()
