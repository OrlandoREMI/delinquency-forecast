"""
Agrega los microdatos unificados a serie temporal por H3 res=9 × mes.
Solo usa registros con h3_9 válido.
Genera panel completo (ceros incluidos) para todos los H3 × meses del rango.
"""
from pathlib import Path

import pandas as pd

INPUT = Path(__file__).parent.parent.parent / "data/processed/iieg_unified.parquet"
OUTPUT = Path(__file__).parent.parent.parent / "data/processed/crime_timeseries.parquet"


def main():
    print("Cargando iieg_unified.parquet...")
    df = pd.read_parquet(INPUT)
    print(f"  Total registros: {len(df):,}")

    df = df[df["h3_9"].notna()].copy()
    print(f"  Registros con H3: {len(df):,}")

    df["año_mes"] = df["fecha"].dt.to_period("M")

    static = (
        df.groupby("h3_9")
        .agg(
            municipio=("municipio", lambda s: s.mode().iloc[0]),
            clave_mun=("clave_mun", lambda s: s.mode().iloc[0]),
            zona_geografica=("zona_geografica", lambda s: s.mode().iloc[0]),
            region=("region", lambda s: s.mode().iloc[0]),
        )
        .reset_index()
    )

    counts = (
        df.groupby(["h3_9", "año_mes"])
        .size()
        .reset_index(name="conteo")
    )

    all_h3 = counts["h3_9"].unique()
    all_months = pd.period_range(
        start=df["año_mes"].min(),
        end=df["año_mes"].max(),
        freq="M",
    )
    print(f"\n  H3 únicos activos: {len(all_h3):,}")
    print(f"  Rango temporal: {all_months[0]} → {all_months[-1]} ({len(all_months)} meses)")
    print(f"  Tamaño del panel completo: {len(all_h3) * len(all_months):,} filas")

    idx = pd.MultiIndex.from_product([all_h3, all_months], names=["h3_9", "año_mes"])
    panel = pd.DataFrame(index=idx).reset_index()
    panel = panel.merge(counts, on=["h3_9", "año_mes"], how="left")
    panel["conteo"] = panel["conteo"].fillna(0).astype(int)

    panel = panel.merge(static, on="h3_9", how="left")

    panel["año"] = panel["año_mes"].dt.year
    panel["mes"] = panel["año_mes"].dt.month

    panel["año_mes"] = panel["año_mes"].dt.to_timestamp()

    panel = panel.sort_values(["h3_9", "año_mes"]).reset_index(drop=True)

    print(f"\nShape final: {panel.shape}")
    print(f"Columnas: {list(panel.columns)}")
    print(f"Distribución de conteo:")
    print(f"  Filas con conteo = 0: {(panel['conteo'] == 0).sum():,} ({(panel['conteo'] == 0).mean()*100:.1f}%)")
    print(f"  Filas con conteo > 0: {(panel['conteo'] > 0).sum():,}")
    print(f"  Conteo máximo:        {panel['conteo'].max()}")
    print(f"  Conteo promedio (>0): {panel.loc[panel['conteo']>0,'conteo'].mean():.2f}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT, index=False)
    print(f"\nGuardado en {OUTPUT}")


if __name__ == "__main__":
    main()
