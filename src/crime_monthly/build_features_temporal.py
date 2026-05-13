"""
Genera features temporales para el modelo LightGBM sobre la serie temporal H3 × mes.
Incluye: lags, rolling stats, encoding cíclico, tendencia lineal y features de calendario.
"""
import numpy as np
import pandas as pd
from pathlib import Path

INPUT = Path(__file__).parent.parent.parent / "data/processed/crime_timeseries.parquet"
OUTPUT = Path(__file__).parent.parent.parent / "data/processed/features_temporal.parquet"

LAG_MONTHS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]


def cyclic_encode(series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    sin = np.sin(2 * np.pi * series / period)
    cos = np.cos(2 * np.pi * series / period)
    return sin, cos


def main():
    print("Cargando crime_timeseries.parquet...")
    df = pd.read_parquet(INPUT)
    df = df.sort_values(["h3_9", "año_mes"]).reset_index(drop=True)
    print(f"  Shape: {df.shape}")

    print("Generando features temporales...")

    df["mes_sin"], df["mes_cos"] = cyclic_encode(df["mes"], 12)

    min_date = df["año_mes"].min()
    df["trend"] = ((df["año_mes"].dt.year - min_date.year) * 12
                   + (df["año_mes"].dt.month - min_date.month))

    df["trimestre"] = df["año_mes"].dt.quarter

    df["es_fin_año"] = df["mes"].isin([11, 12]).astype(int)
    df["es_verano"] = df["mes"].isin([7, 8]).astype(int)

    print(f"  Calculando lags: {LAG_MONTHS}")
    grouped = df.groupby("h3_9")["conteo"]

    for lag in LAG_MONTHS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    print(f"  Calculando rolling: ventanas {ROLLING_WINDOWS} meses")
    for w in ROLLING_WINDOWS:
        rolled = grouped.shift(1).groupby(df["h3_9"]).transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        df[f"rolling_mean_{w}"] = rolled

    df["rolling_std_3"] = grouped.shift(1).groupby(df["h3_9"]).transform(
        lambda s: s.rolling(3, min_periods=1).std().fillna(0)
    )

    month_avg = (
        df.groupby(["h3_9", "mes"])["conteo"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    df["hist_mean_mes"] = month_avg

    df["hist_max"] = grouped.shift(1).groupby(df["h3_9"]).transform(
        lambda s: s.expanding().max().fillna(0)
    )

    feature_cols = [c for c in df.columns if c not in
                    ["h3_9", "año_mes", "conteo", "municipio", "clave_mun",
                     "zona_geografica", "region"]]

    print("\nNaNs por feature (esperados en períodos iniciales):")
    for col in feature_cols:
        n = df[col].isna().sum()
        if n > 0:
            print(f"  {col:<25} {n:>10,}  ({n/len(df)*100:.1f}%)")

    lag_roll_cols = [c for c in df.columns if c.startswith(("lag_", "rolling_", "hist_"))]
    df[lag_roll_cols] = df[lag_roll_cols].fillna(0)

    print(f"\nShape final: {df.shape}")
    print(f"Columnas: {list(df.columns)}")

    train = df[df["año_mes"].dt.year <= 2023]
    val = df[df["año_mes"].dt.year == 2024]
    test = df[df["año_mes"].dt.year == 2025]
    print(f"\nSplit temporal:")
    print(f"  Train (≤2023): {len(train):>10,} filas")
    print(f"  Val   (2024):  {len(val):>10,} filas")
    print(f"  Test  (2025):  {len(test):>10,} filas")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    print(f"\nGuardado en {OUTPUT}")


if __name__ == "__main__":
    main()
