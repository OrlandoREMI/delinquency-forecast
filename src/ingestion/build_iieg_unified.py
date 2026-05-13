"""
Unifica los 12 archivos de microdatos IIEG de incidencia delictiva en Jalisco.
- Limpia headers (BOM, comillas)
- Estandariza fechas
- Reporta cobertura de coordenadas (x, y) por región y zona geográfica
- Codifica coordenadas válidas a H3 resolución 9
"""
import io
import re
import zipfile
from pathlib import Path

import h3
import pandas as pd

INPUT_DIR = Path(__file__).parent.parent.parent / "data/iieg"
OUTPUT = Path(__file__).parent.parent.parent / "data/processed/iieg_unified.parquet"

# Jalisco bounding box holgado (lon, lat)
LON_MIN, LON_MAX = -106.0, -101.0
LAT_MIN, LAT_MAX = 18.5, 23.0

EXPECTED_COLS = [
    "fecha", "delito", "x", "y", "colonia",
    "municipio", "clave_mun", "hora", "bien_afectado", "zona_geografica",
]


def region_from_path(path: Path) -> str:
    name = path.stem.replace("incidencia_", "").replace("incidencia", "")
    return name.strip("_")


def clean_headers(headers: list[str]) -> list[str]:
    cleaned = []
    for h in headers:
        h = h.strip().lstrip("﻿").strip('"').strip()
        cleaned.append(h.lower())
    return cleaned


def read_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise ValueError(f"No CSV encontrado en {zip_path}")
        with z.open(csvs[0]) as f:
            raw = f.read()
    return parse_csv_bytes(raw)


def parse_csv_bytes(raw: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(text), dtype=str, low_memory=False)
    df.columns = clean_headers(list(df.columns))
    df = df[[c for c in df.columns if c in EXPECTED_COLS]]
    return df


def standardize_date(series: pd.Series) -> pd.Series:
    """Parsea fechas tolerando YYYY-MM-DD y DD/MM/YYYY en el mismo DataFrame."""
    result = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    mask = result.isna() & series.notna()
    if mask.any():
        result[mask] = pd.to_datetime(series[mask], format="%d/%m/%Y", errors="coerce")
    return result


def is_valid_coord(x_series: pd.Series, y_series: pd.Series) -> pd.Series:
    x = pd.to_numeric(x_series, errors="coerce")
    y = pd.to_numeric(y_series, errors="coerce")
    return (
        x.notna() & y.notna()
        & (x != 0) & (y != 0)
        & x.between(LON_MIN, LON_MAX)
        & y.between(LAT_MIN, LAT_MAX)
    )


def encode_h3(row) -> str | None:
    try:
        return h3.latlng_to_cell(float(row["y"]), float(row["x"]), 9)
    except Exception:
        return None


def print_coverage_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("COBERTURA DE COORDENADAS (x, y)")
    print("=" * 60)

    print("\nPor región:")
    cov = (
        df.groupby("region")["coord_valida"]
        .agg(total="count", con_coords="sum")
        .assign(pct=lambda d: d["con_coords"] / d["total"] * 100)
        .sort_values("pct", ascending=False)
    )
    print(cov.to_string(float_format="%.1f"))

    print("\nPor zona geográfica:")
    cov_zona = (
        df.groupby("zona_geografica")["coord_valida"]
        .agg(total="count", con_coords="sum")
        .assign(pct=lambda d: d["con_coords"] / d["total"] * 100)
        .sort_values("pct", ascending=False)
    )
    print(cov_zona.to_string(float_format="%.1f"))

    amg_pct = cov_zona.loc["AMG", "pct"] if "AMG" in cov_zona.index else 0.0
    print("\n" + "=" * 60)
    if amg_pct >= 50:
        print(f"DECISION: AMG con {amg_pct:.1f}% de cobertura → ENFOQUE H3 res=9 (AMG)")
    else:
        print(f"DECISION: AMG con {amg_pct:.1f}% de cobertura → ENFOQUE DUAL")
        print("  (a) municipio con todos los registros")
        print("  (b) H3 res=9 solo con registros que tienen coordenadas")
    print("=" * 60 + "\n")


def main():
    files = sorted(INPUT_DIR.glob("incidencia_*"))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos en {INPUT_DIR}")

    frames = []
    for path in files:
        region = region_from_path(path)
        print(f"Leyendo {path.name} ...")
        if path.suffix == ".zip":
            df = read_csv_from_zip(path)
        else:
            df = parse_csv_bytes(path.read_bytes())
        df["region"] = region
        frames.append(df)
        print(f"  {len(df):,} filas, {list(df.columns)}")

    print("\nUnificando...")
    unified = pd.concat(frames, ignore_index=True)
    print(f"Total filas: {len(unified):,}")

    unified["fecha"] = standardize_date(unified["fecha"])
    unified["clave_mun"] = pd.to_numeric(unified["clave_mun"], errors="coerce").astype("Int64")
    unified["id_mun"] = unified["clave_mun"].apply(
        lambda v: f"ENT14MUN{int(v):03d}" if pd.notna(v) else None
    )

    unified["zona_geografica"] = unified["zona_geografica"].str.strip().str.upper()
    unified["zona_geografica"] = unified["zona_geografica"].replace(
        {"AMG": "AMG", "INTERIOR": "Interior"}
    ).fillna("Interior")

    unified["coord_valida"] = is_valid_coord(unified["x"], unified["y"])
    unified["x"] = pd.to_numeric(unified["x"], errors="coerce")
    unified["y"] = pd.to_numeric(unified["y"], errors="coerce")

    print_coverage_report(unified)

    print("Codificando H3 res=9...")
    mask = unified["coord_valida"]
    unified["h3_9"] = None
    unified.loc[mask, "h3_9"] = unified.loc[mask].apply(encode_h3, axis=1)
    print(f"  Registros con H3: {unified['h3_9'].notna().sum():,}")

    unified = unified.drop(columns=["coord_valida"])

    print("\nVerificaciones:")
    print(f"  Total filas:           {len(unified):,}")
    print(f"  NaN en fecha:          {unified['fecha'].isna().sum():,}")
    print(f"  NaN en clave_mun:      {unified['clave_mun'].isna().sum():,}")
    print(f"  NaN en delito:         {unified['delito'].isna().sum():,}")
    print(f"  NaN en region:         {unified['region'].isna().sum():,}")
    print(f"  Registros sin H3:      {unified['h3_9'].isna().sum():,}")
    print(f"  Columnas: {list(unified.columns)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    unified.to_parquet(OUTPUT, index=False)
    print(f"\nGuardado en {OUTPUT}")


if __name__ == "__main__":
    main()
