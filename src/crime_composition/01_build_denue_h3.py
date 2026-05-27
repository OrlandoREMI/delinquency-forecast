"""
Proceso DENUE 2025 → conteos de POIs por celda H3 res=9.
Output: data/processed/denue_h3.parquet
"""
import zipfile
from pathlib import Path

import h3
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
DENUE_ZIP = ROOT / "data/inegi_denue2025/denue_14_csv.zip"
OUT = ROOT / "data/processed/denue_h3.parquet"

# Prefijos SCIAN 2018 (6 dígitos) por categoría de POI relevante para crimen
POI_PREFIXES = {
    "poi_bancos":       ("522", "523"),          # banca, crédito, casas de cambio
    "poi_bares":        ("7224", "7225"),         # bares, cantinas, centros nocturnos
    "poi_escuelas":     ("611",),                 # toda la educación formal
    "poi_salud":        ("621", "622"),           # consultorios, hospitales
    "poi_conveniencia": ("46111", "46212"),       # abarrotes, tiendas de autoservicio
    "poi_hoteles":      ("7211", "7212"),         # hoteles, moteles
    "poi_gasolineras":  ("46711",),              # gasolinerías
    "poi_mercados":     ("46211",),              # mercados y tianguis
    "poi_farmacias":    ("46412",),              # farmacias
}

POI_COLS = list(POI_PREFIXES.keys())


def classify(codigo: str) -> dict:
    return {cat: int(any(codigo.startswith(p) for p in prefixes))
            for cat, prefixes in POI_PREFIXES.items()}


def main() -> None:
    print("Cargando DENUE 2025 (Jalisco)...")
    with zipfile.ZipFile(DENUE_ZIP) as z:
        with z.open("conjunto_de_datos/denue_inegi_14_.csv") as f:
            df = pd.read_csv(
                f, encoding="latin-1",
                usecols=["codigo_act", "latitud", "longitud"],
                dtype={"codigo_act": str},
            )

    print(f"  Registros cargados: {len(df):,}")
    df = df.dropna(subset=["latitud", "longitud"])
    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")
    df = df.dropna(subset=["latitud", "longitud"])

    # Filtro bbox Jalisco
    df = df[
        (df["latitud"] >= 18.5) & (df["latitud"] <= 23.0) &
        (df["longitud"] >= -106.0) & (df["longitud"] <= -101.0)
    ]
    print(f"  Tras filtro bbox: {len(df):,}")

    # Mapeo a H3
    print("  Mapeando a H3 res=9...")
    df["h3_9"] = df.apply(
        lambda r: h3.latlng_to_cell(r["latitud"], r["longitud"], 9), axis=1
    )

    # Clasificar POIs
    clf = df["codigo_act"].apply(classify)
    df = pd.concat([df[["h3_9"]], pd.DataFrame(list(clf))], axis=1)

    # Agregar por H3
    agg = df.groupby("h3_9")[POI_COLS].sum().reset_index()
    agg["poi_total"] = agg[POI_COLS].sum(axis=1)
    for col in POI_COLS:
        agg[f"{col}_flag"] = (agg[col] > 0).astype("int8")

    print(f"  Celdas H3 con POIs: {len(agg):,}")
    agg.to_parquet(OUT, index=False)
    print(f"  Guardado en {OUT}")
    print(agg[POI_COLS + ["poi_total"]].describe().round(2))


if __name__ == "__main__":
    main()
