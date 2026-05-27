"""
Procesa INV 2020 Jalisco (frentes de manzana) → variables de infraestructura urbana por celda H3 res=9.
Output: data/processed/inegi_inv_h3.parquet
"""
import io
import os
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
INV_ZIP = ROOT / "data/inegi_inv2020/00_Frentes_INV2020_shp.zip"
OUT = ROOT / "data/processed/inegi_inv_h3.parquet"

# Columnas de infraestructura binaria (1 = Dispone, 3 = No dispone)
INF_COLS = [
    "BANQUETA",   # banqueta / acera
    "ALUMPUB",    # alumbrado público
    "DRENAJEP",   # drenaje pluvial
    "TRANSCOL",   # transporte colectivo
    "SEMAPEAT",   # semáforo peatonal
    "PARATRAN",   # parada de transporte
    "RAMPAS",     # rampas de accesibilidad
    "PASOPEAT",   # paso peatonal
    "CICLOVIA",   # ciclovía
    "PUESTOSF",   # puestos fijos (comercio informal)
    "PUESTOAM",   # puestos ambulantes
    "ARBOLES",    # arbolado
    "GUARNICI",   # guarnición
]


def load_jalisco_gdf() -> gpd.GeoDataFrame:
    with zipfile.ZipFile(INV_ZIP) as outer:
        with outer.open("00_Frentes_INV2020_shp/14_Frentes_INV2020_shp.zip") as ib:
            inner_data = io.BytesIO(ib.read())

    with zipfile.ZipFile(inner_data) as inner:
        with tempfile.TemporaryDirectory() as tmp:
            inner.extractall(tmp)
            subdir = os.path.join(tmp, "14_Frentes_INV2020_shp")
            shp = next(
                os.path.join(subdir, f)
                for f in os.listdir(subdir)
                if f.endswith(".shp")
            )
            return gpd.read_file(shp)


def main() -> None:
    print("Cargando INV 2020 Jalisco...")
    gdf = load_jalisco_gdf()
    print(f"  Segmentos cargados: {len(gdf):,} | CRS: {gdf.crs}")

    # Reproyectar a WGS84
    gdf = gdf.to_crs("EPSG:4326")

    # Centroide de cada segmento (LINESTRING → punto)
    centroids = gdf.geometry.centroid
    gdf = gdf.copy()
    gdf["lat"] = centroids.y
    gdf["lon"] = centroids.x

    # Filtrar coordenadas fuera de bbox Jalisco
    gdf = gdf[
        (gdf["lat"] >= 18.5) & (gdf["lat"] <= 23.0) &
        (gdf["lon"] >= -106.0) & (gdf["lon"] <= -101.0)
    ]
    print(f"  Tras filtro bbox: {len(gdf):,}")

    # Mapeo a H3 res=9
    print("  Mapeando centroides a H3 res=9...")
    gdf["h3_9"] = gdf.apply(
        lambda r: h3.latlng_to_cell(r["lat"], r["lon"], 9), axis=1
    )

    # Convertir columnas de infraestructura a binario (1 = tiene, 0 = no tiene)
    available = [c for c in INF_COLS if c in gdf.columns]
    df = gdf[["h3_9"] + available].copy()
    for col in available:
        df[col] = (df[col].astype("Int64", errors="ignore") == 1).astype("float32")

    # Agregar por H3: proporción de segmentos con cada característica
    pct = df.groupby("h3_9")[available].mean().reset_index()
    pct.columns = ["h3_9"] + [f"inv_{c.lower()}_pct" for c in available]

    cnt = df.groupby("h3_9").size().reset_index(name="inv_n_segmentos")
    result = pct.merge(cnt, on="h3_9")

    print(f"  Celdas H3 con datos INV: {len(result):,}")
    result.to_parquet(OUT, index=False)
    print(f"  Guardado en {OUT}")
    inv_cols = [c for c in result.columns if c.startswith("inv_")]
    print(result[inv_cols].describe().round(3))


if __name__ == "__main__":
    main()
