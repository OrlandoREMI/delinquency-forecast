"""
Fixtures compartidas. Los datos son completamente sintéticos —
no dependen de archivos externos del proyecto.
"""
import h3
import numpy as np
import pandas as pd
import pytest

from delinquency_forecast import DelinquencyPipeline
from delinquency_forecast.schemas import CLASES, INV_COLS, NR_CAT_COLS, POI_COLS

# GDL centro — 19 celdas H3 res=9
_CENTER = h3.latlng_to_cell(20.6597, -103.3496, 9)
_CELLS = list(h3.grid_disk(_CENTER, 2))

# Fechas de referencia para los tests
HIST_DATE = "2023-11-28"      # dentro del rango del historial sintético
FUTURE_DATE = "2025-08-15"    # fuera del historial — debe activar fallback futuro


@pytest.fixture(scope="session")
def pipeline():
    return DelinquencyPipeline.load()


@pytest.fixture(scope="session")
def h3_cells():
    return _CELLS


@pytest.fixture(scope="session")
def crime_history_monthly():
    """24 meses de conteos mensuales sintéticos para las 19 celdas."""
    months = pd.date_range("2022-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(42)
    rows = [
        {
            "h3_index": cell,
            "año_mes": m,
            "conteo": int(rng.poisson(3)),
            "zona_geografica": "AMG",
            "region": "Centro",
            "clave_mun": 39,
            "municipio": "Guadalajara",
        }
        for cell in _CELLS
        for m in months
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def crime_history_daily():
    """30 días de conteos diarios con categoría de delito para near-repeat."""
    dates = pd.date_range("2023-11-01", periods=30, freq="D")
    rng = np.random.default_rng(42)
    rows = []
    for cell in _CELLS:
        for d in dates:
            if rng.random() > 0.6:
                rows.append({
                    "h3_index": cell,
                    "fecha": d,
                    "conteo": int(rng.poisson(1) + 1),
                    "categoria": rng.choice(CLASES),
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["h3_index", "fecha", "conteo", "categoria"]
    )


@pytest.fixture(scope="session")
def denue():
    rng = np.random.default_rng(42)
    n = len(_CELLS)
    df = pd.DataFrame({"h3_index": _CELLS})
    poi_base = [c for c in POI_COLS if c != "poi_total"]
    for col in poi_base:
        df[col] = rng.integers(0, 8, n).astype(float)
    df["poi_total"] = df[poi_base].sum(axis=1)
    return df


@pytest.fixture(scope="session")
def inegi_inv():
    rng = np.random.default_rng(42)
    n = len(_CELLS)
    df = pd.DataFrame({"h3_index": _CELLS})
    for col in INV_COLS:
        df[col] = (
            rng.integers(1, 20, n).astype(float)
            if col == "inv_n_segmentos"
            else rng.uniform(0, 1, n)
        )
    return df


@pytest.fixture(scope="session")
def precomputed_features(denue, inegi_inv):
    """
    Features_df para predict_from_features: contiene todas las columnas
    que el backend habría precomputado (E2 + POI + INV + NR_CAT + metadata).
    """
    rng = np.random.default_rng(42)
    n = len(_CELLS)

    df = pd.DataFrame({"h3_index": _CELLS})

    # E2 features
    df["log_lam_dia_base"] = rng.uniform(-3.0, -0.5, n)
    df["dia_semana"]       = 2
    df["es_fin_semana"]    = 0
    df["es_festivo"]       = 0
    df["mes"]              = 11
    df["lag_1"]            = rng.integers(0, 4, n).astype(float)
    df["lag_7"]            = rng.integers(0, 4, n).astype(float)
    df["lag_14"]           = rng.integers(0, 4, n).astype(float)
    df["rolling_mean_7"]   = rng.uniform(0, 3, n)
    df["rolling_mean_14"]  = rng.uniform(0, 3, n)
    df["rolling_std_7"]    = rng.uniform(0, 1, n)
    df["zona_geografica"]  = 0   # encoded integer
    df["clave_mun"]        = 0
    df["nr_ring1_1d"]      = rng.integers(0, 3, n).astype(float)
    df["nr_ring1_7d"]      = rng.integers(0, 8, n).astype(float)
    df["nr_ring1_14d"]     = rng.integers(0, 15, n).astype(float)
    df["nr_ring2_7d"]      = rng.integers(0, 8, n).astype(float)

    # Near-repeat por categoría
    for col in NR_CAT_COLS:
        df[col] = rng.integers(0, 5, n).astype(float)

    # POI + INV
    poi = denue.set_index("h3_index").reindex(_CELLS).fillna(0).reset_index()
    inv = inegi_inv.set_index("h3_index").reindex(_CELLS).fillna(0).reset_index()
    for col in POI_COLS:
        df[col] = poi[col].values
    for col in INV_COLS:
        df[col] = inv[col].values

    # Metadata (opcional, mejora el output)
    df["municipio"]          = "Guadalajara"
    df["zona_geografica_str"] = "AMG"

    return df.set_index("h3_index")
