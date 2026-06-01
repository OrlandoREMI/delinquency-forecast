"""
Near-repeat features vectorizados mediante matrices sparse CSR.
Reemplaza el bucle por celda de batch_predict.py original.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h3

from ..schemas import CLASES, CAT_SHORTS, DELITO_CATEGORIA


def _build_adjacency(target_cells: list[str], ring_k: int) -> tuple[sp.csr_matrix, list[str]]:
    """Matriz adjacencia: filas=target_cells, columnas=target_cells ∪ vecinos ring-k."""
    nb_sets = {cell: set(h3.grid_ring(cell, ring_k)) for cell in target_cells}
    all_cells = sorted(set(target_cells) | {nb for nbs in nb_sets.values() for nb in nbs})
    cell_idx = {c: i for i, c in enumerate(all_cells)}

    rows, cols = [], []
    for i, cell in enumerate(target_cells):
        for nb in nb_sets[cell]:
            j = cell_idx.get(nb)
            if j is not None:
                rows.append(i)
                cols.append(j)

    A = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(target_cells), len(all_cells)),
    )
    return A, all_cells


def _crime_vector(
    all_cells: list[str],
    pivot: pd.DataFrame,
    target_date: pd.Timestamp,
    window_days: int,
) -> np.ndarray:
    cutoff = target_date - pd.Timedelta(days=window_days)
    window_cols = [c for c in pivot.columns if cutoff <= c < target_date]
    if not window_cols:
        return np.zeros(len(all_cells), dtype=np.float32)
    return (
        pivot.reindex(all_cells).fillna(0.0)[window_cols]
        .sum(axis=1)
        .values.astype(np.float32)
    )


def build_nearrepeat_global(
    h3_indexes: list[str],
    target_date: pd.Timestamp,
    history_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    nr_ring1_1d, nr_ring1_7d, nr_ring1_14d, nr_ring2_7d para cada celda.
    history_daily: columnas [h3_index, fecha, conteo]
    """
    hist = history_daily.copy()
    hist["fecha"] = pd.to_datetime(hist["fecha"])
    hist = hist[hist["fecha"] < target_date]

    pivot = (
        hist.pivot_table(index="h3_index", columns="fecha", values="conteo", aggfunc="sum")
        .fillna(0.0)
    )

    A1, all1 = _build_adjacency(h3_indexes, ring_k=1)
    A2, all2 = _build_adjacency(h3_indexes, ring_k=2)

    result = pd.DataFrame({"h3_index": h3_indexes})
    for window, adj, all_cells, col in [
        (1,  A1, all1, "nr_ring1_1d"),
        (7,  A1, all1, "nr_ring1_7d"),
        (14, A1, all1, "nr_ring1_14d"),
        (7,  A2, all2, "nr_ring2_7d"),
    ]:
        v = _crime_vector(all_cells, pivot, target_date, window)
        result[col] = (adj @ v).astype(float)

    return result.set_index("h3_index")


def build_nearrepeat_by_category(
    h3_indexes: list[str],
    target_date: pd.Timestamp,
    history_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    nr_cat_{alto,viol,conf,patr}_ring1_{7,14}d para cada celda.
    history_daily: columnas [h3_index, fecha, conteo, categoria]
    Si 'categoria' no está presente, todas las features se rellenan con 0.
    """
    zero_cols = {f"nr_cat_{s}_ring1_{w}d": 0.0 for s in CAT_SHORTS for w in [7, 14]}
    base = pd.DataFrame({"h3_index": h3_indexes}).set_index("h3_index")

    if "categoria" not in history_daily.columns:
        return base.assign(**zero_cols)

    hist = history_daily.copy()
    hist["fecha"] = pd.to_datetime(hist["fecha"])
    hist = hist[hist["fecha"] < target_date].dropna(subset=["categoria"])

    A1, all1 = _build_adjacency(h3_indexes, ring_k=1)

    records: dict[str, np.ndarray] = {}
    for short, cat in zip(CAT_SHORTS, CLASES):
        cat_hist = hist[hist["categoria"] == cat]
        if cat_hist.empty:
            for w in [7, 14]:
                records[f"nr_cat_{short}_ring1_{w}d"] = np.zeros(len(h3_indexes))
            continue

        pivot = (
            cat_hist.assign(_one=1.0)
            .pivot_table(index="h3_index", columns="fecha", values="_one", aggfunc="sum")
            .fillna(0.0)
        )
        for w in [7, 14]:
            v = _crime_vector(all1, pivot, target_date, w)
            records[f"nr_cat_{short}_ring1_{w}d"] = (A1 @ v).astype(float)

    return pd.DataFrame(records, index=pd.Index(h3_indexes, name="h3_index"))
