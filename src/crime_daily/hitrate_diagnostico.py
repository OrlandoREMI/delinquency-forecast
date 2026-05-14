import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
FEATURES_PATH = ROOT / "data/processed/features_daily.parquet"
MODEL_PATH    = ROOT / "models/lgbm_daily_v1.pkl"

FEATURE_COLS = [
    "log_lam_dia_base", "dia_semana", "es_fin_semana", "es_festivo",
    "mes", "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    "zona_geografica", "clave_mun",
    "nr_ring1_1d", "nr_ring1_7d", "nr_ring1_14d",
    "nr_ring2_7d",
]

print("Cargando test 2025...")
test = pd.read_parquet(FEATURES_PATH, filters=[("año", ">=", 2025)])
print(f"  Filas: {len(test):,}  |  Crimenes reales: {int(test['conteo'].sum()):,}\n")

with open(MODEL_PATH, "rb") as f:
    art = pickle.load(f)
model = art["model"]

test = test.copy()

# Tres scores a comparar
test["score_azar"]    = 1.0                                           # baseline uniforme
test["score_etapa1"]  = np.exp(test["log_lam_dia_base"].values)       # λ_mensual / dias_mes
test["score_etapa12"] = np.clip(
    model.predict(test[FEATURE_COLS].values.astype("float32")), 0, None
)

total_crimes = int(test["conteo"].sum())
total_rows   = len(test)

CORTES = [0.001, 0.005, 0.01, 0.05, 0.10, 0.20]

def hit_rate_table(df, score_col, label):
    print(f"\n{'─'*64}")
    print(f"  {label}")
    print(f"{'─'*64}")
    print(f"  {'Corte':<8} {'% cap':>8} {'Lift':>8}")
    print(f"  {'-'*28}")
    lifts = {}
    for pct in CORTES:
        n_top    = max(1, int(total_rows * pct))
        top      = df.nlargest(n_top, score_col)
        captured = int(top["conteo"].sum())
        cap_pct  = captured / total_crimes * 100
        lift     = cap_pct / (pct * 100)
        lifts[pct] = lift
        print(f"  Top {pct*100:>5.1f}%   {cap_pct:>7.1f}%   {lift:>6.1f}x")
    # Gini
    s = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    cum_c = s["conteo"].cumsum().values.astype(float) / total_crimes
    cum_p = (np.arange(1, len(s) + 1)) / total_rows
    gini  = float(2 * np.trapezoid(cum_c, cum_p) - 1)
    print(f"\n  Gini: {gini:.4f}")
    return lifts

lifts_azar   = hit_rate_table(test, "score_azar",    "AZAR (baseline uniforme)")
lifts_e1     = hit_rate_table(test, "score_etapa1",  "ETAPA 1 — solo modelo mensual (λ/días)")
lifts_e12    = hit_rate_table(test, "score_etapa12", "ETAPA 1+2 — modelo diario completo")

# Resumen: cuánto agrega la Etapa 2 sobre la Etapa 1
print(f"\n{'═'*64}")
print(f"  DESCOMPOSICIÓN DEL LIFT")
print(f"{'═'*64}")
print(f"  {'Corte':<8} {'E1 lift':>10} {'E1+2 lift':>12} {'Aporte E2':>12} {'E2/E1':>10}")
print(f"  {'-'*52}")
for pct in CORTES:
    l1  = lifts_e1[pct]
    l12 = lifts_e12[pct]
    delta = l12 - l1
    ratio = l12 / l1 if l1 > 0 else float("inf")
    print(f"  Top {pct*100:>5.1f}%   {l1:>9.1f}x   {l12:>11.1f}x   {delta:>+10.1f}x   {ratio:>8.2f}x")
