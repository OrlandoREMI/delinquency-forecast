"""
Entrena el modelo diario LightGBM (Etapa 2 del pipeline de dos etapas).

El feature log_lam_dia_base codifica la predicción del modelo mensual como
punto de partida; el modelo diario aprende la desviación semanal y estacional
sobre esa base. Esto resuelve la limitación del factor DOW fijo: la interacción
entre nivel_mensual × día_semana se aprende directamente de los datos.

Split temporal: train ≤2023, val 2024, test 2025.
Solo entrena sobre H3 con al menos MIN_CRIMES_TRAIN crímenes totales en
el set de entrenamiento para evitar que los millones de celdas estructuralmente
vacías dominen el gradiente.
"""
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
INPUT = ROOT / "data/processed/features_daily.parquet"
MONTHLY_MODEL_PATH = ROOT / "models/lgbm_poisson_v1.pkl"
MODEL_DIR = ROOT / "models"

FEATURE_COLS_DAILY = [
    "log_lam_dia_base",
    "dia_semana", "es_fin_semana", "es_festivo",
    "mes",
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14",
    "rolling_std_7",
    "zona_geografica", "clave_mun",
    "nr_ring1_1d", "nr_ring1_7d", "nr_ring1_14d",
    "nr_ring2_7d",
]

MIN_CRIMES_TRAIN = 10   # crímenes mínimos en período de entrenamiento para incluir la celda


def mape(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true, y_pred, label):
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape_val = mape(y_true, y_pred)
    print(f"  {label:<25} MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape_val:.1f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape_val}


def main():
    print("Cargando features_daily.parquet...")
    df = pd.read_parquet(INPUT)
    print(f"  Shape total: {df.shape}")

    # Cargar encoders del modelo mensual para reutilizarlos
    with open(MONTHLY_MODEL_PATH, "rb") as f:
        monthly_art = pickle.load(f)
    encoders = monthly_art["encoders"]

    # Split temporal
    train_df = df[df["año"] <= 2023].copy()
    val_df   = df[df["año"] == 2024].copy()
    test_df  = df[df["año"] >= 2025].copy()

    # Filtrar celdas activas para el entrenamiento
    cell_activity = train_df.groupby("h3_9")["conteo"].sum()
    active_cells  = cell_activity[cell_activity >= MIN_CRIMES_TRAIN].index
    train_df = train_df[train_df["h3_9"].isin(active_cells)]

    print(f"\n  Celdas activas (>= {MIN_CRIMES_TRAIN} crímenes en train): {len(active_cells):,}")
    print(f"  Train filtrado: {len(train_df):,} filas")
    print(f"  Val:            {len(val_df):,} filas")
    print(f"  Test:           {len(test_df):,} filas")

    X_train = train_df[FEATURE_COLS_DAILY].values.astype(np.float32)
    X_val   = val_df[FEATURE_COLS_DAILY].values.astype(np.float32)
    X_test  = test_df[FEATURE_COLS_DAILY].values.astype(np.float32)

    # Entrenamiento y early-stopping sobre target suavizado
    y_train_smooth = train_df["target_smoothed"].values.astype(float)
    y_val_smooth   = val_df["target_smoothed"].values.astype(float)

    # Evaluación siempre sobre delitos reales
    y_val_real  = val_df["conteo"].values.astype(float)
    y_test_real = test_df["conteo"].values.astype(float)

    params = {
        "objective": "poisson",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "n_jobs": -1,
        "verbose": -1,
        "seed": 42,
    }

    print("\nEntrenando LightGBM diario (poisson)...")
    dtrain = lgb.Dataset(X_train, label=y_train_smooth, feature_name=FEATURE_COLS_DAILY)
    dval   = lgb.Dataset(X_val,   label=y_val_smooth,   reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    print(f"\n  Mejor iteración: {model.best_iteration}")

    pred_val  = np.clip(model.predict(X_val),  0, None)
    pred_test = np.clip(model.predict(X_test), 0, None)

    print("\nMétricas globales:")
    evaluate(y_val_real,  pred_val,  "Val  (2024)")
    evaluate(y_test_real, pred_test, "Test (2025)")

    # Métricas por zona geográfica (solo val)
    if "zona_geografica" in val_df.columns:
        print("\nMétricas val (2024) por zona geográfica:")
        for zona_code in sorted(val_df["zona_geografica"].unique()):
            mask = val_df["zona_geografica"].values == zona_code
            try:
                zona_name = encoders["zona_geografica"].inverse_transform([zona_code])[0]
            except Exception:
                zona_name = str(zona_code)
            evaluate(y_val_real[mask], pred_val[mask], zona_name)

    # Métricas por día de la semana (val)
    print("\nMétricas val (2024) por día de la semana:")
    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    for d in range(7):
        mask = val_df["dia_semana"].values == d
        if mask.sum() > 0:
            evaluate(y_val_real[mask], pred_val[mask], dias[d])

    # Importancia de features
    print("\nImportancia de features (gain):")
    imp = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=FEATURE_COLS_DAILY,
    ).sort_values(ascending=False)
    for feat, val_ in imp.items():
        print(f"  {feat:<25} {val_:,.0f}")

    # Guardar modelo
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "lgbm_daily_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "encoders": encoders,
            "feature_cols": FEATURE_COLS_DAILY,
            "objective": "poisson",
            "min_crimes_train_filter": MIN_CRIMES_TRAIN,
        }, f)
    print(f"\nModelo guardado en {model_path}")


if __name__ == "__main__":
    main()
