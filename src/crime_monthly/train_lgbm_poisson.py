"""
Entrena LightGBM para pronóstico de conteo de delitos por H3 × mes.
- Diagnostica sobredispersión para elegir objective (poisson vs tweedie)
- Split temporal: train ≤2023, val 2024, test 2025
- Reporta métricas por conjunto y por zona geográfica
- Serializa modelo en models/
"""
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

INPUT = Path(__file__).parent.parent.parent / "data/processed/features_temporal.parquet"
MODEL_DIR = Path(__file__).parent.parent.parent / "models"

CAT_COLS = ["zona_geografica", "region", "trimestre", "clave_mun"]
DROP_COLS = ["h3_9", "año_mes", "municipio", "conteo"]

FEATURE_COLS = None  # se calcula dinámicamente


def overdispersion_test(y: pd.Series) -> tuple[float, str]:
    """Calcula ratio varianza/media. >2 → Tweedie; ≤2 → Poisson."""
    nonzero = y[y > 0]
    ratio = nonzero.var() / nonzero.mean()
    objective = "tweedie" if ratio > 2 else "poisson"
    return ratio, objective


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"  {label:<20} MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.1f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def main():
    print("Cargando features_temporal.parquet...")
    df = pd.read_parquet(INPUT)
    print(f"  Shape: {df.shape}")

    df, encoders = encode_categoricals(df)

    train_df = df[df["año_mes"].dt.year <= 2023].copy()
    val_df   = df[df["año_mes"].dt.year == 2024].copy()
    test_df  = df[df["año_mes"].dt.year == 2025].copy()

    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    X_train = train_df[feature_cols].values
    y_train = train_df["conteo"].values.astype(float)
    X_val   = val_df[feature_cols].values
    y_val   = val_df["conteo"].values.astype(float)
    X_test  = test_df[feature_cols].values
    y_test  = test_df["conteo"].values.astype(float)

    print(f"\n  Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")

    ratio, objective = overdispersion_test(train_df["conteo"])
    print(f"\nDiagnóstico de sobredispersión:")
    print(f"  Varianza/Media (valores > 0): {ratio:.2f}")
    print(f"  Objetivo seleccionado: {objective.upper()}")

    tweedie_power = 1.5 if objective == "tweedie" else None

    params = {
        "objective": objective,
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "verbose": -1,
        "seed": 42,
    }
    if objective == "tweedie":
        params["tweedie_variance_power"] = tweedie_power

    print(f"\nEntrenando LightGBM ({objective})...")
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    print(f"\n  Mejor iteración: {model.best_iteration}")

    pred_val  = np.clip(model.predict(X_val),  0, None)
    pred_test = np.clip(model.predict(X_test), 0, None)

    print("\nMétricas globales:")
    evaluate(y_val,  pred_val,  "Val  (2024)")
    evaluate(y_test, pred_test, "Test (2025)")

    print("\nMétricas val (2024) por zona geográfica:")
    for zona_code in sorted(val_df["zona_geografica"].unique()):
        mask = val_df["zona_geografica"].values == zona_code
        zona_name = encoders["zona_geografica"].inverse_transform([zona_code])[0]
        evaluate(y_val[mask], pred_val[mask], zona_name)

    print("\nTop 15 features por importancia (gain):")
    imp = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=feature_cols,
    ).sort_values(ascending=False)
    for feat, val_ in imp.head(15).items():
        print(f"  {feat:<30} {val_:,.0f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "lgbm_poisson_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "encoders": encoders,
                     "feature_cols": feature_cols, "objective": objective,
                     "overdispersion_ratio": ratio}, f)
    print(f"\nModelo guardado en {model_path}")


if __name__ == "__main__":
    main()
