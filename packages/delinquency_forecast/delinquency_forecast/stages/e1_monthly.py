import pickle
from pathlib import Path
import numpy as np
import pandas as pd


class E1Stage:
    def __init__(self, artifacts: dict):
        self.model        = artifacts["model"]
        self.encoders     = artifacts["encoders"]
        self.feature_cols = artifacts["feature_cols"]

    @classmethod
    def load(cls, path: Path) -> "E1Stage":
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        X = features_df[self.feature_cols].values.astype(float)
        return np.clip(self.model.predict(X), 0, None)
