import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from ..schemas import CLASES


class E3Stage:
    def __init__(self, artifacts: dict):
        self.model        = artifacts["model"]
        self.feature_cols = artifacts["feature_cols"]
        self.calibrators  = artifacts.get("calibrators")
        self.thresholds   = artifacts.get("thresholds", np.ones(4))

    @classmethod
    def load(cls, path: Path) -> "E3Stage":
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        for col in self.feature_cols:
            if col not in features_df.columns:
                features_df = features_df.copy()
                features_df[col] = 0.0

        X = features_df[self.feature_cols].values.astype(float)
        probs = self.model.predict(X)

        if self.calibrators is not None:
            cal = np.column_stack([c.predict(probs[:, i]) for i, c in enumerate(self.calibrators)])
            cal = np.clip(cal, 1e-7, 1.0)
            probs = cal / cal.sum(axis=1, keepdims=True)

        return probs

    def predict_classes(self, probs: np.ndarray) -> list[str]:
        return [CLASES[i] for i in (probs * self.thresholds).argmax(axis=1)]
