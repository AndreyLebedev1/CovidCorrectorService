"""Inference-only Fisher corrector."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np


@dataclass
class FisherCorrector:
    """Applies a previously fitted COVID-vs-legacy discriminant."""

    regularization: float = 1e-6
    covariance_shrinkage: float = 0.0
    threshold: float = 0.0
    mean_legacy: np.ndarray | None = None
    mean_covid: np.ndarray | None = None
    covariance_: np.ndarray | None = None
    direction_: np.ndarray | None = None
    intercept_: float = 0.0
    n_covid: int = 0

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.direction_ is None:
            raise RuntimeError("The corrector is not fitted")
        return np.asarray(x) @ self.direction_ + self.intercept_

    def predict_covid(self, x: np.ndarray) -> np.ndarray:
        return self.decision_function(x) >= self.threshold

    @classmethod
    def load(cls, path: str) -> FisherCorrector:
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Unexpected corrector artifact: {type(value)}")
        return value

