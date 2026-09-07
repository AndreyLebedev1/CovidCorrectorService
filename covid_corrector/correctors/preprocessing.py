"""Inference-only PCA, whitening and L2 transformation."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class PCAPreprocessor:
    """Applies a previously fitted PCA transformation."""

    n_components: int | None = 50
    whiten: bool = True
    l2_normalize: bool = True
    random_state: int = 42
    svd_solver: str = "auto"
    pca: PCA | None = None
    scale_: np.ndarray | None = None

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("The preprocessor is not fitted")
        result = self.pca.transform(np.asarray(x, dtype=np.float64))
        if self.whiten and self.scale_ is not None:
            result = result / self.scale_
        if self.l2_normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / np.maximum(norms, 1e-12)
        return result.astype(np.float32)

    @classmethod
    def load(cls, path: str) -> PCAPreprocessor:
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Unexpected preprocessor artifact: {type(value)}")
        return value

