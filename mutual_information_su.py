"""
Symmetrical Uncertainty (SU) feature scorer and selector.
- Assumes label column either first or last (configurable).
- Computes SU via quantile binning.
- Supports top-k or threshold selection utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SUScores:
    su_scores: np.ndarray
    bins_used: int


@dataclass
class SUResult:
    selected_mask: np.ndarray
    selected_feature_names: List[str]
    X_selected: np.ndarray
    y: np.ndarray
    stats: SUScores


def load_dataset(csv_path: Path, sep: str = ",", label_last: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path, sep=sep)
    if label_last:
        y = df.iloc[:, -1].to_numpy()
        X = df.iloc[:, :-1].to_numpy(dtype=float)
        feature_names = list(df.columns[:-1])
    else:
        y = df.iloc[:, 0].to_numpy()
        X = df.iloc[:, 1:].to_numpy(dtype=float)
        feature_names = list(df.columns[1:])
    return X, y, feature_names


def symmetrical_uncertainty_scores(
    X: np.ndarray,
    y: np.ndarray,
    bins: int = 10,
    smoothing: float = 1e-9,
) -> np.ndarray:
    """Compute SU(X;Y) = 2 * I(X;Y) / (H(X) + H(Y)) per feature via binning."""

    y = np.asarray(y)
    classes, y_idx = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    n_features = X.shape[1]

    label_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
    label_counts += smoothing
    p_y = label_counts / label_counts.sum()
    H_y = -np.sum(p_y * np.log(p_y))

    scores = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        col = X[:, j]
        if np.all(col == col[0]):
            scores[j] = 0.0
            continue

        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(col, quantiles)
        edges = np.unique(edges)
        if len(edges) < 2:
            scores[j] = 0.0
            continue

        bin_ids = np.digitize(col, edges[1:-1], right=False)
        n_bins = len(edges) - 1

        contingency = np.zeros((n_classes, n_bins), dtype=float)
        for cls in range(n_classes):
            cls_mask = y_idx == cls
            np.add.at(contingency[cls], bin_ids[cls_mask], 1.0)

        contingency += smoothing
        total = contingency.sum()
        p_xy = contingency / total
        p_x = p_xy.sum(axis=1)
        p_z = p_xy.sum(axis=0)

        ratio = p_xy / (p_x[:, None] * p_z[None, :])
        mi = (p_xy * np.log(ratio)).sum()

        H_x = -np.sum(p_z * np.log(p_z))
        su = (2.0 * mi) / (H_x + H_y + 1e-12)
        scores[j] = su

    return scores


def select_features(
    scores: np.ndarray,
    feature_names: Iterable[str],
    top_k: int | None = None,
    threshold: float = 0.0,
) -> tuple[np.ndarray, List[str]]:
    names = list(feature_names)
    n_features = len(names)

    if top_k is not None:
        top_k = max(0, min(top_k, n_features))
        order = np.argsort(scores)[::-1]
        selected_idx = order[:top_k]
        mask = np.zeros(n_features, dtype=bool)
        mask[selected_idx] = True
    else:
        mask = scores >= threshold

    selected_names = [name for name, keep in zip(names, mask) if keep]
    return mask, selected_names
