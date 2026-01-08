"""
Manual Variance Threshold feature selector.
- Computes per-feature mean, variance, and standard deviation.
- Optionally normalizes (z-score) before applying the variance threshold.
- Keeps only features whose variance (raw or normalized) meets the threshold.

Assumes the first column of the CSV is the label and the rest are features.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class VarianceStats:
    means: np.ndarray
    variances: np.ndarray
    stds: np.ndarray
    variances_after_norm: np.ndarray | None


@dataclass
class VarianceResult:
    selected_mask: np.ndarray
    selected_feature_names: List[str]
    X_selected: np.ndarray
    y: np.ndarray
    stats: VarianceStats


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    y = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    feature_names = list(df.columns[1:])
    return X, y, feature_names


def variance_threshold(
    X: np.ndarray,
    feature_names: Iterable[str],
    threshold: float = 0.0,
    normalize_before: bool = False,
    eps: float = 1e-8,
) -> VarianceResult:
    """Apply a manual variance threshold.

    Steps:
    1) Compute per-feature mean, variance, std on the raw data.
    2) Optionally normalize each feature: (x - mean) / (std + eps).
    3) Recompute variance on the normalized data if requested.
    4) Keep features whose variance (raw or normalized) is >= threshold.
    """

    means = X.mean(axis=0)
    variances = ((X - means) ** 2).mean(axis=0)
    stds = np.sqrt(variances)

    if normalize_before:
        X_norm = (X - means) / (stds + eps)
        variances_after_norm = X_norm.var(axis=0)
        used_variances = variances_after_norm
    else:
        variances_after_norm = None
        used_variances = variances

    mask = used_variances >= threshold
    selected_feature_names = [name for name, keep in zip(feature_names, mask) if keep]
    X_selected = X[:, mask]

    stats = VarianceStats(
        means=means,
        variances=variances,
        stds=stds,
        variances_after_norm=variances_after_norm,
    )

    return VarianceResult(
        selected_mask=mask,
        selected_feature_names=selected_feature_names,
        X_selected=X_selected,
        y=None,  # filled by caller when labels are present
        stats=stats,
    )


def run(csv_path: Path, threshold: float, normalize_before: bool) -> None:
    X, y, feature_names = load_dataset(csv_path)
    result = variance_threshold(
        X=X,
        feature_names=feature_names,
        threshold=threshold,
        normalize_before=normalize_before,
    )
    result.y = y

    report_lines = []
    report_lines.append(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append(f"Dataset: {csv_path.name}")
    report_lines.append(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
    report_lines.append(f"Threshold: {threshold} | Normalize before threshold: {normalize_before}")
    report_lines.append(
        f"Selected features: {len(result.selected_feature_names)} / {len(feature_names)}"
    )

    dropped = [name for name, keep in zip(feature_names, result.selected_mask) if not keep]
    if dropped:
        report_lines.append("Dropped features (below threshold):")
        report_lines.append(", ".join(dropped))
    else:
        report_lines.append("No features were dropped.")

    report_lines.append("")
    report_lines.append("First 5 variances (raw):")
    report_lines.append(str(result.stats.variances[:5]))
    if normalize_before and result.stats.variances_after_norm is not None:
        report_lines.append("First 5 variances (after normalization):")
        report_lines.append(str(result.stats.variances_after_norm[:5]))

    report = "\n".join(report_lines)
    print(report)
    if args.log_file is not None:
        log_path = Path(args.log_file)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(report + "\n\n---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Variance Threshold")
    parser.add_argument("csv", type=Path, help="CSV file with label in first column")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Keep features whose variance is >= threshold",
    )
    parser.add_argument(
        "--normalize-before",
        action="store_true",
        help="Z-score normalize features before computing variance for threshold",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Append the run report to this file",
    )

    args = parser.parse_args()
    run(csv_path=args.csv, threshold=args.threshold, normalize_before=args.normalize_before)
