"""
ReliefF selector using scikit-learn's ReliefF implementation (if available).
Falls back to manual k-NN based ReliefF similar to manual relief_f.py but with sklearn utilities.
- Computes ReliefF weights with k nearest neighbors.
- Supports top-k selection and cross-validated k search.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score


@dataclass
class ReliefFResult:
    selected_mask: np.ndarray
    selected_feature_names: List[str]
    X_selected: np.ndarray
    y: np.ndarray
    weights: np.ndarray


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    y = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    feature_names = list(df.columns[1:])
    return X, y, feature_names


def reliefF_weights(X: np.ndarray, y: np.ndarray, k: int = 10) -> np.ndarray:
    """Compute ReliefF weights using min-max scaling and k-NN."""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    y = np.asarray(y)
    m, d = X_scaled.shape

    classes, counts = np.unique(y, return_counts=True)
    priors = {c: cnt / m for c, cnt in zip(classes, counts)}

    weights = np.zeros(d, dtype=float)

    for i in range(m):
        xi = X_scaled[i]
        yi = y[i]

        diff = X_scaled - xi
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        dists[i] = np.inf

        # Hits: same class
        same_mask = y == yi
        hit_candidates = np.where(same_mask)[0]
        hit_dists = dists[hit_candidates]
        hit_idx_sorted = hit_candidates[np.argsort(hit_dists)]
        hits = hit_idx_sorted[:k]

        # Misses: other classes
        miss_classes = [c for c in classes if c != yi]
        misses_per_class = {}
        for c in miss_classes:
            miss_candidates = np.where(y == c)[0]
            miss_dists = dists[miss_candidates]
            miss_idx_sorted = miss_candidates[np.argsort(miss_dists)]
            misses_per_class[c] = miss_idx_sorted[:k]

        # Update weights
        if len(hits) > 0:
            diffs_hit = np.abs(X_scaled[hits] - xi)
            weights -= diffs_hit.sum(axis=0) / (m * len(hits))

        for c, idxs in misses_per_class.items():
            if len(idxs) == 0:
                continue
            weight_c = priors[c] / max(1e-12, 1.0 - priors[yi])
            diffs_miss = np.abs(X_scaled[idxs] - xi)
            weights += weight_c * diffs_miss.sum(axis=0) / (m * len(idxs))

    return weights


def select_by_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    n_features = len(scores)
    k = max(0, min(k, n_features))
    order = np.argsort(scores)[::-1]
    selected_idx = order[:k]
    mask = np.zeros(n_features, dtype=bool)
    mask[selected_idx] = True
    return mask


def run(
    csv_path: Path,
    threshold: float,
    top_k: int | None,
    n_neighbors: int,
    log_file: Path | None,
    cv_topk_grid: list[int] | None,
    cv_folds: int,
    cv_metric: str,
) -> None:
    X, y, feature_names = load_dataset(csv_path)

    weights = reliefF_weights(X, y, k=n_neighbors)

    cv_summary = None
    if cv_topk_grid:
        cv_scores = []
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        for k in cv_topk_grid:
            k = max(1, min(k, X.shape[1]))
            mask_k = select_by_top_k(weights, k)
            X_k = X[:, mask_k]

            fold_metrics = []
            for train_idx, test_idx in skf.split(X_k, y):
                X_train, X_test = X_k[train_idx], X_k[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                clf = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "logreg",
                            LogisticRegression(
                                max_iter=500,
                                multi_class="auto",
                                solver="lbfgs",
                            ),
                        ),
                    ]
                )

                clf.fit(X_train, y_train)
                if cv_metric == "accuracy":
                    score = clf.score(X_test, y_test)
                elif cv_metric == "macro-f1":
                    y_pred = clf.predict(X_test)
                    score = f1_score(y_test, y_pred, average="macro")
                else:
                    score = clf.score(X_test, y_test)
                fold_metrics.append(score)

            mean_score = float(np.mean(fold_metrics)) if fold_metrics else 0.0
            cv_scores.append((k, mean_score))

        cv_scores.sort(key=lambda t: (-t[1], t[0]))
        best_k, best_score = cv_scores[0]
        top_k = best_k
        cv_summary = {"scores": cv_scores, "best_k": best_k, "best_score": best_score}

    if top_k is not None:
        mask = select_by_top_k(weights, top_k)
    else:
        mask = weights >= threshold

    selected_names = [name for name, keep in zip(feature_names, mask) if keep]
    X_selected = X[:, mask]

    report_lines: List[str] = []
    report_lines.append(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append(f"Dataset: {csv_path.name}")
    report_lines.append(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
    mode = f"top_k={top_k}" if top_k is not None else f"threshold={threshold}"
    report_lines.append(f"Selection mode: {mode} | n_neighbors={n_neighbors}")
    report_lines.append(f"Selected features: {len(selected_names)} / {len(feature_names)}")

    if cv_summary is not None:
        report_lines.append(f"Cross-validation top-k search (mean {cv_metric}):")
        for k, sc in cv_summary["scores"]:
            report_lines.append(f"  k={k}: {cv_metric}={sc:.4f}")
        report_lines.append(
            f"Chosen k={cv_summary['best_k']} ({cv_metric}={cv_summary['best_score']:.4f})"
        )

    dropped = [name for name, keep in zip(feature_names, mask) if not keep]
    if dropped:
        report_lines.append("Dropped features (below cut):")
        report_lines.append(", ".join(dropped))
    else:
        report_lines.append("No features were dropped.")

    top_order = np.argsort(weights)[::-1][:5]
    top_preview = [(feature_names[i], weights[i]) for i in top_order]
    report_lines.append("")
    report_lines.append("Top 5 ReliefF weights (name, weight):")
    report_lines.append(str(top_preview))

    report = "\n".join(report_lines)
    print(report)

    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(report + "\n\n---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReliefF selector using scikit-learn")
    parser.add_argument("csv", type=Path, help="CSV file with label in first column")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Keep features whose ReliefF weight is >= threshold",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Select the top-k features by ReliefF weight",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
        help="Number of nearest neighbors (hits/misses) for ReliefF",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Append the run report to this file",
    )
    parser.add_argument(
        "--cv-topk-grid",
        type=str,
        help="Comma-separated list of candidate k values for cross-validation",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )
    parser.add_argument(
        "--cv-metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "macro-f1"],
        help="Metric for cross-validation",
    )

    args = parser.parse_args()

    cv_grid = None
    if args.cv_topk_grid:
        cv_grid = [int(x) for x in args.cv_topk_grid.split(",") if x.strip()]

    run(
        csv_path=args.csv,
        threshold=args.threshold,
        top_k=args.top_k,
        n_neighbors=args.n_neighbors,
        log_file=args.log_file,
        cv_topk_grid=cv_grid,
        cv_folds=args.cv_folds,
        cv_metric=args.cv_metric,
    )
