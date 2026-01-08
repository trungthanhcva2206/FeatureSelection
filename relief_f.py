"""
Manual ReliefF feature selector.
- Assumes the first column of the CSV is the label (categorical).
- Scales features to [0,1] per column, then computes ReliefF weights with k nearest hits/misses.
- Supports selection by threshold or top-k, optional cross-validated search for k.
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
class ReliefFStats:
    weights: np.ndarray
    n_neighbors: int
    epsilon: float


@dataclass
class ReliefFResult:
    selected_mask: np.ndarray
    selected_feature_names: List[str]
    X_selected: np.ndarray
    y: np.ndarray
    stats: ReliefFStats


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    y = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    feature_names = list(df.columns[1:])
    return X, y, feature_names


def minmax_scale(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    denom = maxs - mins
    denom[denom < eps] = 1.0  # avoid zero division; constant cols stay 0 after centering
    return (X - mins) / denom


def reliefF_scores(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 10,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute ReliefF weights using L1 diffs on min-max scaled features."""

    Xn = minmax_scale(X, eps=epsilon)
    y = np.asarray(y)
    m, d = Xn.shape

    classes, counts = np.unique(y, return_counts=True)
    priors = {c: cnt / m for c, cnt in zip(classes, counts)}

    weights = np.zeros(d, dtype=float)

    for i in range(m):
        xi = Xn[i]
        yi = y[i]

        diff = Xn - xi  # broadcast
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        dists[i] = np.inf

        # hits: same class
        same_mask = y == yi
        hit_candidates = np.where(same_mask)[0]
        hit_dists = dists[hit_candidates]
        hit_idx_sorted = hit_candidates[np.argsort(hit_dists)]
        hits = hit_idx_sorted[:n_neighbors]

        # misses: per other class
        miss_classes = [c for c in classes if c != yi]
        misses_per_class = {}
        for c in miss_classes:
            miss_candidates = np.where(y == c)[0]
            miss_dists = dists[miss_candidates]
            miss_idx_sorted = miss_candidates[np.argsort(miss_dists)]
            misses_per_class[c] = miss_idx_sorted[:n_neighbors]

        # update weights
        if len(hits) > 0:
            diffs_hit = np.abs(Xn[hits] - xi)
            weights -= diffs_hit.sum(axis=0) / (m * len(hits))

        for c, idxs in misses_per_class.items():
            if len(idxs) == 0:
                continue
            weight_c = priors[c] / max(1e-12, 1.0 - priors[yi])
            diffs_miss = np.abs(Xn[idxs] - xi)
            weights += weight_c * diffs_miss.sum(axis=0) / (m * len(idxs))

    return weights


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


def run(
    csv_path: Path,
    threshold: float,
    top_k: int | None,
    n_neighbors: int,
    epsilon: float,
    log_file: Path | None,
    cv_topk_grid: list[int] | None,
    cv_folds: int,
    cv_metric: str,
) -> None:
    X, y, feature_names = load_dataset(csv_path)

    scores = reliefF_scores(X, y, n_neighbors=n_neighbors, epsilon=epsilon)

    cv_summary = None
    if cv_topk_grid:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import f1_score
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for CV selection. Install with `pip install scikit-learn`."
            ) from e

        cv_scores = []
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        for k in cv_topk_grid:
            k = max(1, min(k, X.shape[1]))
            mask_k, _ = select_features(scores=scores, feature_names=feature_names, top_k=k)
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

    mask, selected_names = select_features(
        scores=scores,
        feature_names=feature_names,
        top_k=top_k,
        threshold=threshold,
    )
    X_selected = X[:, mask]

    stats = ReliefFStats(weights=scores, n_neighbors=n_neighbors, epsilon=epsilon)
    result = ReliefFResult(
        selected_mask=mask,
        selected_feature_names=selected_names,
        X_selected=X_selected,
        y=y,
        stats=stats,
    )

    report_lines: List[str] = []
    report_lines.append(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append(f"Dataset: {csv_path.name}")
    report_lines.append(f"Samples: {X.shape[0]} | Features: {X.shape[1]}" )
    mode = f"top_k={top_k}" if top_k is not None else f"threshold={threshold}"
    report_lines.append(
        f"Selection mode: {mode} | n_neighbors={n_neighbors} | epsilon={epsilon}"
    )
    report_lines.append(
        f"Selected features: {len(result.selected_feature_names)} / {len(feature_names)}"
    )

    if cv_summary is not None:
        report_lines.append(f"Cross-validation top-k search (mean {cv_metric}):")
        for k, sc in cv_summary["scores"]:
            report_lines.append(f"  k={k}: {cv_metric}={sc:.4f}")
        report_lines.append(
            f"Chosen k={cv_summary['best_k']} ({cv_metric}={cv_summary['best_score']:.4f})"
        )

    dropped = [name for name, keep in zip(feature_names, result.selected_mask) if not keep]
    if dropped:
        report_lines.append("Dropped features (below cut):")
        report_lines.append(", ".join(dropped))
    else:
        report_lines.append("No features were dropped.")

    top_order = np.argsort(scores)[::-1][:5]
    top_preview = [(feature_names[i], scores[i]) for i in top_order]
    report_lines.append("")
    report_lines.append("Top 5 ReliefF scores (name, score):")
    report_lines.append(str(top_preview))

    report = "\n".join(report_lines)
    print(report)

    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(report + "\n\n---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual ReliefF feature selector")
    parser.add_argument("csv", type=Path, help="CSV file with label in first column")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Keep features whose ReliefF weight is >= threshold (ignored if top-k is set)",
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
        "--epsilon",
        type=float,
        default=1e-12,
        help="Small value to avoid division by zero",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Append the run report to this file",
    )
    parser.add_argument(
        "--cv-topk-grid",
        type=str,
        help="Comma-separated list of candidate k values to search with cross-validation",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation when searching k",
    )
    parser.add_argument(
        "--cv-metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "macro-f1"],
        help="Metric for cross-validation model scoring",
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
        epsilon=args.epsilon,
        log_file=args.log_file,
        cv_topk_grid=cv_grid,
        cv_folds=args.cv_folds,
        cv_metric=args.cv_metric,
    )
