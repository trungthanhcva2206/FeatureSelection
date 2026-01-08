"""
Manual Mutual Information feature selector.
- Assumes the first column of the CSV is the label (categorical).
- Computes mutual information between each feature and the label.
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

try:
    from sklearn.feature_selection import mutual_info_classif

    SKLEARN_MI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional
    SKLEARN_MI_AVAILABLE = False


@dataclass
class MIScores:
    mi_scores: np.ndarray
    bins_used: int


@dataclass
class MIResult:
    selected_mask: np.ndarray
    selected_feature_names: List[str]
    X_selected: np.ndarray
    y: np.ndarray
    stats: MIScores


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    y = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    feature_names = list(df.columns[1:])
    return X, y, feature_names


def mi_scores(
    X: np.ndarray,
    y: np.ndarray,
    bins: int = 10,
    smoothing: float = 1e-9,
    use_knn: bool = False,
    n_neighbors: int = 3,
    use_knn_manual: bool = False,
) -> np.ndarray:
    """Estimate MI per feature.

    - use_knn=True: use sklearn's k-NN MI estimator (continuous), no binning.
    - use_knn_manual=True: custom k-NN estimator (continuous), no sklearn.
    - else: discretize into quantile bins and compute MI on counts.
    """

    if use_knn_manual:
        return _mi_knn_manual(X, y, k=n_neighbors)

    if use_knn:
        if not SKLEARN_MI_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for k-NN MI estimation. Install with `pip install scikit-learn`."
            )
        return mutual_info_classif(X, y, n_neighbors=n_neighbors, random_state=42)

    y = np.asarray(y)
    classes, y_idx = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    n_features = X.shape[1]

    scores = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        col = X[:, j]

        # Constant feature => MI = 0
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

        contingency += smoothing  # smoothing to avoid zeros
        total = contingency.sum()
        p_xy = contingency / total
        p_x = p_xy.sum(axis=1, keepdims=True)
        p_y = p_xy.sum(axis=0, keepdims=True)

        ratio = p_xy / (p_x * p_y)
        mi = (p_xy * np.log(ratio)).sum()
        scores[j] = mi

    return scores


def _mi_knn_manual(X: np.ndarray, y: np.ndarray, k: int = 3) -> np.ndarray:
    """Manual k-NN MI estimator for continuous features and discrete labels.

    For each feature (1D), we estimate $I(X;Y) = H(Y) - E[H(Y|X=x)]$ where
    $H(Y|X=x)$ is approximated from the class distribution within the k-nearest
    neighbors (including the point itself) based on absolute distance in X.

    This is a simple, local-probability estimator (not a full Kraskov), but it
    avoids sklearn dependency and works reasonably on small n.
    """

    y = np.asarray(y)
    X = np.asarray(X)
    n_samples, n_features = X.shape
    if n_samples == 0:
        return np.zeros(n_features, dtype=float)

    # Global label entropy H(Y)
    unique, counts = np.unique(y, return_counts=True)
    p_y = counts / counts.sum()
    H_y = -np.sum(p_y * np.log(p_y + 1e-12))

    scores = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        col = X[:, j]

        # Constant feature => MI = 0
        if np.all(col == col[0]):
            scores[j] = 0.0
            continue

        # For each sample, find epsilon = distance to k-th neighbor (include self)
        # Then collect neighbors within epsilon and compute local label entropy.
        H_cond_list: list[float] = []
        for idx in range(n_samples):
            dists = np.abs(col - col[idx])
            # kth neighbor distance (k includes self, so need k-th smallest)
            k_eff = min(k, n_samples)
            eps = np.partition(dists, k_eff - 1)[k_eff - 1]
            # neighbors within eps (inclusive)
            neigh_mask = dists <= eps + 1e-12
            neigh_labels = y[neigh_mask]
            cnts = np.bincount(neigh_labels.astype(int), minlength=int(unique.max()) + 1)
            cnts = cnts[cnts > 0]
            p_local = cnts / cnts.sum()
            H_local = -np.sum(p_local * np.log(p_local + 1e-12))
            H_cond_list.append(H_local)

        H_cond = float(np.mean(H_cond_list)) if H_cond_list else 0.0
        scores[j] = max(H_y - H_cond, 0.0)

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


def run(
    csv_path: Path,
    threshold: float,
    top_k: int | None,
    bins: int,
    log_file: Path | None,
    cv_topk_grid: list[int] | None,
    cv_folds: int,
    cv_metric: str,
    use_knn: bool,
    n_neighbors: int,
    use_knn_manual: bool,
) -> None:
    X, y, feature_names = load_dataset(csv_path)

    scores = mi_scores(
        X,
        y,
        bins=bins,
        use_knn=use_knn,
        n_neighbors=n_neighbors,
        use_knn_manual=use_knn_manual,
    )

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

    bins_used = 0 if use_knn_manual or use_knn else bins
    stats = MIScores(mi_scores=scores, bins_used=bins_used)
    result = MIResult(
        selected_mask=mask,
        selected_feature_names=selected_names,
        X_selected=X_selected,
        y=y,
        stats=stats,
    )

    report_lines: List[str] = []
    report_lines.append(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append(f"Dataset: {csv_path.name}")
    report_lines.append(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
    mode = f"top_k={top_k}" if top_k is not None else f"threshold={threshold}"
    if use_knn_manual:
        detail = f"knn-manual n_neighbors={n_neighbors}"
    elif use_knn:
        detail = f"knn n_neighbors={n_neighbors}"
    else:
        detail = f"bins={bins}"
    report_lines.append(f"Selection mode: {mode} | {detail}")
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
    report_lines.append("Top 5 MI scores (name, score):")
    report_lines.append(str(top_preview))

    report = "\n".join(report_lines)
    print(report)

    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(report + "\n\n---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Mutual Information feature selector")
    parser.add_argument("csv", type=Path, help="CSV file with label in first column")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Keep features whose MI score is >= threshold (ignored if top-k is set)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Select the top-k features by MI score",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of quantile bins per feature for MI estimation",
    )
    parser.add_argument(
        "--use-knn",
        action="store_true",
        help="Use sklearn k-NN MI estimator (continuous, no binning)",
    )
    parser.add_argument(
        "--use-knn-manual",
        action="store_true",
        help="Use manual k-NN MI estimator (continuous, no sklearn)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=3,
        help="Number of neighbors for k-NN MI",
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
        bins=args.bins,
        log_file=args.log_file,
        cv_topk_grid=cv_grid,
        cv_folds=args.cv_folds,
        cv_metric=args.cv_metric,
        use_knn=args.use_knn,
        n_neighbors=args.n_neighbors,
        use_knn_manual=args.use_knn_manual,
    )
