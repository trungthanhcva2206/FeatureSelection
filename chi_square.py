"""
Manual Chi-square feature selector.
- Assumes the first column of the CSV is the label (categorical).
- Bins each feature (quantile bins) to build a contingency table with the label.
- Computes Chi-square statistic per feature and keeps top features by threshold or top-k.

Note: Chi-square expects non-negative counts. This script bins continuous values
so it can work on gene-expression-like datasets.
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SKLEARN_AVAILABLE = False


@dataclass
class ChiSquareStats:
    chi2_scores: np.ndarray
    bins_used: int


@dataclass
class ChiSquareResult:
    selected_mask: np.ndarray
    selected_feature_names: List[str]
    X_selected: np.ndarray
    y: np.ndarray
    stats: ChiSquareStats


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    y = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    feature_names = list(df.columns[1:])
    return X, y, feature_names


def chi_square_scores(
    X: np.ndarray,
    y: np.ndarray,
    bins: int | None = 10,
    smoothing: float = 1e-9,
    use_raw: bool = False,
) -> np.ndarray:
    """Compute Chi-square score per feature.

    - If use_raw=True: shift features to non-negative and use sklearn's chi2-like formula without binning.
    - Else: bin each feature into quantile-based bins and compute contingency chi-square.
    """

    y = np.asarray(y)
    classes, y_idx = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    n_features = X.shape[1]

    chi2_scores = np.zeros(n_features, dtype=float)

    if use_raw:
        # Match sklearn's chi2: treat feature values as non-negative counts and
        # build a contingency table of shape (n_classes, n_features) using sums per class.
        X_shift = X - np.minimum(0, X.min(axis=0))

        # If everything is zero, return zeros.
        if np.all(X_shift == 0):
            return chi2_scores

        observed = np.zeros((n_classes, n_features), dtype=float)
        for cls in range(n_classes):
            cls_mask = y_idx == cls
            if not np.any(cls_mask):
                continue
            # Sum of each feature within the class (same as Y^T X in sklearn)
            observed[cls] = X_shift[cls_mask].sum(axis=0)

        row_sum = observed.sum(axis=1, keepdims=True)
        col_sum = observed.sum(axis=0, keepdims=True)
        total = observed.sum()

        if total == 0:
            return chi2_scores

        expected = row_sum * col_sum / total
        chi = ((observed - expected) ** 2) / (expected + smoothing)
        chi2_scores = chi.sum(axis=0)
        return chi2_scores

    # Binned mode (original behavior)
    for j in range(n_features):
        col = X[:, j]

        # If constant feature, score is zero.
        if np.all(col == col[0]):
            chi2_scores[j] = 0.0
            continue

        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(col, quantiles)
        edges = np.unique(edges)

        # If not enough unique edges, treat as constant.
        if len(edges) < 2:
            chi2_scores[j] = 0.0
            continue

        bin_ids = np.digitize(col, edges[1:-1], right=False)
        n_bins = len(edges) - 1

        contingency = np.zeros((n_classes, n_bins), dtype=float)
        for cls in range(n_classes):
            cls_mask = y_idx == cls
            np.add.at(contingency[cls], bin_ids[cls_mask], 1.0)

        row_sum = contingency.sum(axis=1, keepdims=True)
        col_sum = contingency.sum(axis=0, keepdims=True)
        total = contingency.sum()
        expected = row_sum * col_sum / max(total, 1.0)

        chi = ((contingency - expected) ** 2) / (expected + smoothing)
        chi2_scores[j] = chi.sum()

    return chi2_scores


def select_features(
    chi2_scores: np.ndarray,
    feature_names: Iterable[str],
    top_k: int | None = None,
    threshold: float = 0.0,
) -> tuple[np.ndarray, List[str]]:
    names = list(feature_names)
    n_features = len(names)

    if top_k is not None:
        top_k = max(0, min(top_k, n_features))
        order = np.argsort(chi2_scores)[::-1]
        selected_idx = order[:top_k]
        mask = np.zeros(n_features, dtype=bool)
        mask[selected_idx] = True
    else:
        mask = chi2_scores >= threshold

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
    use_raw: bool,
) -> None:
    X, y, feature_names = load_dataset(csv_path)

    chi2 = chi_square_scores(X, y, bins=bins, use_raw=use_raw)

    cv_summary = None
    if cv_topk_grid:
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for CV selection. Install with `pip install scikit-learn`."
            )

        scores = []
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        for k in cv_topk_grid:
            k = max(1, min(k, X.shape[1]))
            mask_k, _ = select_features(
                chi2_scores=chi2, feature_names=feature_names, top_k=k
            )
            X_k = X[:, mask_k]

            fold_scores = []
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
                else:  # fallback
                    score = clf.score(X_test, y_test)
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
            scores.append((k, mean_score))

        scores.sort(key=lambda t: (-t[1], t[0]))
        best_k, best_score = scores[0]
        top_k = best_k
        cv_summary = {"scores": scores, "best_k": best_k, "best_score": best_score}

    mask, selected_names = select_features(
        chi2_scores=chi2,
        feature_names=feature_names,
        top_k=top_k,
        threshold=threshold,
    )
    X_selected = X[:, mask]

    stats = ChiSquareStats(chi2_scores=chi2, bins_used=bins)
    result = ChiSquareResult(
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
    report_lines.append(f"Selection mode: {mode} | bins={bins}")
    report_lines.append(
        f"Selected features: {len(result.selected_feature_names)} / {len(feature_names)}"
    )

    if cv_summary is not None:
        report_lines.append(
            f"Cross-validation top-k search (mean {cv_metric}):"
        )
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

    # Show top 5 scores for quick inspection
    top_order = np.argsort(chi2)[::-1][:5]
    top_preview = [(feature_names[i], chi2[i]) for i in top_order]
    report_lines.append("")
    report_lines.append("Top 5 chi-square scores (name, score):")
    report_lines.append(str(top_preview))

    report = "\n".join(report_lines)
    print(report)

    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(report + "\n\n---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Chi-square feature selector")
    parser.add_argument("csv", type=Path, help="CSV file with label in first column")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Keep features whose chi-square score is >= threshold (ignored if top-k is set)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Select the top-k features by chi-square score",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of quantile bins per feature",
    )
    parser.add_argument(
        "--use-raw",
        action="store_true",
        help="Compute chi-square on raw non-negative shifted data (no binning)",
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
        use_raw=args.use_raw,
    )
