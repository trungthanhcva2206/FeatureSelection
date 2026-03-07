"""
Pipeline: train/test split -> MI-based top-k search (train CV) -> SVM fit -> forward wrapper on MI pool -> test metrics.
- Label column defaults to first; can switch to last via --label-last.
- MI scoring can use bins (default) or k-NN estimators from mutual_information.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mutual_information import mi_scores, select_features


@dataclass
class KSearchResult:
    best_k: int
    cv_scores: List[Tuple[int, float]]  # (k, mean metric)


@dataclass
class ForwardResult:
    selected_indices: List[int]
    selected_names: List[str]
    history: List[Tuple[int, str, float]]  # (step, feature, score)


def load_dataset(csv_path: Path, sep: str, label_last: bool) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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


def parse_int_grid(raw: str | None, default: Sequence[int]) -> List[int]:
    if not raw:
        return list(default)
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError:
            pass
    return values or list(default)


def parse_gamma(raw: str) -> str | float:
    lowered = raw.strip().lower()
    if lowered in {"scale", "auto"}:
        return lowered
    try:
        return float(lowered)
    except ValueError:
        return "scale"


def evaluate_svm_cv(
    X: np.ndarray,
    y: np.ndarray,
    metric: str,
    cv_folds: int,
    C: float,
    gamma: str | float,
    kernel: str,
    class_weight: str | None,
) -> float:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores: List[float] = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(C=C, gamma=gamma, kernel=kernel, class_weight=class_weight),
                ),
            ]
        )
        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_val)
        if metric == "macro-f1":
            score = f1_score(y_val, preds, average="macro")
        else:
            score = accuracy_score(y_val, preds)
        scores.append(float(score))

    return float(np.mean(scores)) if scores else 0.0


def search_best_k(
    X: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    feature_names: Iterable[str],
    k_grid: Iterable[int],
    metric: str,
    cv_folds: int,
    C: float,
    gamma: str | float,
    kernel: str,
    class_weight: str | None,
) -> KSearchResult:
    cv_scores: List[Tuple[int, float]] = []
    for k in k_grid:
        k = max(1, min(k, X.shape[1]))
        mask_k, _ = select_features(scores=scores, feature_names=feature_names, top_k=k)
        X_k = X[:, mask_k]
        mean_score = evaluate_svm_cv(
            X=X_k,
            y=y,
            metric=metric,
            cv_folds=cv_folds,
            C=C,
            gamma=gamma,
            kernel=kernel,
            class_weight=class_weight,
        )
        cv_scores.append((k, mean_score))

    cv_scores.sort(key=lambda t: (-t[1], t[0]))
    best_k = cv_scores[0][0] if cv_scores else 1
    return KSearchResult(best_k=best_k, cv_scores=cv_scores)


def forward_wrapper(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    candidate_indices: List[int],
    metric: str,
    cv_folds: int,
    C: float,
    gamma: str | float,
    kernel: str,
    class_weight: str | None,
    min_improvement: float,
) -> ForwardResult:
    selected: List[int] = []
    remaining = list(candidate_indices)
    history: List[Tuple[int, str, float]] = []
    current_score = 0.0
    step = 0

    while remaining:
        best_idx = None
        best_score = current_score

        for idx in remaining:
            subset = selected + [idx]
            score = evaluate_svm_cv(
                X=X[:, subset],
                y=y,
                metric=metric,
                cv_folds=cv_folds,
                C=C,
                gamma=gamma,
                kernel=kernel,
                class_weight=class_weight,
            )
            if score > best_score:
                best_score = score
                best_idx = idx

        improvement = best_score - current_score
        if best_idx is None or improvement < min_improvement:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        current_score = best_score
        step += 1
        history.append((step, feature_names[best_idx], current_score))

    if not selected and candidate_indices:
        selected = [candidate_indices[0]]
        history.append((1, feature_names[candidate_indices[0]], current_score))

    selected_names = [feature_names[i] for i in selected]
    return ForwardResult(selected_indices=selected, selected_names=selected_names, history=history)


def run(args: argparse.Namespace) -> None:
    X, y, feature_names = load_dataset(args.csv, sep=args.sep, label_last=args.label_last)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    mi = mi_scores(
        X_train,
        y_train,
        bins=args.bins,
        smoothing=1e-9,
        use_knn=args.use_knn,
        n_neighbors=args.n_neighbors,
        use_knn_manual=args.use_knn_manual,
    )

    k_grid = parse_int_grid(args.cv_topk_grid, default=args.default_k_grid)
    class_weight = "balanced" if args.class_weight_balanced else None
    gamma_value = parse_gamma(args.svm_gamma)

    k_result = search_best_k(
        X=X_train,
        y=y_train,
        scores=mi,
        feature_names=feature_names,
        k_grid=k_grid,
        metric=args.metric,
        cv_folds=args.cv_folds,
        C=args.svm_C,
        gamma=gamma_value,
        kernel=args.svm_kernel,
        class_weight=class_weight,
    )

    mask_best, selected_names_best = select_features(
        scores=mi,
        feature_names=feature_names,
        top_k=k_result.best_k,
        threshold=0.0,
    )
    X_train_best = X_train[:, mask_best]
    X_test_best = X_test[:, mask_best]

    pipeline_final = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=args.svm_C, gamma=gamma_value, kernel=args.svm_kernel)),
        ]
    )
    pipeline_final.fit(X_train_best, y_train)
    preds_test = pipeline_final.predict(X_test_best)
    test_acc = accuracy_score(y_test, preds_test)
    test_f1 = f1_score(y_test, preds_test, average="macro")

    candidate_indices = list(np.where(mask_best)[0])
    forward_result = forward_wrapper(
        X=X_train,
        y=y_train,
        feature_names=feature_names,
        candidate_indices=candidate_indices,
        metric=args.metric,
        cv_folds=args.cv_folds,
        C=args.svm_C,
        gamma=gamma_value,
        kernel=args.svm_kernel,
        class_weight=class_weight,
        min_improvement=args.min_improvement,
    )

    mask_forward = np.zeros_like(mask_best, dtype=bool)
    mask_forward[forward_result.selected_indices] = True
    X_train_fw = X_train[:, mask_forward]
    X_test_fw = X_test[:, mask_forward]

    pipeline_fw = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=args.svm_C, gamma=gamma_value, kernel=args.svm_kernel)),
        ]
    )
    pipeline_fw.fit(X_train_fw, y_train)
    preds_fw = pipeline_fw.predict(X_test_fw)
    test_acc_fw = accuracy_score(y_test, preds_fw)
    test_f1_fw = f1_score(y_test, preds_fw, average="macro")

    report_lines = []
    report_lines.append(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append(f"Dataset: {args.csv.name}")
    report_lines.append(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
    report_lines.append(f"Train/Test split: {1 - args.test_size:.2f}/{args.test_size:.2f} (stratified)")
    report_lines.append(f"Metric for CV: {args.metric}")
    report_lines.append(f"MI mode: {'knn-manual' if args.use_knn_manual else 'knn' if args.use_knn else f'bins={args.bins}'}")
    report_lines.append(f"SVM kernel={args.svm_kernel}, C={args.svm_C}, gamma={gamma_value}, class_weight={'balanced' if class_weight else 'None'}")
    report_lines.append("")
    report_lines.append("MI k search (train CV):")
    for k, sc in k_result.cv_scores:
        report_lines.append(f"  k={k}: mean {args.metric}={sc:.4f}")
    report_lines.append(f"Chosen k={k_result.best_k}")
    report_lines.append(f"Selected features (MI): {len(selected_names_best)}")
    report_lines.append(f"Test with MI-only set -> accuracy={test_acc:.4f}, macro-f1={test_f1:.4f}")
    report_lines.append("")
    report_lines.append("Forward wrapper on MI pool:")
    for step, fname, score in forward_result.history:
        report_lines.append(f"  Step {step}: +{fname} -> CV {args.metric}={score:.4f}")
    report_lines.append(f"Final forward set size: {len(forward_result.selected_names)}")
    report_lines.append(f"Test with forward set -> accuracy={test_acc_fw:.4f}, macro-f1={test_f1_fw:.4f}")

    report = "\n".join(report_lines)
    print(report)

    log_path = args.log_file if args.log_file else Path("mi_svm_forward_wrapper_log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(report + "\n\n---\n\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MI -> SVM -> forward wrapper pipeline")
    parser.add_argument("csv", type=Path, help="CSV file path")
    parser.add_argument("--sep", type=str, default=",", help="CSV separator (default ',')")
    parser.add_argument("--label-last", action="store_true", help="Label column is last (default first)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction (default 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for split")

    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "macro-f1"], help="CV metric")
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified folds for CV")
    parser.add_argument("--cv-topk-grid", type=str, help="Comma-separated k grid for MI search")
    parser.add_argument("--default-k-grid", nargs="*", type=int, default=[10, 50, 100, 200, 500], help="Fallback k grid if --cv-topk-grid is absent")

    parser.add_argument("--use-knn", action="store_true", help="Use sklearn k-NN MI (no bins)")
    parser.add_argument("--use-knn-manual", action="store_true", help="Use manual k-NN MI (no sklearn)")
    parser.add_argument("--n-neighbors", type=int, default=3, help="Neighbors for k-NN MI")
    parser.add_argument("--bins", type=int, default=10, help="Quantile bins for MI (when not k-NN)")

    parser.add_argument("--svm-kernel", type=str, default="rbf", choices=["linear", "rbf", "poly", "sigmoid"], help="SVM kernel")
    parser.add_argument("--svm-C", type=float, default=1.0, help="SVM C")
    parser.add_argument("--svm-gamma", type=str, default="scale", help="SVM gamma (scale/auto/float)")
    parser.add_argument("--class-weight-balanced", action="store_true", help="Use class_weight='balanced' for SVM")

    parser.add_argument("--min-improvement", type=float, default=0.0, help="Min CV gain to keep adding in forward wrapper")
    parser.add_argument("--log-file", type=Path, help="Append report to this file")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
