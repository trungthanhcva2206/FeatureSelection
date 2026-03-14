"""
SVM baseline (no feature selection).
- Split train/test (stratified).
- Optional CV over C/gamma/kernel to pick best combo.
- Train on full feature set, report test accuracy and macro-f1; always log to file.
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


@dataclass
class CVResult:
    best_params: Tuple[float, str | float, str]
    scores: List[Tuple[float, str | float, str, float]]  # (C, gamma, kernel, mean_metric)


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


def parse_float_grid(raw: str | None, default: Sequence[float]) -> List[float]:
    if not raw:
        return list(default)
    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values or list(default)


def parse_gamma_grid(raw: str | None, default: Sequence[str]) -> List[str | float]:
    if not raw:
        return list(default)
    values: List[str | float] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"scale", "auto"}:
            values.append(token)
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values or list(default)


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
                ("svm", SVC(C=C, gamma=gamma, kernel=kernel, class_weight=class_weight)),
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


def search_best_params(
    X: np.ndarray,
    y: np.ndarray,
    metric: str,
    cv_folds: int,
    C_grid: Iterable[float],
    gamma_grid: Iterable[str | float],
    kernel_grid: Iterable[str],
    class_weight: str | None,
) -> CVResult:
    scores: List[Tuple[float, str | float, str, float]] = []
    for C in C_grid:
        for gamma in gamma_grid:
            for kernel in kernel_grid:
                mean_score = evaluate_svm_cv(
                    X=X,
                    y=y,
                    metric=metric,
                    cv_folds=cv_folds,
                    C=C,
                    gamma=gamma,
                    kernel=kernel,
                    class_weight=class_weight,
                )
                scores.append((C, gamma, kernel, mean_score))
    scores.sort(key=lambda t: (-t[3], t[0]))
    best = scores[0] if scores else (1.0, "scale", "rbf", 0.0)
    return CVResult(best_params=(best[0], best[1], best[2]), scores=scores)


def run(args: argparse.Namespace) -> None:
    X, y, feature_names = load_dataset(args.csv, sep=args.sep, label_last=args.label_last)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    class_weight = "balanced" if args.class_weight_balanced else None

    C_grid = parse_float_grid(args.C_grid, args.default_C_grid)
    gamma_grid = parse_gamma_grid(args.gamma_grid, args.default_gamma_grid)
    kernel_grid = args.kernel_grid or args.default_kernel_grid

    cv_result = search_best_params(
        X=X_train,
        y=y_train,
        metric=args.metric,
        cv_folds=args.cv_folds,
        C_grid=C_grid,
        gamma_grid=gamma_grid,
        kernel_grid=kernel_grid,
        class_weight=class_weight,
    )

    best_C, best_gamma, best_kernel = cv_result.best_params

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=best_C, gamma=best_gamma, kernel=best_kernel, class_weight=class_weight)),
        ]
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, preds)
    test_f1 = f1_score(y_test, preds, average="macro")

    report_lines: List[str] = []
    report_lines.append(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append(f"Dataset: {args.csv.name}")
    report_lines.append(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
    report_lines.append(f"Train/Test split: {1 - args.test_size:.2f}/{args.test_size:.2f} (stratified)")
    report_lines.append(f"Metric (CV): {args.metric}")
    report_lines.append(
        f"Best params: C={best_C}, gamma={best_gamma}, kernel={best_kernel} | class_weight={'balanced' if class_weight else 'None'}"
    )
    report_lines.append("CV scores (sorted):")
    for C, gamma, kernel, sc in cv_result.scores[:20]:  # show top 20 for brevity
        report_lines.append(f"  C={C}, gamma={gamma}, kernel={kernel} -> {args.metric}={sc:.4f}")
    if len(cv_result.scores) > 20:
        report_lines.append(f"  ... ({len(cv_result.scores) - 20} more combos not shown)")
    report_lines.append(f"Test -> accuracy={test_acc:.4f}, macro-f1={test_f1:.4f}")

    report = "\n".join(report_lines)
    print(report)

    log_path = args.log_file if args.log_file else Path("svm_baseline_log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(report + "\n\n---\n\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SVM baseline without feature selection")
    parser.add_argument("csv", type=Path, help="CSV file path")
    parser.add_argument("--sep", type=str, default=",", help="CSV separator")
    parser.add_argument("--label-last", action="store_true", help="Label column is last")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "macro-f1"], help="CV metric")
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified folds for CV")

    parser.add_argument("--C-grid", dest="C_grid", type=str, help="Comma-separated C grid")
    parser.add_argument("--default-C-grid", dest="default_C_grid", nargs="*", type=float, default=[0.1, 1.0, 10.0], help="Fallback C grid")
    parser.add_argument("--gamma-grid", type=str, help="Comma-separated gamma grid (values or scale/auto)")
    parser.add_argument(
        "--default-gamma-grid",
        dest="default_gamma_grid",
        nargs="*",
        type=str,
        default=["scale", "auto"],
        help="Fallback gamma grid",
    )
    parser.add_argument(
        "--kernel-grid",
        nargs="*",
        type=str,
        choices=["linear", "rbf", "poly", "sigmoid"],
        help="Kernels to try; defaults to ['rbf']",
    )
    parser.add_argument(
        "--default-kernel-grid",
        dest="default_kernel_grid",
        nargs="*",
        type=str,
        default=["rbf"],
        help="Fallback kernels when --kernel-grid is not provided",
    )

    parser.add_argument("--class-weight-balanced", action="store_true", help="Use class_weight='balanced' for SVM")
    parser.add_argument("--log-file", type=Path, help="Append report to this file")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
