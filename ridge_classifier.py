"""
Ridge classification (L2) with optional hyperparameter grid search.
Reports accuracy and macro-f1 on a hold-out test split.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class RidgeSearchResult:
    best_alpha: float
    best_score: float
    cv_scores: List[Tuple[float, float]]


@dataclass
class SVMFullResult:
    best_c: float
    best_gamma: str | float
    best_score: float
    cv_scores: List[Tuple[float, str | float, float]]


@dataclass
class SVMKResult:
    best_k: int
    best_c: float
    best_gamma: str | float
    best_score: float
    cv_scores: List[Tuple[int, float, str | float, float]]


def load_dataset(csv_path: Path, sep: str, label_last: bool):
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
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            values.append(float(tok))
    return values or list(default)


def parse_int_grid(raw: str | None, default: Sequence[int]) -> List[int]:
    if not raw:
        return list(default)
    values: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            try:
                values.append(int(tok))
            except ValueError:
                continue
    return values or list(default)


def parse_gamma_grid(raw: str | None, default: Sequence[str | float]) -> List[str | float]:
    if not raw:
        return list(default)
    values: List[str | float] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        low = tok.lower()
        if low in {"scale", "auto"}:
            values.append(low)
        else:
            try:
                values.append(float(tok))
            except ValueError:
                continue
    return values or list(default)


def evaluate_metric(y_true, y_pred, metric: str) -> float:
    if metric == "macro-f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    return float(accuracy_score(y_true, y_pred))


def search_svm_full(
    X: np.ndarray,
    y: np.ndarray,
    c_grid: List[float],
    gamma_grid: List[str | float],
    kernel: str,
    cv_folds: int,
    metric: str,
    class_weight: str | None,
) -> SVMFullResult:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    best_score = -np.inf
    best = (c_grid[0], gamma_grid[0])
    cv_scores: List[Tuple[float, str | float, float]] = []

    for C in c_grid:
        gammas = ["scale"] if kernel == "linear" else gamma_grid
        for gamma in gammas:
            fold_scores = []
            for tr_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X[tr_idx], X[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]
                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("svm", SVC(C=C, gamma=gamma, kernel=kernel, class_weight=class_weight)),
                    ]
                )
                pipe.fit(X_tr, y_tr)
                preds = pipe.predict(X_val)
                fold_scores.append(evaluate_metric(y_val, preds, metric))
            mean_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
            cv_scores.append((float(C), gamma, mean_score))
            if mean_score > best_score:
                best_score = mean_score
                best = (float(C), gamma)

    return SVMFullResult(
        best_c=float(best[0]),
        best_gamma=best[1],
        best_score=float(best_score),
        cv_scores=cv_scores,
    )


def search_svm_topk(
    X: np.ndarray,
    y: np.ndarray,
    order: np.ndarray,
    k_grid: List[int],
    c_grid: List[float],
    gamma_grid: List[str | float],
    kernel: str,
    cv_folds: int,
    metric: str,
    class_weight: str | None,
) -> SVMKResult:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    best_score = -np.inf
    best = (k_grid[0], c_grid[0], gamma_grid[0])
    cv_scores: List[Tuple[int, float, str | float, float]] = []

    for k in k_grid:
        k_eff = max(1, min(int(k), X.shape[1]))
        cols = order[:k_eff]
        Xk = X[:, cols]
        for C in c_grid:
            gammas = ["scale"] if kernel == "linear" else gamma_grid
            for gamma in gammas:
                fold_scores = []
                for tr_idx, val_idx in skf.split(Xk, y):
                    X_tr, X_val = Xk[tr_idx], Xk[val_idx]
                    y_tr, y_val = y[tr_idx], y[val_idx]
                    pipe = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("svm", SVC(C=C, gamma=gamma, kernel=kernel, class_weight=class_weight)),
                        ]
                    )
                    pipe.fit(X_tr, y_tr)
                    preds = pipe.predict(X_val)
                    fold_scores.append(evaluate_metric(y_val, preds, metric))
                mean_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
                cv_scores.append((k_eff, float(C), gamma, mean_score))
                if mean_score > best_score:
                    best_score = mean_score
                    best = (k_eff, float(C), gamma)

    return SVMKResult(
        best_k=int(best[0]),
        best_c=float(best[1]),
        best_gamma=best[2],
        best_score=float(best_score),
        cv_scores=cv_scores,
    )


def search_ridge(X, y, alpha_grid: List[float], cv_folds: int, metric: str, class_weight: str | None) -> RidgeSearchResult:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    best_score = -np.inf
    best_alpha: float | None = None
    cv_scores: List[Tuple[float, float]] = []

    for alpha in alpha_grid:
        fold_scores = []
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", RidgeClassifier(alpha=alpha, class_weight=class_weight)),
                ]
            )
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_val)
            fold_scores.append(evaluate_metric(y_val, preds, metric))
        mean_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
        cv_scores.append((alpha, mean_score))
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha

    if best_alpha is None:
        best_alpha = alpha_grid[0]
    return RidgeSearchResult(best_alpha=best_alpha, best_score=best_score, cv_scores=cv_scores)


def run(
    csv_path: Path,
    sep: str,
    label_last: bool,
    test_size: float,
    random_state: int,
    metric: str,
    cv_folds: int,
    alpha_grid: List[float],
    class_weight_balanced: bool,
    log_file: Path | None,
    top_k: int,
    svm_k_grid: List[int],
    svm_c_grid: List[float],
    svm_gamma_grid: List[str | float],
    svm_kernel: str,
):
    X, y, feature_names = load_dataset(csv_path, sep=sep, label_last=label_last)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    ridge_result = search_ridge(
        X_train,
        y_train,
        alpha_grid=alpha_grid,
        cv_folds=cv_folds,
        metric=metric,
        class_weight="balanced" if class_weight_balanced else None,
    )

    svm_full_result = search_svm_full(
        X_train,
        y_train,
        c_grid=svm_c_grid,
        gamma_grid=svm_gamma_grid,
        kernel=svm_kernel,
        cv_folds=cv_folds,
        metric=metric,
        class_weight="balanced" if class_weight_balanced else None,
    )

    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeClassifier(alpha=ridge_result.best_alpha, class_weight="balanced" if class_weight_balanced else None)),
        ]
    )
    final_model.fit(X_train, y_train)
    preds = final_model.predict(X_test)

    ridge_coefs = final_model.named_steps["ridge"].coef_
    coef_mag = np.max(np.abs(ridge_coefs), axis=0) if ridge_coefs.ndim > 1 else np.abs(ridge_coefs)
    order = np.argsort(coef_mag)[::-1]
    k_disp = max(1, min(top_k, len(order)))
    top_feats = [(feature_names[i], float(coef_mag[i])) for i in order[:k_disp]]

    svm_topk_result = search_svm_topk(
        X_train,
        y_train,
        order=order,
        k_grid=svm_k_grid,
        c_grid=svm_c_grid,
        gamma_grid=svm_gamma_grid,
        kernel=svm_kernel,
        cv_folds=cv_folds,
        metric=metric,
        class_weight="balanced" if class_weight_balanced else None,
    )

    # Evaluate SVM full features on test
    svm_full_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=svm_full_result.best_c,
                    gamma=svm_full_result.best_gamma if svm_kernel != "linear" else "scale",
                    kernel=svm_kernel,
                    class_weight="balanced" if class_weight_balanced else None,
                ),
            ),
        ]
    )
    svm_full_pipe.fit(X_train, y_train)
    svm_full_preds = svm_full_pipe.predict(X_test)
    svm_full_test_metric = evaluate_metric(y_test, svm_full_preds, metric)
    svm_full_test_acc = float(accuracy_score(y_test, svm_full_preds))
    svm_full_test_f1 = float(f1_score(y_test, svm_full_preds, average="macro"))

    # Evaluate best SVM top-k on test
    cols = order[:svm_topk_result.best_k]
    X_train_svm = X_train[:, cols]
    X_test_svm = X_test[:, cols]
    svm_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=svm_topk_result.best_c,
                    gamma=svm_topk_result.best_gamma if svm_kernel != "linear" else "scale",
                    kernel=svm_kernel,
                    class_weight="balanced" if class_weight_balanced else None,
                ),
            ),
        ]
    )
    svm_pipe.fit(X_train_svm, y_train)
    svm_preds = svm_pipe.predict(X_test_svm)
    svm_test_metric = evaluate_metric(y_test, svm_preds, metric)
    svm_test_acc = float(accuracy_score(y_test, svm_preds))
    svm_test_f1 = float(f1_score(y_test, svm_preds, average="macro"))

    test_metric = evaluate_metric(y_test, preds, metric)
    test_accuracy = float(accuracy_score(y_test, preds))
    test_macro_f1 = float(f1_score(y_test, preds, average="macro"))

    report_lines = [
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Dataset: {csv_path.name}",
        f"Samples: {X.shape[0]} | Features: {X.shape[1]}",
        f"Train/Test split: {1 - test_size:.2f}/{test_size:.2f} (stratified)",
        f"Metric: {metric}",
        f"Class weight balanced: {class_weight_balanced}",
        f"Ridge alpha grid: {alpha_grid} | best alpha: {ridge_result.best_alpha}",
        f"CV best score: {ridge_result.best_score:.4f}",
        f"Top {k_disp} coefficients by |coef| (name, value):",
        str(top_feats),
        f"Test {metric}: {test_metric:.4f} | accuracy: {test_accuracy:.4f} | macro-f1: {test_macro_f1:.4f}",
        f"SVM full features -> best C: {svm_full_result.best_c}, best gamma: {svm_full_result.best_gamma}, CV score: {svm_full_result.best_score:.4f}",
        f"SVM full test {metric}: {svm_full_test_metric:.4f} | accuracy: {svm_full_test_acc:.4f} | macro-f1: {svm_full_test_f1:.4f}",
        f"SVM on Ridge top-k -> k grid: {svm_k_grid}, best k: {svm_topk_result.best_k}, best C: {svm_topk_result.best_c}, best gamma: {svm_topk_result.best_gamma}, CV score: {svm_topk_result.best_score:.4f}",
        f"SVM top-k test {metric}: {svm_test_metric:.4f} | accuracy: {svm_test_acc:.4f} | macro-f1: {svm_test_f1:.4f}",
    ]

    report = "\n".join(report_lines)
    print(report)

    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(report + "\n\n---\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ridge classifier with optional alpha grid")
    parser.add_argument("csv", type=Path, help="CSV file path")
    parser.add_argument("--sep", type=str, default=",", help="CSV separator (default ',')")
    parser.add_argument("--label-last", action="store_true", help="Set if the label/target is in the last column")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction (default 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "macro-f1"], help="Metric for CV and test evaluation")
    parser.add_argument("--alpha-grid", type=str, help="Comma-separated alpha grid", default="0.1,1,5,10")
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified folds for CV searches (default 5)")
    parser.add_argument("--class-weight-balanced", action="store_true", help="Use class_weight='balanced'")
    parser.add_argument("--log-file", type=Path, help="Append report to this file")
    parser.add_argument("--top-k", type=int, default=20, help="Show top-k coefficients in the report")
    parser.add_argument("--svm-k-grid", type=str, default="5,10,20,50", help="Comma-separated k values (number of top features) for SVM CV")
    parser.add_argument("--svm-C-grid", type=str, default="0.1,1,10", help="Comma-separated C grid for SVM")
    parser.add_argument("--svm-kernel", type=str, default="linear", choices=["linear", "rbf"], help="SVM kernel for feature selection stage")
    parser.add_argument("--svm-gamma-grid", type=str, default="scale,auto", help="Comma-separated gamma grid (used if kernel != linear)")

    args = parser.parse_args()

    alpha_grid = parse_float_grid(args.alpha_grid, [0.1, 1, 5, 10])
    svm_k_grid = parse_int_grid(args.svm_k_grid, [5, 10, 20, 50])
    svm_c_grid = parse_float_grid(args.svm_C_grid, [0.1, 1, 10])
    svm_gamma_grid = parse_gamma_grid(args.svm_gamma_grid, ["scale", "auto"])

    run(
        csv_path=args.csv,
        sep=args.sep,
        label_last=args.label_last,
        test_size=args.test_size,
        random_state=args.random_state,
        metric=args.metric,
        cv_folds=args.cv_folds,
        alpha_grid=alpha_grid,
        class_weight_balanced=args.class_weight_balanced,
        log_file=args.log_file,
        top_k=max(1, args.top_k),
        svm_k_grid=svm_k_grid,
        svm_c_grid=svm_c_grid,
        svm_gamma_grid=svm_gamma_grid,
        svm_kernel=args.svm_kernel,
    )


if __name__ == "__main__":
    main()
