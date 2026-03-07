"""
Random Forest classification with optional hyperparameter grid search.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class RFSearchResult:
    best_params: dict
    best_score: float
    cv_scores: List[Tuple[dict, float]]


@dataclass
class SVMKSearchResult:
    best_k: int
    best_c: float
    best_gamma: str | float
    best_score: float
    cv_scores: List[Tuple[int, float, str | float, float]]


@dataclass
class SVMFullResult:
    best_c: float
    best_gamma: str | float
    best_score: float
    cv_scores: List[Tuple[float, str | float, float]]


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


def parse_grid(raw: str | None, default: Sequence[float]) -> List[float]:
    if not raw:
        return list(default)
    values: List[float] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            values.append(float(tok))
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


def search_rf(X, y, param_grid: dict, cv_folds: int, metric: str, class_weight: str | None) -> RFSearchResult:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid = list(ParameterGrid(param_grid)) or [{}]
    best_score = -np.inf
    best_params: dict | None = None
    cv_scores: List[Tuple[dict, float]] = []

    for params in grid:
        fold_scores = []
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            clf = RandomForestClassifier(
                n_jobs=-1,
                class_weight=class_weight,
                random_state=42,
                **params,
            )
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            fold_scores.append(evaluate_metric(y_val, preds, metric))
        mean_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
        cv_scores.append((params, mean_score))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    if best_params is None:
        best_params = {}
    return RFSearchResult(best_params=best_params, best_score=best_score, cv_scores=cv_scores)


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
) -> SVMKSearchResult:
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
                            (
                                "svm",
                                SVC(
                                    C=C,
                                    gamma=gamma,
                                    kernel=kernel,
                                    class_weight=class_weight,
                                ),
                            ),
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

    return SVMKSearchResult(
        best_k=int(best[0]),
        best_c=float(best[1]),
        best_gamma=best[2],
        best_score=float(best_score),
        cv_scores=cv_scores,
    )


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
                        (
                            "svm",
                            SVC(
                                C=C,
                                gamma=gamma,
                                kernel=kernel,
                                class_weight=class_weight,
                            ),
                        ),
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


def run(
    csv_path: Path,
    sep: str,
    label_last: bool,
    test_size: float,
    random_state: int,
    metric: str,
    cv_folds: int,
    n_estimators_grid: List[int],
    max_depth_grid: List[int | None],
    max_features_grid: List[str | float | int | None],
    min_samples_leaf_grid: List[int],
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

    param_grid = {
        "n_estimators": n_estimators_grid,
        "max_depth": max_depth_grid,
        "max_features": max_features_grid,
        "min_samples_leaf": min_samples_leaf_grid,
    }

    rf_result = search_rf(
        X_train,
        y_train,
        param_grid=param_grid,
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

    clf = RandomForestClassifier(
        n_jobs=-1,
        class_weight="balanced" if class_weight_balanced else None,
        random_state=random_state,
        **rf_result.best_params,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    k = max(1, min(top_k, len(order)))
    top_feats = [(feature_names[i], float(importances[i])) for i in order[:k]]

    svm_result = search_svm_topk(
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

    # Evaluate best SVM on full feature set
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

    # Evaluate best SVM on held-out test using selected top-k features
    cols = order[:svm_result.best_k]
    X_train_svm = X_train[:, cols]
    X_test_svm = X_test[:, cols]
    svm_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=svm_result.best_c,
                    gamma=svm_result.best_gamma if svm_kernel != "linear" else "scale",
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
        f"RF grid: {rf_result.best_params}",
        f"CV best score: {rf_result.best_score:.4f}",
        f"Top {k} features by importance (name, importance):",
        str(top_feats),
        f"Test {metric}: {test_metric:.4f} | accuracy: {test_accuracy:.4f} | macro-f1: {test_macro_f1:.4f}",
        f"SVM full features -> best C: {svm_full_result.best_c}, best gamma: {svm_full_result.best_gamma}, CV score: {svm_full_result.best_score:.4f}",
        f"SVM full test {metric}: {svm_full_test_metric:.4f} | accuracy: {svm_full_test_acc:.4f} | macro-f1: {svm_full_test_f1:.4f}",
        f"SVM on RF top-k -> k grid: {svm_k_grid}, best k: {svm_result.best_k}, best C: {svm_result.best_c}, best gamma: {svm_result.best_gamma}, CV score: {svm_result.best_score:.4f}",
        f"SVM test {metric}: {svm_test_metric:.4f} | accuracy: {svm_test_acc:.4f} | macro-f1: {svm_test_f1:.4f}",
    ]

    report = "\n".join(report_lines)
    print(report)

    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(report + "\n\n---\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Forest classifier with optional grid search")
    parser.add_argument("csv", type=Path, help="CSV file path")
    parser.add_argument("--sep", type=str, default=",", help="CSV separator (default ',')")
    parser.add_argument("--label-last", action="store_true", help="Set if the label/target is in the last column")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction (default 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "macro-f1"], help="Metric for CV and test evaluation")
    parser.add_argument("--n-estimators-grid", type=str, help="Comma-separated n_estimators grid", default="200,400,800")
    parser.add_argument("--max-depth-grid", type=str, help="Comma-separated max_depth grid (use 'None' for unlimited)", default="None,10,20")
    parser.add_argument("--max-features-grid", type=str, help="Comma-separated max_features grid (e.g., sqrt, log2, 0.5)", default="sqrt")
    parser.add_argument("--min-samples-leaf-grid", type=str, help="Comma-separated min_samples_leaf grid", default="1,2")
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified folds for CV searches (default 5)")
    parser.add_argument("--class-weight-balanced", action="store_true", help="Use class_weight='balanced'")
    parser.add_argument("--log-file", type=Path, help="Append report to this file")
    parser.add_argument("--top-k", type=int, default=20, help="Show top-k features by importance in the report")
    parser.add_argument("--svm-k-grid", type=str, default="5,10,20,50", help="Comma-separated k values (number of top RF features) for SVM CV")
    parser.add_argument("--svm-C-grid", type=str, default="0.1,1,10", help="Comma-separated C grid for SVM on selected features")
    parser.add_argument("--svm-kernel", type=str, default="linear", choices=["linear", "rbf"], help="SVM kernel for the feature-selection stage")
    parser.add_argument("--svm-gamma-grid", type=str, default="scale,auto", help="Comma-separated gamma grid (used if kernel != linear)")

    args = parser.parse_args()

    def parse_mixed(seq: str, default: Sequence[str | float | int | None]):
        if not seq:
            return list(default)
        out: List[str | float | int | None] = []
        for tok in seq.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.lower() == "none":
                out.append(None)
            elif tok.lower() in {"sqrt"}:
                out.append(tok.lower())
            else:
                try:
                    out.append(int(tok))
                except ValueError:
                    out.append(float(tok))
        return out or list(default)

    n_estimators_grid = [int(v) for v in parse_grid(args.n_estimators_grid, [200, 400, 800])]
    max_depth_grid = parse_mixed(args.max_depth_grid, [None, 10, 20])
    max_features_grid = parse_mixed(args.max_features_grid, ["sqrt"])
    min_samples_leaf_grid = [int(v) for v in parse_grid(args.min_samples_leaf_grid, [1, 2])]
    svm_k_grid = [int(v) for v in parse_grid(args.svm_k_grid, [5, 10, 20, 50])]
    svm_c_grid = [float(v) for v in parse_grid(args.svm_C_grid, [0.1, 1, 10])]
    svm_gamma_grid = parse_gamma_grid(args.svm_gamma_grid, ["scale", "auto"])

    run(
        csv_path=args.csv,
        sep=args.sep,
        label_last=args.label_last,
        test_size=args.test_size,
        random_state=args.random_state,
        metric=args.metric,
        cv_folds=args.cv_folds,
        n_estimators_grid=n_estimators_grid,
        max_depth_grid=max_depth_grid,
        max_features_grid=max_features_grid,
        min_samples_leaf_grid=min_samples_leaf_grid,
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
