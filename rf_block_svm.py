"""
Block-wise Random Forest feature scoring then SVM training.
- Shuffle features, split into blocks.
- For each block: cross-validate RF params, rank by importance, choose best k per block via CV.
- Union selected features, search SVM, evaluate on test.
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
class RFBlockResult:
    selected_indices: List[int]
    block_scores: List[Tuple[int, float, int]]  # (block_id, best_cv_score_for_k, best_k)


@dataclass
class SVMSearchResult:
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


def parse_int_grid(raw: str | None, default: Sequence[int]) -> List[int]:
    if not raw:
        return list(default)
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            try:
                out.append(int(tok))
            except ValueError:
                continue
    return out or list(default)


def parse_float_grid(raw: str | None, default: Sequence[float]) -> List[float]:
    if not raw:
        return list(default)
    out: List[float] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out or list(default)


def parse_mixed_grid(raw: str | None, default: Sequence[str | float | int | None]) -> List[str | float | int | None]:
    if not raw:
        return list(default)
    out: List[str | float | int | None] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        low = tok.lower()
        if low == "none":
            out.append(None)
        elif low in {"sqrt", "log2"}:
            out.append(low)
        else:
            try:
                out.append(int(tok))
            except ValueError:
                try:
                    out.append(float(tok))
                except ValueError:
                    continue
    return out or list(default)


def parse_gamma_grid(raw: str | None, default: Sequence[str | float]) -> List[str | float]:
    if not raw:
        return list(default)
    out: List[str | float] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        low = tok.lower()
        if low in {"scale", "auto"}:
            out.append(low)
        else:
            try:
                out.append(float(tok))
            except ValueError:
                continue
    return out or list(default)


def evaluate_metric(y_true, y_pred, metric: str) -> float:
    if metric == "macro-f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    return float(accuracy_score(y_true, y_pred))


def rf_block_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray,
    block_size: int,
    param_grid: dict,
    k_grid: List[int],
    cv_folds: int,
    metric: str,
    class_weight: str | None,
) -> RFBlockResult:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    selected: List[int] = []
    block_scores: List[Tuple[int, float, int]] = []

    n_blocks = int(np.ceil(len(feature_indices) / block_size))
    grid = list(ParameterGrid(param_grid)) or [{}]

    for b in range(n_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, len(feature_indices))
        block_idx = feature_indices[start:end]
        X_block = X[:, block_idx]

        best_score = -np.inf
        best_params: dict | None = None
        for params in grid:
            fold_scores = []
            for tr_idx, val_idx in skf.split(X_block, y):
                X_tr, X_val = X_block[tr_idx], X_block[val_idx]
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
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        # Fit full block with best params to rank features
        clf_full = RandomForestClassifier(
            n_jobs=-1,
            class_weight=class_weight,
            random_state=42,
            **(best_params or {}),
        )
        clf_full.fit(X_block, y)
        importances = clf_full.feature_importances_
        order = np.argsort(importances)[::-1]

        # Select best k from k_grid via CV using ordered features
        best_k = max(1, min(k_grid[0], len(order)))
        best_k_score = -np.inf
        for k_candidate in k_grid:
            k_eff = max(1, min(int(k_candidate), len(order)))
            cols = order[:k_eff]
            Xk = X_block[:, cols]
            fold_scores = []
            for tr_idx, val_idx in skf.split(Xk, y):
                X_tr, X_val = Xk[tr_idx], Xk[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]
                clf_k = RandomForestClassifier(
                    n_jobs=-1,
                    class_weight=class_weight,
                    random_state=42,
                    **(best_params or {}),
                )
                clf_k.fit(X_tr, y_tr)
                preds = clf_k.predict(X_val)
                fold_scores.append(evaluate_metric(y_val, preds, metric))
            mean_k = float(np.mean(fold_scores)) if fold_scores else -np.inf
            if mean_k > best_k_score:
                best_k_score = mean_k
                best_k = k_eff

        picked_local = block_idx[order[:best_k]]
        selected.extend(int(i) for i in picked_local)
        block_scores.append((b, best_k_score, best_k))

    return RFBlockResult(selected_indices=selected, block_scores=block_scores)


def search_svm(
    X: np.ndarray,
    y: np.ndarray,
    c_grid: List[float],
    gamma_grid: List[str | float],
    kernel: str,
    cv_folds: int,
    metric: str,
    class_weight: str | None,
) -> SVMSearchResult:
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

    return SVMSearchResult(
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
    block_size: int,
    n_estimators_grid: List[int],
    max_depth_grid: List[int | None],
    max_features_grid: List[str | float | int | None],
    min_samples_leaf_grid: List[int],
    k_grid: List[int],
    svm_c_grid: List[float],
    svm_gamma_grid: List[str | float],
    svm_kernel: str,
    class_weight_balanced: bool,
    log_file: Path | None,
):
    X, y, feature_names = load_dataset(csv_path, sep=sep, label_last=label_last)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(X_train.shape[1])

    param_grid = {
        "n_estimators": n_estimators_grid,
        "max_depth": max_depth_grid,
        "max_features": max_features_grid,
        "min_samples_leaf": min_samples_leaf_grid,
    }

    rf_block_result = rf_block_select(
        X_train,
        y_train,
        feature_indices=perm,
        block_size=block_size,
        param_grid=param_grid,
        k_grid=k_grid,
        cv_folds=cv_folds,
        metric=metric,
        class_weight="balanced" if class_weight_balanced else None,
    )

    union = sorted(set(rf_block_result.selected_indices))
    if not union:
        union = [int(perm[0])]

    X_train_sel = X_train[:, union]
    X_test_sel = X_test[:, union]

    svm_result = search_svm(
        X_train_sel,
        y_train,
        c_grid=svm_c_grid,
        gamma_grid=svm_gamma_grid,
        kernel=svm_kernel,
        cv_folds=cv_folds,
        metric=metric,
        class_weight="balanced" if class_weight_balanced else None,
    )

    final_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=svm_result.best_c, gamma=svm_result.best_gamma if svm_kernel != "linear" else "scale", kernel=svm_kernel)),
        ]
    )
    final_pipe.fit(X_train_sel, y_train)
    preds = final_pipe.predict(X_test_sel)

    test_metric = evaluate_metric(y_test, preds, metric)
    test_acc = float(accuracy_score(y_test, preds))
    test_f1 = float(f1_score(y_test, preds, average="macro"))

    k_disp = min(len(union), 20)
    top_preview = [(feature_names[i], float(i)) for i in union[:k_disp]]

    report_lines = [
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Dataset: {csv_path.name}",
        f"Samples: {X.shape[0]} | Features: {X.shape[1]}",
        f"Train/Test split: {1 - test_size:.2f}/{test_size:.2f} (stratified)",
        f"Metric: {metric}",
        f"Class weight balanced: {class_weight_balanced}",
        f"Block size: {block_size} | blocks: {int(np.ceil(X_train.shape[1]/block_size))}",
        f"RF grid: {param_grid} | k grid per block: {k_grid}",
        f"Selected union size: {len(union)}",
        f"SVM grid C: {svm_c_grid}, gamma: {svm_gamma_grid}, kernel: {svm_kernel} | best: C={svm_result.best_c}, gamma={svm_result.best_gamma}",
        f"Test {metric}: {test_metric:.4f} | accuracy: {test_acc:.4f} | macro-f1: {test_f1:.4f}",
        f"Selected feature indices (first {k_disp}):",
        str(top_preview),
    ]

    report = "\n".join(report_lines)
    print(report)

    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(report + "\n\n---\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Block-wise RF feature scoring + SVM")
    parser.add_argument("csv", type=Path, help="CSV file path")
    parser.add_argument("--sep", type=str, default=",", help="CSV separator (default ',')")
    parser.add_argument("--label-last", action="store_true", help="Set if the label/target is in the last column")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction (default 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splits and shuffling")
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "macro-f1"], help="Metric for CV and test evaluation")
    parser.add_argument("--block-size", type=int, default=10, help="Number of features per block")
    parser.add_argument("--n-estimators-grid", type=str, default="200,400", help="Comma-separated n_estimators grid for RF")
    parser.add_argument("--max-depth-grid", type=str, default="None,10,20", help="Comma-separated max_depth grid (use None for unlimited)")
    parser.add_argument("--max-features-grid", type=str, default="sqrt", help="Comma-separated max_features grid")
    parser.add_argument("--min-samples-leaf-grid", type=str, default="1,2", help="Comma-separated min_samples_leaf grid")
    parser.add_argument("--k-grid", type=str, default="1,2,3,5", help="Comma-separated k values to try per block")
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified folds for CV searches (default 5)")
    parser.add_argument("--svm-C-grid", type=str, default="0.1,1,10", help="Comma-separated C grid for SVM")
    parser.add_argument("--svm-gamma-grid", type=str, default="scale,auto", help="Comma-separated gamma grid for SVM (ignored if kernel=linear)")
    parser.add_argument("--svm-kernel", type=str, default="linear", choices=["linear", "rbf"], help="SVM kernel for final model")
    parser.add_argument("--class-weight-balanced", action="store_true", help="Use class_weight='balanced' for RF and SVM")
    parser.add_argument("--log-file", type=Path, help="Append report to this file")

    args = parser.parse_args()

    n_estimators_grid = parse_int_grid(args.n_estimators_grid, [200, 400])
    max_depth_grid = parse_mixed_grid(args.max_depth_grid, [None, 10, 20])
    max_features_grid = parse_mixed_grid(args.max_features_grid, ["sqrt"])
    min_samples_leaf_grid = parse_int_grid(args.min_samples_leaf_grid, [1, 2])
    k_grid = parse_int_grid(args.k_grid, [1, 2, 3, 5])
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
        block_size=args.block_size,
        n_estimators_grid=n_estimators_grid,
        max_depth_grid=max_depth_grid,
        max_features_grid=max_features_grid,
        min_samples_leaf_grid=min_samples_leaf_grid,
        k_grid=k_grid,
        svm_c_grid=svm_c_grid,
        svm_gamma_grid=svm_gamma_grid,
        svm_kernel=args.svm_kernel,
        class_weight_balanced=args.class_weight_balanced,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
