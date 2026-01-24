"""
Block-wise L1 (lasso) selection: split features into blocks of fixed size, run lasso on each block,
union all selected features, then train/evaluate an SVM on the union.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class BlockResult:
    block_index: int
    selected_indices: List[int]
    best_c: float
    cv_scores: List[Tuple[float, float]]


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
        if token:
            values.append(float(token))
    if not values:
        values = list(default)
    return values


def parse_gamma_grid(raw: str | None, default: Sequence[str | float]) -> List[str | float]:
    if not raw:
        return list(default)
    values: List[str | float] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"scale", "auto"}:
            values.append(token)
        else:
            values.append(float(token))
    if not values:
        values = list(default)
    return values


def evaluate_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "macro-f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    return float(accuracy_score(y_true, y_pred))


def l1_on_block(
    X_block: np.ndarray,
    y: np.ndarray,
    block_offset: int,
    feature_names: List[str],
    c_grid: Iterable[float],
    max_iter: int,
    metric: str,
    coef_threshold: float,
    min_features: int,
    cv_folds: int,
    class_weight: str | None,
    random_state: int,
) -> BlockResult:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    best_score = -np.inf
    best_c: float | None = None
    cv_scores: List[Tuple[float, float]] = []

    c_grid_list = list(c_grid)
    for C in c_grid_list:
        fold_scores = []
        for tr_idx, val_idx in skf.split(X_block, y):
            X_tr, X_val = X_block[tr_idx], X_block[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            n_classes = len(np.unique(y_tr))
            solver = "saga" if n_classes > 2 else "liblinear"
            clf_kwargs = {
                "penalty": "l1",
                "C": C,
                "solver": solver,
                "max_iter": max_iter,
                "n_jobs": None,
                "class_weight": class_weight,
                "random_state": random_state,
            }
            if solver == "saga" and n_classes > 2:
                clf_kwargs["multi_class"] = "multinomial"

            clf = LogisticRegression(**clf_kwargs)
            clf.fit(X_tr_s, y_tr)
            preds = clf.predict(X_val_s)
            fold_scores.append(evaluate_metric(y_val, preds, metric))

        mean_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
        cv_scores.append((float(C), mean_score))
        if mean_score > best_score:
            best_score = mean_score
            best_c = float(C)

    if best_c is None:
        best_c = float(c_grid_list[0])

    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X_block)
    n_classes = len(np.unique(y))
    solver = "saga" if n_classes > 2 else "liblinear"
    final_kwargs = {
        "penalty": "l1",
        "C": best_c,
        "solver": solver,
        "max_iter": max_iter,
        "n_jobs": None,
        "class_weight": class_weight,
        "random_state": random_state,
    }
    if solver == "saga" and n_classes > 2:
        final_kwargs["multi_class"] = "multinomial"

    final_clf = LogisticRegression(**final_kwargs)
    final_clf.fit(X_scaled, y)

    coefs = np.abs(final_clf.coef_)
    coef_magnitudes = np.max(coefs, axis=0) if coefs.ndim > 1 else coefs
    mask = coef_magnitudes > coef_threshold

    if mask.sum() == 0:
        order = np.argsort(coef_magnitudes)[::-1]
        top_k = max(1, min(min_features, coef_magnitudes.shape[0]))
        mask = np.zeros_like(coef_magnitudes, dtype=bool)
        mask[order[:top_k]] = True

    selected_local = np.nonzero(mask)[0].tolist()
    selected_global = [block_offset + idx for idx in selected_local]

    return BlockResult(
        block_index=block_offset,
        selected_indices=selected_global,
        best_c=best_c,
        cv_scores=cv_scores,
    )


def run(
    csv_path: Path,
    sep: str,
    label_last: bool,
    test_size: float,
    random_state: int,
    block_size: int,
    lasso_c_grid: List[float],
    lasso_max_iter: int,
    svm_kernel: str,
    svm_c_grid: List[float],
    svm_gamma_grid: List[str | float],
    coef_threshold: float,
    min_features: int,
    metric: str,
    cv_folds: int,
    log_file: Path | None,
    use_class_weight: bool,
) -> None:
    np.random.seed(random_state)
    X, y, feature_names = load_dataset(csv_path, sep=sep, label_last=label_last)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    n_features = X_train.shape[1]
    union_mask = np.zeros(n_features, dtype=bool)
    block_results: List[BlockResult] = []

    for start in range(0, n_features, block_size):
        end = min(start + block_size, n_features)
        block_slice = slice(start, end)
        X_block = X_train[:, block_slice]
        block_res = l1_on_block(
            X_block=X_block,
            y=y_train,
            block_offset=start,
            feature_names=feature_names,
            c_grid=lasso_c_grid,
            max_iter=lasso_max_iter,
            metric=metric,
            coef_threshold=coef_threshold,
            min_features=min_features,
            cv_folds=cv_folds,
            class_weight="balanced" if use_class_weight else None,
            random_state=random_state,
        )
        union_mask[block_res.selected_indices] = True
        block_results.append(block_res)

    selected_indices = np.nonzero(union_mask)[0].tolist()
    selected_names = [feature_names[i] for i in selected_indices]

    if not selected_indices:
        selected_indices = list(range(min_features))
        union_mask[:min_features] = True
        selected_names = [feature_names[i] for i in selected_indices]

    X_train_sel = X_train[:, union_mask]
    X_test_sel = X_test[:, union_mask]

    gamma_grid = svm_gamma_grid if svm_kernel != "linear" else ["scale"]
    svm_result_best_score = -np.inf
    svm_best_params: Tuple[float, str | float] | None = None

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    for C in svm_c_grid:
        for gamma in gamma_grid:
            fold_scores = []
            for tr_idx, val_idx in skf.split(X_train_sel, y_train):
                X_tr, X_val = X_train_sel[tr_idx], X_train_sel[val_idx]
                y_tr, y_val = y_train[tr_idx], y_train[val_idx]
                pipeline = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("svm", SVC(C=C, gamma=gamma, kernel=svm_kernel, class_weight="balanced" if use_class_weight else None)),
                    ]
                )
                pipeline.fit(X_tr, y_tr)
                preds = pipeline.predict(X_val)
                fold_scores.append(evaluate_metric(y_val, preds, metric))
            mean_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
            if mean_score > svm_result_best_score:
                svm_result_best_score = mean_score
                svm_best_params = (float(C), gamma)

    if svm_best_params is None:
        svm_best_params = (float(svm_c_grid[0]), gamma_grid[0])

    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=svm_best_params[0],
                    gamma=svm_best_params[1],
                    kernel=svm_kernel,
                    class_weight="balanced" if use_class_weight else None,
                ),
            ),
        ]
    )
    final_model.fit(X_train_sel, y_train)
    test_preds = final_model.predict(X_test_sel)

    test_metric = evaluate_metric(y_test, test_preds, metric)
    test_accuracy = float(accuracy_score(y_test, test_preds))
    test_macro_f1 = float(f1_score(y_test, test_preds, average="macro"))

    report_lines: List[str] = []
    report_lines.append(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append(f"Dataset: {csv_path.name}")
    report_lines.append(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
    report_lines.append(f"Train/Test split: {1 - test_size:.2f}/{test_size:.2f} (stratified)")
    report_lines.append(f"Metric: {metric}")
    report_lines.append(f"Class weight balanced: {use_class_weight}")
    report_lines.append(f"Block size: {block_size} | blocks: {len(block_results)}")
    report_lines.append(
        f"Lasso C grid: {lasso_c_grid} | union selected: {len(selected_names)} / {len(feature_names)}"
    )
    report_lines.append(
        f"SVM grid (C x gamma): {len(svm_c_grid)} x {len(gamma_grid)} | best: C={svm_best_params[0]}, gamma={svm_best_params[1]}"
    )
    report_lines.append(
        f"Test {metric}: {test_metric:.4f} | accuracy: {test_accuracy:.4f} | macro-f1: {test_macro_f1:.4f}"
    )

    report_lines.append("Selected features (names):")
    report_lines.append(str(selected_names))

    report = "\n".join(report_lines)
    print(report)

    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(report + "\n\n---\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Block-wise L1 + SVM")
    parser.add_argument("csv", type=Path, help="CSV file path")
    parser.add_argument("--sep", type=str, default=",", help="CSV separator (default ',')")
    parser.add_argument("--label-last", action="store_true", help="Set if the label/target is in the last column")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction (default 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--block-size", type=int, default=10, help="Number of features per block for L1 selection")
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "macro-f1"], help="Metric for CV and test evaluation")
    parser.add_argument("--lasso-C", type=float, default=1.0, help="Inverse regularization for L1 logistic (used if no grid)")
    parser.add_argument("--lasso-c-grid", type=str, help="Comma-separated C grid for L1 logistic (overrides --lasso-C)")
    parser.add_argument("--lasso-max-iter", type=int, default=2000, help="Max iterations for L1 logistic (default 2000)")
    parser.add_argument("--coef-threshold", type=float, default=1e-4, help="Minimum |coef| to keep a feature inside a block")
    parser.add_argument("--min-features", type=int, default=1, help="Fallback minimum per-block and overall")
    parser.add_argument("--svm-kernel", type=str, default="rbf", choices=["linear", "rbf", "poly", "sigmoid"], help="SVM kernel (default rbf)")
    parser.add_argument("--svm-C", type=float, default=1.0, help="SVM C (used if no grid)")
    parser.add_argument("--svm-c-grid", type=str, help="Comma-separated C grid for SVM (overrides --svm-C)")
    parser.add_argument("--svm-gamma", type=str, default="scale", help="SVM gamma (used if no grid; 'scale', 'auto' or float)")
    parser.add_argument("--svm-gamma-grid", type=str, help="Comma-separated gamma grid for SVM (numbers or scale/auto)")
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified folds for CV searches (default 5)")
    parser.add_argument("--class-weight-balanced", action="store_true", help="Use class_weight='balanced' for both L1 selector and SVM")
    parser.add_argument("--log-file", type=Path, help="Append report to this file")

    args = parser.parse_args()

    lasso_c_grid = parse_float_grid(args.lasso_c_grid, default=[args.lasso_C])
    svm_c_grid = parse_float_grid(args.svm_c_grid, default=[args.svm_C])
    svm_gamma_grid = parse_gamma_grid(args.svm_gamma_grid, default=[args.svm_gamma])

    run(
        csv_path=args.csv,
        sep=args.sep,
        label_last=args.label_last,
        test_size=args.test_size,
        random_state=args.random_state,
        block_size=args.block_size,
        lasso_c_grid=lasso_c_grid,
        lasso_max_iter=args.lasso_max_iter,
        svm_kernel=args.svm_kernel,
        svm_c_grid=svm_c_grid,
        svm_gamma_grid=svm_gamma_grid,
        coef_threshold=args.coef_threshold,
        min_features=args.min_features,
        metric=args.metric,
        cv_folds=args.cv_folds,
        log_file=args.log_file,
        use_class_weight=args.class_weight_balanced,
    )


if __name__ == "__main__":
    main()
