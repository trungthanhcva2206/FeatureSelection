"""
Symmetrical Uncertainty (SU) -> SVM -> forward extension pipeline.
- Split train/test (stratified).
- Compute SU scores on train via quantile binning.
- Search top-k by SU on train via CV (SVM scorer).
- Base set = chosen SU top-k.
- Forward step tries to add features outside the base set if they improve CV.
- Report test metrics for base set and extended set; always log to file.
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

from mutual_information_su import load_dataset, select_features, symmetrical_uncertainty_scores


@dataclass
class KSearchResult:
    best_k: int
    cv_scores: List[Tuple[int, float]]  # (k, mean metric)


@dataclass
class ForwardResult:
    selected_indices: List[int]
    selected_names: List[str]
    history: List[Tuple[int, str, float]]  # (step, feature, cv_score)


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
            continue
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


def forward_extend(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    base_indices: List[int],
    metric: str,
    cv_folds: int,
    C: float,
    gamma: str | float,
    kernel: str,
    class_weight: str | None,
    min_improvement: float,
) -> ForwardResult:
    selected = list(base_indices)
    remaining = [i for i in range(X.shape[1]) if i not in selected]
    history: List[Tuple[int, str, float]] = []

    if selected:
        current_score = evaluate_svm_cv(
            X=X[:, selected],
            y=y,
            metric=metric,
            cv_folds=cv_folds,
            C=C,
            gamma=gamma,
            kernel=kernel,
            class_weight=class_weight,
        )
    else:
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

    selected_names = [feature_names[i] for i in selected]
    return ForwardResult(selected_indices=selected, selected_names=selected_names, history=history)


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


def run(args: argparse.Namespace) -> None:
    X, y, feature_names = load_dataset(args.csv, sep=args.sep, label_last=args.label_last)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    su_scores = symmetrical_uncertainty_scores(
        X_train,
        y_train,
        bins=args.bins,
        smoothing=1e-9,
    )

    k_grid = parse_int_grid(args.cv_topk_grid, default=args.default_k_grid)
    class_weight = "balanced" if args.class_weight_balanced else None
    gamma_value = parse_gamma(args.svm_gamma)

    k_result = search_best_k(
        X=X_train,
        y=y_train,
        scores=su_scores,
        feature_names=feature_names,
        k_grid=k_grid,
        metric=args.metric,
        cv_folds=args.cv_folds,
        C=args.svm_C,
        gamma=gamma_value,
        kernel=args.svm_kernel,
        class_weight=class_weight,
    )

    mask_base, base_names = select_features(
        scores=su_scores,
        feature_names=feature_names,
        top_k=k_result.best_k,
        threshold=0.0,
    )
    base_indices = list(np.where(mask_base)[0])

    X_train_base = X_train[:, mask_base]
    X_test_base = X_test[:, mask_base]

    pipeline_base = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=args.svm_C, gamma=gamma_value, kernel=args.svm_kernel)),
        ]
    )
    pipeline_base.fit(X_train_base, y_train)
    preds_base = pipeline_base.predict(X_test_base)
    test_acc_base = accuracy_score(y_test, preds_base)
    test_f1_base = f1_score(y_test, preds_base, average="macro")

    forward_result = forward_extend(
        X=X_train,
        y=y_train,
        feature_names=feature_names,
        base_indices=base_indices,
        metric=args.metric,
        cv_folds=args.cv_folds,
        C=args.svm_C,
        gamma=gamma_value,
        kernel=args.svm_kernel,
        class_weight=class_weight,
        min_improvement=args.min_improvement,
    )

    mask_forward = np.zeros(X_train.shape[1], dtype=bool)
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

    report_lines: List[str] = []
    report_lines.append(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append(f"Dataset: {args.csv.name}")
    report_lines.append(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
    report_lines.append(f"Train/Test split: {1 - args.test_size:.2f}/{args.test_size:.2f} (stratified)")
    report_lines.append(f"Metric (CV): {args.metric}")
    report_lines.append(f"SU mode: bins={args.bins}")
    report_lines.append(
        f"Chosen k={k_result.best_k} from grid {list(k_grid)} | base features: {len(base_names)}"
    )
    report_lines.append("SU k search (train CV):")
    for k, sc in k_result.cv_scores:
        report_lines.append(f"  k={k}: mean {args.metric}={sc:.4f}")
    report_lines.append(
        f"Base set test -> accuracy={test_acc_base:.4f}, macro-f1={test_f1_base:.4f}"
    )
    report_lines.append("Forward extension (adds outside base):")
    for step, fname, score in forward_result.history:
        report_lines.append(f"  Step {step}: +{fname} -> CV {args.metric}={score:.4f}")
    report_lines.append(
        f"Final forward size: {len(forward_result.selected_names)} | base size: {len(base_names)}"
    )
    report_lines.append(
        f"Test with forward set -> accuracy={test_acc_fw:.4f}, macro-f1={test_f1_fw:.4f}"
    )

    report = "\n".join(report_lines)
    print(report)

    log_path = args.log_file if args.log_file else Path("su_svm_forward_extend_log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(report + "\n\n---\n\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Symmetrical Uncertainty -> SVM -> forward extension pipeline")
    parser.add_argument("csv", type=Path, help="CSV file path")
    parser.add_argument("--sep", type=str, default=",", help="CSV separator")
    parser.add_argument("--label-last", action="store_true", help="Label column is last")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    parser.add_argument("--bins", type=int, default=10, help="Quantile bins for SU discretization")
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "macro-f1"], help="CV metric")
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified folds for CV")
    parser.add_argument("--cv-topk-grid", type=str, help="Comma-separated k grid for SU search")
    parser.add_argument("--default-k-grid", nargs="*", type=int, default=[10, 50, 100, 200, 500], help="Fallback k grid")

    parser.add_argument("--svm-kernel", type=str, default="rbf", choices=["linear", "rbf", "poly", "sigmoid"], help="SVM kernel")
    parser.add_argument("--svm-C", type=float, default=1.0, help="SVM C")
    parser.add_argument("--svm-gamma", type=str, default="scale", help="SVM gamma")
    parser.add_argument("--class-weight-balanced", action="store_true", help="Use class_weight='balanced' for SVM")

    parser.add_argument("--min-improvement", type=float, default=0.0, help="Min CV gain to add a feature in forward stage")
    parser.add_argument("--log-file", type=Path, help="Append report to this file")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
