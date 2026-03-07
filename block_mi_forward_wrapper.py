"""
Block-wise MI -> union -> forward wrapper pipeline.
- Split train/test (stratified).
- Partition features into blocks of fixed size.
- For each block: compute MI on train, keep top-k in that block.
- Union all kept features, then run forward selection (SVM-based CV) on the union.
- Report test metrics for union set and forward-selected set; always log to file.
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

from mutual_information import mi_scores


@dataclass
class ForwardResult:
    selected_indices: List[int]
    selected_names: List[str]
    history: List[Tuple[int, str, float]]  # (step, feature, cv_score)


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


def parse_gamma(raw: str) -> str | float:
    lowered = raw.strip().lower()
    if lowered in {"scale", "auto"}:
        return lowered
    try:
        return float(lowered)
    except ValueError:
        return "scale"


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


def block_partition(n_features: int, block_size: int) -> List[Tuple[int, int]]:
    blocks: List[Tuple[int, int]] = []
    start = 0
    while start < n_features:
        end = min(start + block_size, n_features)
        blocks.append((start, end))
        start = end
    return blocks


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
    gamma_value = parse_gamma(args.svm_gamma)

    n_features = X_train.shape[1]
    blocks = block_partition(n_features, args.block_size)

    kept_indices: List[int] = []
    block_reports: List[str] = []
    block_k_grid = parse_int_grid(args.block_cv_topk_grid, default=args.block_cv_topk_default)

    for b_start, b_end in blocks:
        X_block = X_train[:, b_start:b_end]
        names_block = feature_names[b_start:b_end]
        scores_block = mi_scores(
            X_block,
            y_train,
            bins=args.bins,
            smoothing=1e-9,
            use_knn=args.use_knn,
            n_neighbors=args.n_neighbors,
            use_knn_manual=args.use_knn_manual,
        )

        k_grid_effective = [k for k in block_k_grid if k > 0]
        k_grid_effective = [min(k, scores_block.shape[0]) for k in k_grid_effective]
        k_grid_effective = sorted(set(k_grid_effective)) or [min(1, scores_block.shape[0])]

        cv_scores_block: List[Tuple[int, float]] = []
        for k in k_grid_effective:
            order_k = np.argsort(scores_block)[::-1][:k]
            X_block_k = X_block[:, order_k]
            score_k = evaluate_svm_cv(
                X=X_block_k,
                y=y_train,
                metric=args.metric,
                cv_folds=args.cv_folds,
                C=args.svm_C,
                gamma=gamma_value,
                kernel=args.svm_kernel,
                class_weight=class_weight,
            )
            cv_scores_block.append((k, score_k))

        cv_scores_block.sort(key=lambda t: (-t[1], t[0]))
        best_k_block = cv_scores_block[0][0] if cv_scores_block else 1

        order_best = np.argsort(scores_block)[::-1][:best_k_block]
        global_indices = [b_start + int(i) for i in order_best]
        kept_indices.extend(global_indices)
        top_preview = [(names_block[i], float(scores_block[i])) for i in order_best[:5]]
        block_reports.append(
            f"Block {b_start}:{b_end} -> best k={best_k_block} (CV {args.metric}={cv_scores_block[0][1]:.4f}) ; top5 {top_preview}"
        )

    kept_indices = sorted(set(kept_indices))
    if not kept_indices:
        kept_indices = list(range(min(args.top_k_per_block, n_features)))

    X_train_union = X_train[:, kept_indices]
    X_test_union = X_test[:, kept_indices]

    pipeline_union = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=args.svm_C, gamma=gamma_value, kernel=args.svm_kernel)),
        ]
    )
    pipeline_union.fit(X_train_union, y_train)
    preds_union = pipeline_union.predict(X_test_union)
    test_acc_union = accuracy_score(y_test, preds_union)
    test_f1_union = f1_score(y_test, preds_union, average="macro")

    forward_result = forward_wrapper(
        X=X_train,
        y=y_train,
        feature_names=feature_names,
        candidate_indices=kept_indices,
        metric=args.metric,
        cv_folds=args.cv_folds,
        C=args.svm_C,
        gamma=gamma_value,
        kernel=args.svm_kernel,
        class_weight=class_weight,
        min_improvement=args.min_improvement,
    )

    mask_forward = np.zeros(n_features, dtype=bool)
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
    report_lines.append(
        f"MI mode: {'knn-manual' if args.use_knn_manual else 'knn' if args.use_knn else f'bins={args.bins}'}"
    )
    report_lines.append(
        f"Blocks: size={args.block_size}, per-block top-k={args.top_k_per_block}, kept union={len(kept_indices)}"
    )
    report_lines.extend(block_reports)
    report_lines.append(
        f"Union test -> accuracy={test_acc_union:.4f}, macro-f1={test_f1_union:.4f}"
    )
    report_lines.append("Forward wrapper on union:")
    for step, fname, score in forward_result.history:
        report_lines.append(f"  Step {step}: +{fname} -> CV {args.metric}={score:.4f}")
    report_lines.append(
        f"Final forward size: {len(forward_result.selected_names)} | union candidates: {len(kept_indices)}"
    )
    report_lines.append(
        f"Test with forward set -> accuracy={test_acc_fw:.4f}, macro-f1={test_f1_fw:.4f}"
    )

    report = "\n".join(report_lines)
    print(report)

    log_path = args.log_file if args.log_file else Path("block_mi_forward_wrapper_log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(report + "\n\n---\n\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Block-wise MI -> union -> forward wrapper")
    parser.add_argument("csv", type=Path, help="CSV file path")
    parser.add_argument("--sep", type=str, default=",", help="CSV separator")
    parser.add_argument("--label-last", action="store_true", help="Label column is last")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    parser.add_argument("--block-size", type=int, default=10, help="Features per block")
    parser.add_argument("--top-k-per-block", type=int, default=20, help="(Legacy) cap top-k per block after CV; keep as upper bound")
    parser.add_argument(
        "--block-cv-topk-grid",
        type=str,
        help="Comma-separated k grid per block (evaluated on train CV before keeping)",
    )
    parser.add_argument(
        "--block-cv-topk-default",
        nargs="*",
        type=int,
        default=[1, 3, 5, 10],
        help="Fallback grid if --block-cv-topk-grid is absent",
    )

    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "macro-f1"], help="CV metric")
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified folds")
    parser.add_argument("--min-improvement", type=float, default=0.0, help="Min CV gain for forward step")

    parser.add_argument("--use-knn", action="store_true", help="Use sklearn k-NN MI")
    parser.add_argument("--use-knn-manual", action="store_true", help="Use manual k-NN MI")
    parser.add_argument("--n-neighbors", type=int, default=3, help="Neighbors for k-NN MI")
    parser.add_argument("--bins", type=int, default=10, help="Quantile bins for MI when not k-NN")

    parser.add_argument("--svm-kernel", type=str, default="rbf", choices=["linear", "rbf", "poly", "sigmoid"], help="SVM kernel")
    parser.add_argument("--svm-C", type=float, default=1.0, help="SVM C")
    parser.add_argument("--svm-gamma", type=str, default="scale", help="SVM gamma")
    parser.add_argument("--class-weight-balanced", action="store_true", help="Use class_weight='balanced' for SVM")

    parser.add_argument("--log-file", type=Path, help="Append report to this file")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
