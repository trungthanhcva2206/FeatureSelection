"""
Forward Selection wrapper method.
- Starts with empty feature set.
- Iteratively adds the feature that improves model performance most.
- Stops when no feature improves performance by at least min_improvement.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Support semicolon-delimited files with label in the last column (e.g., PredictStudent.csv)
    df = pd.read_csv(csv_path, sep=";")
    y = df.iloc[:, -1].to_numpy()
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    feature_names = list(df.columns[:-1])
    return X, y, feature_names


def evaluate_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: list[int],
    cv_folds: int,
    metric: str,
) -> float:
    """Evaluate model performance with given feature subset."""
    if len(feature_indices) == 0:
        return 0.0
    
    X_subset = X[:, feature_indices]
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    scores = []
    for train_idx, test_idx in skf.split(X_subset, y):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, multi_class="auto", solver="lbfgs")),
        ])
        
        clf.fit(X_train, y_train)
        if metric == "accuracy":
            score = clf.score(X_test, y_test)
        elif metric == "macro-f1":
            y_pred = clf.predict(X_test)
            score = f1_score(y_test, y_pred, average="macro")
        else:
            score = clf.score(X_test, y_test)
        scores.append(score)
    
    return float(np.mean(scores))


def forward_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    max_features: int | None,
    min_improvement: float,
    cv_folds: int,
    metric: str,
    verbose: bool = True,
) -> Tuple[List[int], List[str], List[Tuple[int, str, float]]]:
    """
    Perform forward selection.
    
    Returns:
        selected_indices: list of selected feature indices
        selected_names: list of selected feature names
        history: list of (step, feature_name, score) for each step
    """
    n_features = X.shape[1]
    max_features = max_features or n_features
    
    selected_indices = []
    remaining_indices = list(range(n_features))
    history = []
    
    current_score = 0.0
    step = 0
    
    while len(selected_indices) < max_features and remaining_indices:
        best_score = current_score
        best_feature = None
        
        # Try adding each remaining feature
        for idx in remaining_indices:
            candidate_indices = selected_indices + [idx]
            score = evaluate_features(X, y, candidate_indices, cv_folds, metric)
            
            if score > best_score:
                best_score = score
                best_feature = idx
        
        # Check if improvement is sufficient
        improvement = best_score - current_score
        if best_feature is None or improvement < min_improvement:
            if verbose:
                print(f"Stopping: no feature improves {metric} by at least {min_improvement:.4f}")
            break
        
        # Add best feature
        selected_indices.append(best_feature)
        remaining_indices.remove(best_feature)
        current_score = best_score
        step += 1
        
        feature_name = feature_names[best_feature]
        history.append((step, feature_name, current_score))
        
        if verbose:
            print(f"Step {step}: Added '{feature_name}' (idx={best_feature}), {metric}={current_score:.4f}")
    
    selected_names = [feature_names[i] for i in selected_indices]
    return selected_indices, selected_names, history


def run(
    csv_path: Path,
    max_features: int | None,
    min_improvement: float,
    cv_folds: int,
    metric: str,
    log_file: Path | None,
    verbose: bool,
) -> None:
    X, y, feature_names = load_dataset(csv_path)
    
    print(f"Starting forward selection on {csv_path.name}")
    print(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
    print(f"Max features: {max_features or 'unlimited'} | Min improvement: {min_improvement}")
    print(f"CV folds: {cv_folds} | Metric: {metric}")
    print("-" * 60)
    
    selected_indices, selected_names, history = forward_selection(
        X=X,
        y=y,
        feature_names=feature_names,
        max_features=max_features,
        min_improvement=min_improvement,
        cv_folds=cv_folds,
        metric=metric,
        verbose=verbose,
    )
    
    print("-" * 60)
    print(f"Selected {len(selected_names)} features:")
    print(", ".join(selected_names))
    
    if history:
        final_score = history[-1][2]
        print(f"\nFinal {metric}: {final_score:.4f}")
    
    # Write log
    if log_file:
        report_lines = []
        report_lines.append(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
        report_lines.append(f"Dataset: {csv_path.name}")
        report_lines.append(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
        report_lines.append(f"Method: Forward Selection")
        report_lines.append(f"Max features: {max_features or 'unlimited'}")
        report_lines.append(f"Min improvement: {min_improvement}")
        report_lines.append(f"CV folds: {cv_folds} | Metric: {metric}")
        report_lines.append(f"Selected features: {len(selected_names)} / {len(feature_names)}")
        report_lines.append("")
        report_lines.append("Selection history:")
        for step, fname, score in history:
            report_lines.append(f"  Step {step}: {fname} → {metric}={score:.4f}")
        report_lines.append("")
        report_lines.append("Final selected features:")
        report_lines.append(", ".join(selected_names))
        if history:
            report_lines.append(f"\nFinal {metric}: {history[-1][2]:.4f}")
        
        report = "\n".join(report_lines)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(report + "\n\n---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward Selection wrapper method")
    parser.add_argument("csv", type=Path, help="CSV file with label in first column")
    parser.add_argument(
        "--max-features",
        type=int,
        help="Maximum number of features to select (default: no limit)",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.01,
        help="Minimum improvement in metric to continue (default: 0.01)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "macro-f1"],
        help="Evaluation metric (default: accuracy)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Append run report to this file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step output",
    )
    
    args = parser.parse_args()
    
    run(
        csv_path=args.csv,
        max_features=args.max_features,
        min_improvement=args.min_improvement,
        cv_folds=args.cv_folds,
        metric=args.metric,
        log_file=args.log_file,
        verbose=not args.quiet,
    )
