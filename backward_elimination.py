"""
Backward Elimination wrapper method.
- Starts with all features.
- Iteratively removes the feature whose removal improves or least degrades performance.
- Stops when no feature can be removed without degrading performance beyond tolerance.
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


def backward_elimination(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    min_features: int,
    max_degradation: float,
    cv_folds: int,
    metric: str,
    verbose: bool = True,
) -> Tuple[List[int], List[str], List[Tuple[int, str, float]]]:
    """
    Perform backward elimination.
    
    Returns:
        selected_indices: list of selected feature indices
        selected_names: list of selected feature names
        history: list of (step, removed_feature_name, score_after_removal) for each step
    """
    n_features = X.shape[1]
    
    current_indices = list(range(n_features))
    history = []
    
    # Evaluate initial score with all features
    current_score = evaluate_features(X, y, current_indices, cv_folds, metric)
    
    if verbose:
        print(f"Initial {metric} with all {n_features} features: {current_score:.4f}")
    
    step = 0
    
    while len(current_indices) > min_features:
        best_score_after_removal = -np.inf
        best_feature_to_remove = None
        
        # Try removing each current feature
        for idx in current_indices:
            candidate_indices = [i for i in current_indices if i != idx]
            score = evaluate_features(X, y, candidate_indices, cv_folds, metric)
            
            # We prefer removing features that least degrade (or even improve) performance
            if score > best_score_after_removal:
                best_score_after_removal = score
                best_feature_to_remove = idx
        
        # Check if degradation is acceptable
        degradation = current_score - best_score_after_removal
        if degradation > max_degradation:
            if verbose:
                print(f"Stopping: removing any feature degrades {metric} by more than {max_degradation:.4f}")
            break
        
        # Remove the feature
        current_indices.remove(best_feature_to_remove)
        current_score = best_score_after_removal
        step += 1
        
        feature_name = feature_names[best_feature_to_remove]
        history.append((step, feature_name, current_score))
        
        if verbose:
            print(f"Step {step}: Removed '{feature_name}' (idx={best_feature_to_remove}), "
                  f"{metric}={current_score:.4f} (degradation={degradation:.4f})")
    
    selected_names = [feature_names[i] for i in current_indices]
    return current_indices, selected_names, history


def run(
    csv_path: Path,
    min_features: int,
    max_degradation: float,
    cv_folds: int,
    metric: str,
    log_file: Path | None,
    verbose: bool,
) -> None:
    X, y, feature_names = load_dataset(csv_path)
    
    print(f"Starting backward elimination on {csv_path.name}")
    print(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
    print(f"Min features: {min_features} | Max degradation: {max_degradation}")
    print(f"CV folds: {cv_folds} | Metric: {metric}")
    print("-" * 60)
    
    selected_indices, selected_names, history = backward_elimination(
        X=X,
        y=y,
        feature_names=feature_names,
        min_features=min_features,
        max_degradation=max_degradation,
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
        report_lines.append(f"Method: Backward Elimination")
        report_lines.append(f"Min features: {min_features}")
        report_lines.append(f"Max degradation: {max_degradation}")
        report_lines.append(f"CV folds: {cv_folds} | Metric: {metric}")
        report_lines.append(f"Selected features: {len(selected_names)} / {len(feature_names)}")
        report_lines.append("")
        report_lines.append("Elimination history:")
        for step, fname, score in history:
            report_lines.append(f"  Step {step}: Removed {fname} → {metric}={score:.4f}")
        report_lines.append("")
        report_lines.append("Final selected features:")
        report_lines.append(", ".join(selected_names))
        if history:
            report_lines.append(f"\nFinal {metric}: {history[-1][2]:.4f}")
        
        report = "\n".join(report_lines)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(report + "\n\n---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backward Elimination wrapper method")
    parser.add_argument("csv", type=Path, help="CSV file with label in first column")
    parser.add_argument(
        "--min-features",
        type=int,
        default=1,
        help="Minimum number of features to keep (default: 1)",
    )
    parser.add_argument(
        "--max-degradation",
        type=float,
        default=0.01,
        help="Maximum allowed degradation in metric when removing feature (default: 0.01)",
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
        min_features=args.min_features,
        max_degradation=args.max_degradation,
        cv_folds=args.cv_folds,
        metric=args.metric,
        log_file=args.log_file,
        verbose=not args.quiet,
    )
