"""
Run lasso_svm.py across all CSV files in Data/ and print outputs.
Adjust the PARAMS dict if you want different grids or options.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

# Base parameters matching your single-dataset command.
PARAMS = {
    "sep": ",",
    "metric": "accuracy",
    "lasso_c_grid": "0.1,0.5,1,2,5,10",
    "svm_c_grid": "0.5,1,2,5,10",
    "svm_gamma_grid": "scale,auto",
    "class_weight_balanced": True,
    "log_file": "lasso_svm_log.txt",
    "test_size": "0.2",
    "random_state": "42",
}

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
LASSO_SCRIPT = ROOT / "lasso_svm.py"
EXCLUDE_KEYWORDS = {"predictstudent"}


def build_command(csv_path: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(LASSO_SCRIPT),
        str(csv_path),
        "--sep",
        PARAMS["sep"],
        "--metric",
        PARAMS["metric"],
        "--lasso-c-grid",
        PARAMS["lasso_c_grid"],
        "--svm-c-grid",
        PARAMS["svm_c_grid"],
        "--svm-gamma-grid",
        PARAMS["svm_gamma_grid"],
        "--test-size",
        PARAMS["test_size"],
        "--random-state",
        PARAMS["random_state"],
    ]
    if PARAMS["class_weight_balanced"]:
        cmd.append("--class-weight-balanced")
    if PARAMS.get("log_file"):
        cmd.extend(["--log-file", PARAMS["log_file"]])
    return cmd


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")

    csv_files = [
        p
        for p in sorted(DATA_DIR.glob("*.csv"))
        if not any(key in p.name.lower() for key in EXCLUDE_KEYWORDS)
    ]
    if not csv_files:
        raise SystemExit(f"No CSV files found in {DATA_DIR}")

    print(f"Found {len(csv_files)} datasets (excluding {EXCLUDE_KEYWORDS}). Running lasso_svm.py for each...\n")

    for csv_path in csv_files:
        print(f"=== {csv_path.name} ===")
        cmd = build_command(csv_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip())
        print("\n")

if __name__ == "__main__":
    main()
