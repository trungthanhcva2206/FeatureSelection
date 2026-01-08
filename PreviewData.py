import pandas as pd
from pathlib import Path

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
OUTPUT_FILE = BASE_DIR / "dataset_summary.md"

# =========================
# Helper function
# =========================
def analyze_dataset(csv_path):
    df = pd.read_csv(csv_path)

    # label = first column
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    summary = {
        "Dataset": csv_path.name,
        "Samples": X.shape[0],
        "Features": X.shape[1],
        "Classes": y.nunique(),
        "Label distribution": dict(y.value_counts()),
        "Missing values": X.isnull().sum().sum(),
        "Constant features": int((X.nunique() == 1).sum()),
        "High-dimensional": X.shape[1] > X.shape[0],
    }

    return summary

# =========================
# Main
# =========================
summaries = []

for csv_file in DATA_DIR.glob("*.csv"):
    summaries.append(analyze_dataset(csv_file))

# =========================
# Write Markdown report
# =========================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("# Dataset Summary Report\n\n")

    for s in summaries:
        f.write(f"## {s['Dataset']}\n\n")
        f.write(f"- Samples: {s['Samples']}\n")
        f.write(f"- Features: {s['Features']}\n")
        f.write(f"- Classes: {s['Classes']}\n")
        f.write(f"- High-dimensional: {s['High-dimensional']}\n")
        f.write(f"- Missing values: {s['Missing values']}\n")
        f.write(f"- Constant features: {s['Constant features']}\n")
        f.write(f"- Label distribution: {s['Label distribution']}\n\n")

print("Dataset summary saved to dataset_summary.md")
