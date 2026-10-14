"""Test script: classification of obtained plagiocephaly"""

from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent

THRESHOLDS = [3.4, 6.25, 8.75, 11.0]
CATEGORY_LABELS = [
    "Normal / Mínima",
    "Leve",
    "Moderada",
    "Mod. Severa",
    "Severa",
]

data_nominal = pd.read_csv(project_dir / "results" / "cvai_uncertainty_summary.csv")
data_nominal = data_nominal[data_nominal["level"] == 3]

data_nominal["category"] = pd.cut(
    data_nominal["nominal_cvai"],
    bins=[-float("inf")] + THRESHOLDS + [float("inf")],
    labels=CATEGORY_LABELS,
    right=False,
)

counts_nominal = data_nominal["category"].value_counts().reindex(CATEGORY_LABELS, fill_value=0)

summary_nominal = pd.DataFrame({
    "n": counts_nominal,
    "percentage": counts_nominal / len(data_nominal) * 100,
})


data_uncertainty = pd.read_csv(project_dir / "results" / "cvai_uncertainty_samples_level3.csv")

data_uncertainty["category"] = pd.cut(
    data_uncertainty["cvai"],
    bins=[-float("inf")] + THRESHOLDS + [float("inf")],
    labels=CATEGORY_LABELS,
    right=False,
)

counts_uncertainty = data_uncertainty["category"].value_counts().reindex(CATEGORY_LABELS, fill_value=0)

summary_uncertainty = pd.DataFrame({
    "n": counts_uncertainty,
    "percentage": counts_uncertainty / len(data_uncertainty) * 100,
})

print("Datos nominales: ")
print(summary_nominal)
print()
print("Datos modelo incertidumbre: ")
print(summary_uncertainty)
