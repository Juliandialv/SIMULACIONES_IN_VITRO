"""Test script"""

from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent

data = pd.read_csv(project_dir / "results" / "cvai_uncertainty_samples.csv")

data_3l = data[data["level"] == 3]

output_path = project_dir / "results" / "cvai_uncertainty_samples_level3.csv"
data_3l.to_csv(output_path, index=False)
