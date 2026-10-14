import pandas as pd
from pathlib import Path
from full_analysis import plot_reclassification_by_level, apply_paper_style

apply_paper_style()  # por si la importas suelta, sin pasar por main()

summary = pd.read_csv("results/cvai_uncertainty_summary.csv")
samples = pd.read_csv("results/cvai_uncertainty_samples.csv")

plot_reclassification_by_level(summary, samples, Path("figures"), reference_level=3)
