"""Analysis script"""

from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd

project_path = Path(__file__).resolve().parent
data_path = project_path / "results"

data = pd.read_csv(data_path / "cvai_uncertainty_summary.csv")
data.info()

data_third_plane = data[data["level"] == 3]
print(data_third_plane)


plt.figure()

plt.plot(data_third_plane["model"], data_third_plane["cv"])

plt.show()
