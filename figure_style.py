# figure_style.py
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def apply_paper_style():
    plt.rcParams.update({
        # --- Tipografía ---
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",

        # --- Tamaños base (paper-friendly) ---
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        # --- Líneas ---
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,

        # --- Grid suave ---
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": "--"
    })


def format_axis_scientific(ax, both=True):
    """Notación científica consistente estilo paper"""
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    if both:
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    else:
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # estilo del ×10^n
    ax.xaxis.get_offset_text().set_fontsize(10)
    ax.yaxis.get_offset_text().set_fontsize(10)
    