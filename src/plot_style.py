"""Shared matplotlib style so all notebook figures look the same."""

import matplotlib.pyplot as plt

# Fixed colour per dataset / split, used in every notebook.
TRAIN_COLOR = "#2a78d6"  # EuroSAT (Level-1C)
TEST_COLOR = "#eb6834"   # Kaggle test set (Level-2A)
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
MUTED = "#898781"
INK = "#52514e"


def apply_style() -> None:
    """Thin marks, recessive grid and axes."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e8e7e3",
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.fontsize": 8,
    })
