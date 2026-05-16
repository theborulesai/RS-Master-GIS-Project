"""
visualization.py
----------------
Functions to generate the output maps for my project.
Uses matplotlib to plot the raster layers and suitability maps.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUT_MAPS, STUDY_AREA

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.dpi": 150,
})


def add_north_arrow(ax):
    """puts a north arrow on the map"""
    ax.annotate(
        "N", xy=(0.95, 0.95), xycoords="axes fraction",
        fontsize=12, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.8),
    )
    ax.annotate(
        "", xy=(0.95, 0.98), xycoords="axes fraction",
        xytext=(0.95, 0.92), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )


def plot_continuous_map(arr, title, cmap="viridis", unit="",
                        output_name=None, vmin=None, vmax=None):
    """plots a single raster layer as a coloured map"""
    extent = [STUDY_AREA["west"], STUDY_AREA["east"],
              STUDY_AREA["south"], STUDY_AREA["north"]]

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(arr, cmap=cmap, extent=extent, origin="upper",
                   vmin=vmin, vmax=vmax, interpolation="bilinear")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(unit)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    add_north_arrow(ax)

    plt.tight_layout()
    filepath = None
    if output_name:
        os.makedirs(OUTPUT_MAPS, exist_ok=True)
        filepath = os.path.join(OUTPUT_MAPS, output_name)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"    -> Saved map: {output_name}")
    plt.close(fig)
    return filepath


def plot_suitability_map(arr, title="Crop Suitability Map",
                         output_name="suitability_map.png"):
    """plots the final 1-5 classified suitability map with a legend"""
    extent = [STUDY_AREA["west"], STUDY_AREA["east"],
              STUDY_AREA["south"], STUDY_AREA["north"]]

    # red to green colour scheme
    colours = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]
    labels = [
        "1 - Not Suitable",
        "2 - Marginally Suitable",
        "3 - Moderately Suitable",
        "4 - Suitable",
        "5 - Highly Suitable",
    ]
    cmap = ListedColormap(colours)
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.imshow(arr, cmap=cmap, norm=norm, extent=extent,
              origin="upper", interpolation="nearest")

    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, label=l) for c, l in zip(colours, labels)]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=8,
              framealpha=0.9, title="Suitability")

    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    add_north_arrow(ax)

    plt.tight_layout()
    os.makedirs(OUTPUT_MAPS, exist_ok=True)
    filepath = os.path.join(OUTPUT_MAPS, output_name)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"    -> Saved map: {output_name}")
    plt.close(fig)
    return filepath


def plot_all_layers(layers_dict, output_prefix="layer"):
    """generates individual maps for each raster layer"""
    configs = {
        "ndvi": ("NDVI (Vegetation)", "RdYlGn", "NDVI"),
        "lst": ("Land Surface Temperature", "hot_r", "Temp (K)"),
        "dem": ("DEM Elevation", "terrain", "Elevation (m)"),
        "slope": ("Slope", "YlOrRd", "Degrees"),
        "rainfall": ("Annual Rainfall", "Blues", "mm"),
        "soil_ph": ("Soil pH", "RdYlBu_r", "pH"),
        "twi": ("Topographic Wetness Index", "GnBu", "TWI"),
        "tvdi": ("TVDI", "RdYlBu_r", "TVDI"),
        "soil_moisture": ("Soil Moisture Proxy", "YlGnBu", "Index"),
    }

    filepaths = {}
    for name, arr in layers_dict.items():
        if name in configs:
            title, cmap, unit = configs[name]
        else:
            title, cmap, unit = name.replace("_", " ").title(), "viridis", ""
        fname = f"{output_prefix}_{name}.png"
        filepaths[name] = plot_continuous_map(arr, title, cmap=cmap, unit=unit,
                                              output_name=fname)
    return filepaths


def plot_confusion_matrix(cm, classes, output_name="confusion_matrix.png"):
    """plots the confusion matrix as a heatmap"""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(classes))
    ax.set_xticks(ticks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(classes)

    # put the numbers in each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    os.makedirs(OUTPUT_MAPS, exist_ok=True)
    filepath = os.path.join(OUTPUT_MAPS, output_name)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"    -> Saved: {output_name}")
    plt.close(fig)
    return filepath


def plot_model_comparison(results, output_name="ml_comparison_chart.png"):
    """
    Creates a grouped bar chart comparing accuracy and kappa
    across all 3 ML models. Makes it easy to see which algo wins.
    """
    names = list(results.keys())
    accuracies = [results[n]["overall_accuracy"] * 100 for n in names]
    kappas = [results[n]["kappa"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # bar colours — one per algorithm
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    # accuracy chart
    bars1 = ax1.bar(names, accuracies, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Overall Accuracy (%)")
    ax1.set_title("Model Accuracy Comparison", fontweight="bold")
    ax1.set_ylim(0, 105)
    for bar, val in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

    # kappa chart
    bars2 = ax2.bar(names, kappas, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Cohen's Kappa (κ)")
    ax2.set_title("Model Kappa Comparison", fontweight="bold")
    ax2.set_ylim(0, max(kappas) * 1.4 + 0.05)
    for bar, val in zip(bars2, kappas):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    os.makedirs(OUTPUT_MAPS, exist_ok=True)
    filepath = os.path.join(OUTPUT_MAPS, output_name)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"    -> Saved: {output_name}")
    plt.close(fig)
    return filepath
