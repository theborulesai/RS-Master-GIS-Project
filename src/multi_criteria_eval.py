"""
multi_criteria_eval.py
----------------------
Does the weighted overlay for crop suitability.
I reclassify each layer to a 1-5 score and then multiply by the weights.

Target crop: Maize
Weights (from FAO guidelines):
  Soil Moisture 40%, Soil pH 20%, Slope 20%, Rainfall 20%
"""

import numpy as np
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MCE_WEIGHTS, RECLASS_TABLES, SUITABILITY_CLASSES,
    TARGET_CROP, DATA_PROCESSED, OUTPUT_REPORTS
)
from utils.raster_utils import reclassify, weighted_overlay


def print_weight_table():
    """prints out the weights I used"""
    print(f"\n  Target Crop: {TARGET_CROP}")
    print(f"  Method: Weighted Linear Combination\n")
    for name, weight in MCE_WEIGHTS.items():
        label = name.replace("_", " ").title()
        print(f"    {label}: {weight*100:.0f}%")
    print(f"    Total: {sum(MCE_WEIGHTS.values())*100:.0f}%\n")


def print_reclass_table(name):
    """shows the scoring table for a layer"""
    if name not in RECLASS_TABLES:
        return
    table = RECLASS_TABLES[name]
    label = name.replace("_", " ").title()
    print(f"  Reclassification for {label}:")
    for low, high, score, lbl in table:
        print(f"    {low:.1f} - {high:.1f} -> Score {score} ({lbl})")
    print()


def run_mce_workflow(layers):
    """
    Main MCE function. Reclassifies layers to 1-5 scores,
    applies weights, and returns the classified map.
    """
    print_weight_table()

    reclass_layers = {}
    print("  Step 1: Reclassifying layers to 1-5 scores...")

    for name in MCE_WEIGHTS.keys():
        if name not in layers:
            raise KeyError(f"Missing layer: {name}")

        print_reclass_table(name)
        reclass_layers[name] = reclassify(layers[name], RECLASS_TABLES[name])

        # show the class percentages
        total = reclass_layers[name].size
        if total > 0:
            stats = []
            for score in range(1, 6):
                count = np.sum(reclass_layers[name] == score)
                if count > 0:
                    stats.append(f"Class {score}: {count/total*100:.1f}%")
            if stats:
                print(f"    {name}: {', '.join(stats)}\n")

    print("  Step 2: Applying weighted overlay...")
    suitability_continuous = weighted_overlay(reclass_layers, MCE_WEIGHTS)
    valid_mask = ~np.isnan(suitability_continuous) & (suitability_continuous > 0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(f"    Suitability range: {np.nanmin(suitability_continuous):.2f} to {np.nanmax(suitability_continuous):.2f}\n")

    print("  Step 3: Classifying into suitability categories...")
    suitability_classified = np.full_like(suitability_continuous, np.nan)
    suitability_classified[valid_mask] = np.round(suitability_continuous[valid_mask])
    suitability_classified = np.clip(suitability_classified, 1, 5)

    total_valid = np.sum(valid_mask)
    report_lines = []
    if total_valid > 0:
        for i in range(1, 6):
            class_name = "Unknown"
            for (low, high), label in SUITABILITY_CLASSES.items():
                if low <= i < high or (i == 5 and high == 5.1):
                    class_name = label
                    break
            count = np.sum(suitability_classified == i)
            pct = count / total_valid * 100
            line = f"    Class {i} ({class_name}): {pct:.1f}%"
            print(line)
            report_lines.append(line)

    # save results
    os.makedirs(OUTPUT_REPORTS, exist_ok=True)
    report_path = os.path.join(OUTPUT_REPORTS, "mce_report.txt")
    with open(report_path, "w") as f:
        f.write("MCE Report\n")
        f.write(f"Crop: {TARGET_CROP}\n\n")
        f.write("Class Distribution:\n")
        f.write("\n".join(report_lines))

    print(f"\n    Saved report: {report_path}\n")

    return suitability_continuous, suitability_classified
