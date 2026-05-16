#!/usr/bin/env python3
"""
run_pipeline.py
---------------
Main script for my Remote Sensing & GIS course project.
Loads the real satellite datasets, calculates soil moisture proxies,
trains 3 ML models (Random Forest, SVM, Gradient Boosting),
does the MCE weighted overlay, and compares everything.

Study area: Bidar district, Karnataka
Year: 2023

Just run:
    python run_pipeline.py
"""

import os
import sys
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_MAPS, OUTPUT_REPORTS, DATA_PROCESSED
from utils.real_data_loader import load_all_real_layers
from src.soil_moisture_proxy import compute_twi, compute_tvdi, compute_soil_moisture_index
from src.multi_criteria_eval import run_mce_workflow
from src.ml_training import (
    prepare_feature_matrix, train_all_models,
    predict_suitability_map, feature_importance_report,
    save_comparison_report,
)
from src.validation import compute_confusion_matrix, compute_accuracy_metrics
from utils.visualization import (
    plot_continuous_map, plot_suitability_map,
    plot_confusion_matrix as plot_cm, plot_all_layers,
    plot_model_comparison,
)


def fill_nans(arr):
    """fills NaN gaps with the median value"""
    if np.isnan(arr).all():
        return np.zeros_like(arr)
    out = arr.copy()
    out[np.isnan(out)] = np.nanmedian(arr)
    return out


def main():
    start = time.time()

    print("\n===== REMOTE SENSING & GIS COURSE PROJECT =====")
    print("Soil Moisture Proxy + Crop Suitability - Bidar, Karnataka")
    print("=" * 50)

    # --- Phase 1: load the real raster datasets ---
    print("\n--- Phase 1: Loading real satellite data ---")
    layers = load_all_real_layers(verbose=True)

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    for name, arr in layers.items():
        np.save(os.path.join(DATA_PROCESSED, f"{name}.npy"), arr)

    # --- Phase 2: compute derived indices ---
    print("\n--- Phase 2: Computing soil moisture proxies (TVDI + TWI) ---")

    dem_filled = fill_nans(layers["dem"])

    print("  Computing TWI from DEM...")
    layers["twi"] = compute_twi(dem_filled, pixel_size=250.0)
    print(f"  TWI range: {layers['twi'].min():.2f} to {layers['twi'].max():.2f}")

    print("  Computing TVDI (triangle method)...")
    ndvi_f = fill_nans(layers["ndvi"])
    lst_f = fill_nans(layers["lst"])
    layers["tvdi"], edges = compute_tvdi(ndvi_f, lst_f)
    print(f"  TVDI range: {layers['tvdi'].min():.3f} to {layers['tvdi'].max():.3f}")
    print(f"  Dry edge: LST_max = {edges['dry_intercept']:.1f} + {edges['dry_slope']:.2f} * NDVI")
    print(f"  Wet edge: LST_min = {edges['wet_intercept']:.1f} + {edges['wet_slope']:.2f} * NDVI")

    print("  Computing composite Soil Moisture Index...")
    layers["soil_moisture"] = compute_soil_moisture_index(layers["tvdi"], layers["twi"])
    print(f"  SMI range: {layers['soil_moisture'].min():.3f} to {layers['soil_moisture'].max():.3f}")

    for key in ["twi", "tvdi", "soil_moisture"]:
        np.save(os.path.join(DATA_PROCESSED, f"{key}.npy"), layers[key])

    # --- Phase 3: ML training (3 algorithms) ---
    print("\n--- Phase 3: Training 3 ML classifiers ---")

    print("  Building feature matrix...")
    X, y, valid_mask, feature_names = prepare_feature_matrix(layers)

    # train all 3 models on the same train/test split
    results = train_all_models(X, y)

    # print feature importances for RF and GB
    for algo_name in ["Random Forest", "Gradient Boosting"]:
        if "importances" in results[algo_name]:
            print()
            feature_importance_report(results[algo_name]["importances"], algo_name)

    # generate suitability maps for each model
    print("\n  Generating suitability prediction maps for all 3 models...")
    model_maps = {}
    model_shortnames = {
        "Random Forest": "rf",
        "SVM": "svm",
        "Gradient Boosting": "gb",
    }
    for algo_name, res in results.items():
        scaler = res.get("scaler", None)
        pred_map = predict_suitability_map(
            res["model"], layers, valid_mask, feature_names, scaler=scaler
        )
        model_maps[algo_name] = pred_map
        short = model_shortnames[algo_name]
        np.save(os.path.join(DATA_PROCESSED, f"ml_suitability_{short}.npy"), pred_map)
        print(f"    {algo_name}: map generated ({pred_map.shape})")

    # save comparison report
    save_comparison_report(results)

    # --- Phase 4: MCE weighted overlay ---
    print("\n--- Phase 4: Multi-Criteria Evaluation (weighted overlay) ---")

    for key in ["soil_moisture", "soil_ph", "slope", "rainfall"]:
        layers[key] = fill_nans(layers[key])

    mce_continuous, mce_classified = run_mce_workflow(layers)
    np.save(os.path.join(DATA_PROCESSED, "mce_classified.npy"), mce_classified)

    # --- Phase 5: validation ---
    print("\n--- Phase 5: Accuracy assessment ---")

    # compare best ML model (RF) against MCE
    rf_map = model_maps["Random Forest"]
    ml_flat = rf_map.ravel()
    mce_flat = mce_classified.ravel()
    valid = (ml_flat > 0) & (mce_flat > 0)

    cm_val = compute_confusion_matrix(
        ml_flat[valid].astype(int),
        mce_flat[valid].astype(int),
    )
    val_metrics = compute_accuracy_metrics(cm_val)

    print(f"\n  ML (RF) vs MCE agreement: {val_metrics['overall_accuracy']*100:.2f}%")
    print(f"  Kappa: {val_metrics['kappa_coefficient']:.4f}")

    # print all 3 model accuracies
    print("\n  Individual ML Model Results:")
    for name, res in results.items():
        print(f"    {name}: Accuracy={res['overall_accuracy']*100:.2f}%, "
              f"Kappa={res['kappa']:.4f}")

    # save validation report
    os.makedirs(OUTPUT_REPORTS, exist_ok=True)
    class_labels = ["S1", "S2", "S3", "S4", "S5"]
    val_report_path = os.path.join(OUTPUT_REPORTS, "validation_report.txt")
    with open(val_report_path, "w") as f:
        f.write("Validation Report\n")
        f.write("=" * 50 + "\n")
        f.write("ML (Random Forest) vs MCE Agreement\n\n")
        f.write(f"Overall Agreement: {val_metrics['overall_accuracy']*100:.2f}%\n")
        f.write(f"Cohen's Kappa:     {val_metrics['kappa_coefficient']:.4f}\n")
        f.write(f"Total Pixels:      {val_metrics['total_points']}\n\n")
        f.write("Confusion Matrix (rows=Observed/MCE, cols=Predicted/ML):\n")
        f.write("        " + "  ".join(f"{c:>6s}" for c in class_labels) + "\n")
        for i, row_label in enumerate(class_labels):
            row_vals = "  ".join(f"{cm_val[i, j]:6d}" for j in range(5))
            f.write(f"  {row_label:>4s}  {row_vals}\n")
        f.write("\nPer-Class Accuracy:\n")
        f.write(f"  {'Class':<6s}  {'Producer':>10s}  {'User':>10s}\n")
        for i, cl in enumerate(class_labels):
            pa = val_metrics['producers_accuracy'][i] * 100
            ua = val_metrics['users_accuracy'][i] * 100
            f.write(f"  {cl:<6s}  {pa:>9.2f}%  {ua:>9.2f}%\n")
        f.write("\nIndividual ML Model Results:\n")
        f.write(f"  {'Algorithm':<22s}  {'Accuracy':>10s}  {'Kappa':>8s}\n")
        f.write("  " + "-" * 44 + "\n")
        for name, res in results.items():
            f.write(f"  {name:<22s}  {res['overall_accuracy']*100:>9.2f}%  {res['kappa']:>8.4f}\n")
    print(f"\n  Saved: {val_report_path}")

    # --- Generate output maps ---
    print("\n--- Generating output maps ---")
    os.makedirs(OUTPUT_MAPS, exist_ok=True)

    map_layers = {k: layers[k] for k in
        ["ndvi", "lst", "dem", "slope", "rainfall", "soil_ph", "twi", "tvdi", "soil_moisture"]}
    print("  Saving layer maps...")
    plot_all_layers(map_layers, output_prefix="real")

    print("  MCE suitability map...")
    plot_suitability_map(
        mce_classified,
        title="Crop Suitability - Maize - Bidar (MCE)",
        output_name="mce_suitability_bidar.png"
    )

    # suitability maps for all 3 models
    algo_titles = {
        "Random Forest": "Crop Suitability - Maize - Bidar (Random Forest)",
        "SVM": "Crop Suitability - Maize - Bidar (SVM)",
        "Gradient Boosting": "Crop Suitability - Maize - Bidar (Gradient Boosting)",
    }
    for algo_name, pred_map in model_maps.items():
        short = model_shortnames[algo_name]
        print(f"  {algo_name} suitability map...")
        plot_suitability_map(
            np.where(pred_map > 0, pred_map, 1),
            title=algo_titles[algo_name],
            output_name=f"ml_suitability_{short}.png"
        )

    # confusion matrices for all 3 models
    print("  Confusion matrices...")
    class_labels = ["S1", "S2", "S3", "S4", "S5"]
    for algo_name, res in results.items():
        short = model_shortnames[algo_name]
        plot_cm(res["cm"], class_labels, output_name=f"ml_confusion_{short}.png")

    # MCE vs RF comparison matrix
    plot_cm(cm_val, class_labels, output_name="mce_vs_ml_matrix.png")

    # comparison bar chart
    print("  Model comparison chart...")
    plot_model_comparison(results)

    # --- done ---
    elapsed = time.time() - start
    print("\n" + "=" * 50)
    print("DONE!")
    print(f"  Time taken: {elapsed:.1f} seconds")
    print(f"  3 ML models trained and compared")
    for name, res in results.items():
        print(f"    {name}: {res['overall_accuracy']*100:.2f}% accuracy")
    print(f"  Maps saved to: {OUTPUT_MAPS}")
    print(f"  Reports saved to: {OUTPUT_REPORTS}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
