"""
ml_training.py
--------------
Machine learning part of my Remote Sensing project.

I train 3 different classifiers to predict soil moisture
suitability classes and compare their performance:
  1. Random Forest (ensemble of decision trees)
  2. Support Vector Machine (SVM with RBF kernel)
  3. Gradient Boosting (sequential boosting ensemble)

Features: NDVI, LST, Slope, TWI, Rainfall, Soil pH
Target: SWI discretised into 5 classes (1=dry, 5=wet)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RANDOM_SEED, OUTPUT_REPORTS


CLASS_LABELS = {
    1: "Not Suitable",
    2: "Marginally Suitable",
    3: "Moderately Suitable",
    4: "Suitable",
    5: "Highly Suitable",
}

# thresholds to convert SWI (0-1) into classes
SWI_THRESHOLDS = [0.0, 0.20, 0.40, 0.60, 0.80, 1.01]


def swi_to_class(swi_arr):
    """converts continuous SWI values to discrete classes 1-5"""
    labels = np.zeros_like(swi_arr, dtype=np.int32)
    for cls in range(1, 6):
        lo, hi = SWI_THRESHOLDS[cls - 1], SWI_THRESHOLDS[cls]
        labels[(swi_arr >= lo) & (swi_arr < hi)] = cls
    labels[labels == 0] = 1
    return labels


def prepare_feature_matrix(layers):
    """
    Stacks all raster layers into a 2D feature matrix (pixels x features)
    and creates the label array from SWI.
    Only uses pixels where all values are valid (no NaN).
    """
    feature_names = ["ndvi", "lst", "slope", "twi", "rainfall", "soil_ph"]
    features = []
    for name in feature_names:
        if name not in layers:
            raise KeyError(f"Missing layer: {name}")
        features.append(layers[name].ravel())

    X_raw = np.column_stack(features)
    y_raw = swi_to_class(layers["swi"].ravel())

    # keep only valid pixels
    valid = np.all(np.isfinite(X_raw), axis=1) & (y_raw > 0)
    X = X_raw[valid].astype(np.float32)
    y = y_raw[valid].astype(np.int32)

    print(f"    Total pixels: {len(y_raw)}")
    print(f"    Valid pixels: {len(y)} ({len(y)/len(y_raw)*100:.1f}%)")
    print(f"    Class counts:")
    for cls in range(1, 6):
        cnt = (y == cls).sum()
        print(f"      Class {cls} ({CLASS_LABELS[cls]}): {cnt} ({cnt/len(y)*100:.1f}%)")

    return X, y, valid.reshape(layers["ndvi"].shape), feature_names


# ──────────────────────────────────────────────
#  Algorithm 1: Random Forest
# ──────────────────────────────────────────────
def _train_random_forest(X_train, y_train, X_test, y_test):
    """
    Random Forest — my favourite one.
    Uses 150 trees with balanced class weights to handle
    the imbalanced SWI distribution in Bidar.
    """
    print(f"\n  [1/3] Training Random Forest (150 trees)...")

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    oa = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5])

    print(f"    Overall Accuracy: {oa*100:.2f}%")
    print(f"    Kappa: {kappa:.4f}")

    # feature importances (only RF has this built-in)
    importances = dict(zip(
        ["ndvi", "lst", "slope", "twi", "rainfall", "soil_ph"],
        model.feature_importances_
    ))

    return model, cm, oa, kappa, y_pred, importances


# ──────────────────────────────────────────────
#  Algorithm 2: Support Vector Machine
# ──────────────────────────────────────────────
def _train_svm(X_train, y_train, X_test, y_test, scaler):
    """
    SVM with RBF (radial basis function) kernel.
    SVMs need feature scaling so I standardise the data first.
    I use a subset for training if data is too large because
    SVM is slow on huge datasets (O(n^2) complexity).
    """
    print(f"\n  [2/3] Training SVM (RBF kernel)...")

    # SVM is slow on large datasets, so subsample if needed
    max_samples = 50000
    if len(X_train) > max_samples:
        print(f"    Subsampling to {max_samples} for SVM (too slow otherwise)")
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(X_train), max_samples, replace=False)
        X_tr = X_train[idx]
        y_tr = y_train[idx]
    else:
        X_tr = X_train
        y_tr = y_train

    # scale features — SVM is sensitive to feature magnitude
    X_tr_scaled = scaler.transform(X_tr)
    X_te_scaled = scaler.transform(X_test)

    model = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        random_state=RANDOM_SEED,
        max_iter=5000,
    )
    model.fit(X_tr_scaled, y_tr)
    y_pred = model.predict(X_te_scaled)

    oa = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5])

    print(f"    Overall Accuracy: {oa*100:.2f}%")
    print(f"    Kappa: {kappa:.4f}")

    return model, cm, oa, kappa, y_pred


# ──────────────────────────────────────────────
#  Algorithm 3: Gradient Boosting
# ──────────────────────────────────────────────
def _train_gradient_boosting(X_train, y_train, X_test, y_test):
    """
    Gradient Boosting — builds trees sequentially, each one
    trying to fix the mistakes of the previous one.
    Usually gives better accuracy than RF but takes longer.
    I use 200 estimators with a learning rate of 0.1.
    """
    print(f"\n  [3/3] Training Gradient Boosting (200 estimators)...")

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    oa = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5])

    print(f"    Overall Accuracy: {oa*100:.2f}%")
    print(f"    Kappa: {kappa:.4f}")

    # GB also has feature importances
    importances = dict(zip(
        ["ndvi", "lst", "slope", "twi", "rainfall", "soil_ph"],
        model.feature_importances_
    ))

    return model, cm, oa, kappa, y_pred, importances


# ──────────────────────────────────────────────
#  Train all 3 models and compare
# ──────────────────────────────────────────────
def train_all_models(X, y):
    """
    Trains all 3 classifiers on the same train/test split
    so we can compare them fairly. Returns a dict with results.
    """
    # same split for all models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    print(f"    Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # scaler for SVM (fitted on training data only)
    scaler = StandardScaler()
    scaler.fit(X_train)

    results = {}

    # --- 1. Random Forest ---
    rf_model, rf_cm, rf_oa, rf_kappa, rf_pred, rf_imp = _train_random_forest(
        X_train, y_train, X_test, y_test
    )
    results["Random Forest"] = {
        "model": rf_model, "cm": rf_cm,
        "overall_accuracy": rf_oa, "kappa": rf_kappa,
        "y_pred": rf_pred, "importances": rf_imp,
    }

    # --- 2. SVM ---
    svm_model, svm_cm, svm_oa, svm_kappa, svm_pred = _train_svm(
        X_train, y_train, X_test, y_test, scaler
    )
    results["SVM"] = {
        "model": svm_model, "cm": svm_cm,
        "overall_accuracy": svm_oa, "kappa": svm_kappa,
        "y_pred": svm_pred, "scaler": scaler,
    }

    # --- 3. Gradient Boosting ---
    gb_model, gb_cm, gb_oa, gb_kappa, gb_pred, gb_imp = _train_gradient_boosting(
        X_train, y_train, X_test, y_test
    )
    results["Gradient Boosting"] = {
        "model": gb_model, "cm": gb_cm,
        "overall_accuracy": gb_oa, "kappa": gb_kappa,
        "y_pred": gb_pred, "importances": gb_imp,
    }

    # store common test data
    for name in results:
        results[name]["y_test"] = y_test
        results[name]["X_test"] = X_test

    # print comparison table
    print("\n  " + "=" * 55)
    print("  MODEL COMPARISON SUMMARY")
    print("  " + "=" * 55)
    print(f"  {'Algorithm':<22} {'Accuracy':>10} {'Kappa':>10}")
    print("  " + "-" * 55)
    for name, res in results.items():
        print(f"  {name:<22} {res['overall_accuracy']*100:>9.2f}% {res['kappa']:>10.4f}")
    print("  " + "=" * 55)

    # find best model
    best_name = max(results, key=lambda k: results[k]["overall_accuracy"])
    print(f"\n  Best model by accuracy: {best_name} "
          f"({results[best_name]['overall_accuracy']*100:.2f}%)")

    return results


def predict_suitability_map(model, layers, valid_mask, feature_names, scaler=None):
    """
    Applies a trained model to produce a full suitability map.
    If scaler is provided (for SVM), scales the features first.
    """
    shape = layers["ndvi"].shape
    features = [layers[n].ravel() for n in feature_names]
    X_all = np.column_stack(features).astype(np.float32)

    pred_map = np.zeros(shape[0] * shape[1], dtype=np.int32)
    valid_flat = valid_mask.ravel()

    if valid_flat.sum() > 0:
        X_valid = X_all[valid_flat]
        if scaler is not None:
            X_valid = scaler.transform(X_valid)
        pred_map[valid_flat] = model.predict(X_valid)

    pred_map = pred_map.reshape(shape)
    pred_map[~valid_mask] = 0

    return pred_map


def feature_importance_report(importances, algo_name="Random Forest"):
    """prints which features matter most for a given model"""
    print(f"  Feature importances ({algo_name}):")
    for name in sorted(importances, key=importances.get, reverse=True):
        print(f"    {name:<12}: {importances[name]:.4f}")
    return importances


def save_comparison_report(results, output_dir=OUTPUT_REPORTS):
    """saves the comparison of all 3 models to a text file"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "ml_comparison_report.txt")

    with open(path, "w") as f:
        f.write("ML Model Comparison Report\n")
        f.write("=" * 50 + "\n")
        f.write("Soil Moisture Suitability Classification - Bidar\n")
        f.write("3 Algorithms Compared on Same Train/Test Split\n\n")

        f.write(f"{'Algorithm':<22} {'Accuracy':>10} {'Kappa':>10}\n")
        f.write("-" * 45 + "\n")
        for name, res in results.items():
            f.write(f"{name:<22} {res['overall_accuracy']*100:>9.2f}% "
                    f"{res['kappa']:>10.4f}\n")
        f.write("\n")

        # feature importances for RF and GB
        for algo_name in ["Random Forest", "Gradient Boosting"]:
            if algo_name in results and "importances" in results[algo_name]:
                f.write(f"\nFeature Importances ({algo_name}):\n")
                imp = results[algo_name]["importances"]
                for feat in sorted(imp, key=imp.get, reverse=True):
                    f.write(f"  {feat}: {imp[feat]:.4f}\n")

        # note about SVM
        f.write("\nNote: SVM does not provide feature importances directly.\n")

        # best model
        best = max(results, key=lambda k: results[k]["overall_accuracy"])
        f.write(f"\nBest model by accuracy: {best} "
                f"({results[best]['overall_accuracy']*100:.2f}%)\n")

    print(f"    Saved report: {path}")
    return path


# keep backward compatibility with old pipeline call
def train_model(X, y, model_type="random_forest"):
    """Legacy function — trains just Random Forest (for backward compat)"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    oa = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5])
    return model, cm, {
        "overall_accuracy": oa, "kappa": kappa,
        "y_test": y_test, "y_pred": y_pred, "X_test": X_test,
    }


def save_ml_report(metrics, importances, output_dir=OUTPUT_REPORTS):
    """saves the training results to a text file (legacy)"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "ml_training_report.txt")
    with open(path, "w") as f:
        f.write("ML Training Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Overall Accuracy: {metrics['overall_accuracy']*100:.2f}%\n")
        f.write(f"Kappa: {metrics['kappa']:.4f}\n\n")
        f.write("Feature Importances:\n")
        for feat, val in sorted(importances.items(), key=lambda x: -x[1]):
            f.write(f"  {feat}: {val:.4f}\n")
    print(f"    Saved report: {path}")
    return path
