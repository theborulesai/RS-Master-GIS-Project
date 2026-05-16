"""
src/validation.py
-----------------
Phase 5: Accuracy Assessment

This file just contains the maths for computing the confusion matrix
and the Kappa coefficient so I can compare my ML predictions against
my MCE results.

Cohen's Kappa (κ) tells me how much better my classification is 
compared to just random guessing.
"""

import numpy as np


def compute_confusion_matrix(predicted, observed, n_classes=5):
    """
    Builds a standard confusion matrix.
    Rows are the observed (ground truth) classes.
    Columns are what the model predicted.
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, o in zip(predicted, observed):
        # minus 1 because classes are 1-5 but arrays are 0-indexed
        cm[o - 1, p - 1] += 1
    return cm


def compute_accuracy_metrics(cm):
    """
    Calculates Overall Accuracy, Producer's/User's accuracy,
    and Cohen's Kappa coefficient from the confusion matrix.
    """
    n = cm.sum()
    n_classes = cm.shape[0]

    # overall accuracy (sum of diagonal / total pixels)
    oa = cm.diagonal().sum() / n

    # producer's accuracy (how often is real ground-truth correctly shown on map)
    col_sums = cm.sum(axis=0)
    pa = np.zeros(n_classes)
    for i in range(n_classes):
        pa[i] = cm[i, i] / col_sums[i] if col_sums[i] > 0 else 0.0

    # user's accuracy (if map says class X, how often is it actually class X)
    row_sums = cm.sum(axis=1)
    ua = np.zeros(n_classes)
    for i in range(n_classes):
        ua[i] = cm[i, i] / row_sums[i] if row_sums[i] > 0 else 0.0

    # Cohen's Kappa (P_observed - P_expected) / (1 - P_expected)
    p_o = oa
    p_e = sum(row_sums[i] * col_sums[i] for i in range(n_classes)) / (n ** 2)
    kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) > 0 else 0.0

    return {
        "overall_accuracy": float(oa),
        "producers_accuracy": pa.tolist(),
        "users_accuracy": ua.tolist(),
        "kappa_coefficient": float(kappa),
        "total_points": int(n),
    }
