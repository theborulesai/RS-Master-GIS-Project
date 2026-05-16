"""
raster_utils.py
---------------
Helper functions for raster processing.
Normalise, reclassify, weighted overlay, etc.
"""

import numpy as np
import os


def normalise(arr, new_min=0.0, new_max=1.0):
    """min-max normalisation to [new_min, new_max]"""
    old_min, old_max = arr.min(), arr.max()
    if old_max - old_min < 1e-10:
        return np.full_like(arr, (new_min + new_max) / 2.0)
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def reclassify(arr, reclass_table):
    """
    Reclassifies continuous values to discrete scores.
    reclass_table is a list of (min, max, score, label) tuples.
    """
    result = np.zeros_like(arr, dtype=np.int32)
    for low, high, score, _label in reclass_table:
        mask = (arr >= low) & (arr < high)
        result[mask] = score
    result[result == 0] = 1  # fallback
    return result


def weighted_overlay(layers_dict, weights_dict):
    """multiplies each layer by its weight and adds them up"""
    total = sum(weights_dict.values())
    assert abs(total - 1.0) < 1e-6, f"Weights should sum to 1.0, got {total}"

    result = None
    for name, weight in weights_dict.items():
        layer = layers_dict[name].astype(np.float64) * weight
        result = layer if result is None else result + layer
    return result
