"""
config.py
---------
All my project settings in one place.
Study area, weights, thresholds, file paths etc.
"""

import os

# -- file paths --
PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
DATA_RAW       = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_MAPS    = os.path.join(PROJECT_ROOT, "outputs", "maps")
OUTPUT_REPORTS = os.path.join(PROJECT_ROOT, "outputs", "reports")

# -- study area: Bidar district, Karnataka --
STUDY_AREA = {
    "west":  76.800,
    "south": 17.600,
    "east":  77.600,
    "north": 18.300,
}

CRS          = "EPSG:4326"
PIXEL_SIZE   = 250.0       # metres
RASTER_SHAPE = (311, 356)  # approx grid size at 250m

# -- TVDI settings --
TVDI_CONFIG = {
    "ndvi_range": (0.0, 1.0),
    "lst_range_K": (290.0, 330.0),
    "n_bins": 50,
}

# -- MCE weights for Maize --
# these add up to 1.0
# I picked these based on FAO crop requirement guidelines
TARGET_CROP = "Maize (Zea mays)"

MCE_WEIGHTS = {
    "soil_moisture": 0.40,
    "soil_ph":       0.20,
    "slope":         0.20,
    "rainfall":      0.20,
}

# reclassification tables: (min, max, score, label)
# score 1 = bad, 5 = great for maize
RECLASS_TABLES = {
    "soil_moisture": [
        (0.00, 0.15, 1, "Very Low"),
        (0.15, 0.30, 2, "Low"),
        (0.30, 0.50, 3, "Moderate"),
        (0.50, 0.70, 4, "High"),
        (0.70, 1.00, 5, "Very High"),
    ],
    "soil_ph": [
        (0.0, 4.5, 1, "Strongly Acidic"),
        (4.5, 5.5, 2, "Moderately Acidic"),
        (5.5, 6.5, 4, "Slightly Acidic (Optimal)"),
        (6.5, 7.5, 5, "Neutral (Optimal)"),
        (7.5, 14.0, 3, "Alkaline"),
    ],
    "slope": [
        (0,   2,  5, "Flat"),
        (2,   5,  4, "Gentle"),
        (5,  10,  3, "Moderate"),
        (10, 20,  2, "Steep"),
        (20, 90,  1, "Very Steep"),
    ],
    "rainfall": [
        (0,    400, 1, "Very Low"),
        (400,  600, 2, "Low"),
        (600,  900, 4, "Moderate (Optimal)"),
        (900, 1200, 5, "High (Optimal)"),
        (1200, 5000, 3, "Excessive"),
    ],
}

SUITABILITY_CLASSES = {
    (1.0, 1.5): "Not Suitable (S5)",
    (1.5, 2.5): "Marginally Suitable (S4)",
    (2.5, 3.5): "Moderately Suitable (S3)",
    (3.5, 4.5): "Suitable (S2)",
    (4.5, 5.1): "Highly Suitable (S1)",
}

# -- ML settings --
N_GROUND_TRUTH_POINTS = 100
VALIDATION_SPLIT      = 0.30
RANDOM_SEED           = 42
