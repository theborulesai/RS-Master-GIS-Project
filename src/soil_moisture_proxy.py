"""
src/soil_moisture_proxy.py
--------------------------
Phase 2: Soil Moisture Proxy Methodology

This is where I calculate the proxies for soil moisture using the
Temperature-Vegetation Dryness Index (TVDI) and Topographic Wetness Index (TWI).

The idea comes from Sandholt et al. (2002):
- High NDVI + Low LST = Wet soil (max evapotranspiration)
- Low NDVI + High LST = Dry soil (water stressed)

Then I combine it with TWI (from the DEM) to see where water naturally pools.
"""

import numpy as np
from scipy.ndimage import uniform_filter
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TVDI_CONFIG, DATA_PROCESSED
from utils.raster_utils import normalise


# ──────────────────────────────────────────────
#  STEP 1: Compute NDVI
# ──────────────────────────────────────────────
def compute_ndvi(nir, red):
    """
    Standard NDVI calculation using Near-Infrared and Red bands.
    I didn't actually need to use this function in the final pipeline 
    since I downloaded pre-computed NDVI from MODIS, but kept it here 
    to show the formula: (NIR - Red) / (NIR + Red).
    """
    denom = nir + red
    denom[denom == 0] = 1e-10  # stop division by zero errors
    ndvi = (nir - red) / denom
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


# ──────────────────────────────────────────────
#  STEP 2: LST Retrieval
# ──────────────────────────────────────────────
def retrieve_lst_from_thermal(thermal_band, emissivity=0.95):
    """
    Simplified mono-window algorithm to get LST from thermal bands.
    Again, I just used the pre-computed MOD11A2 LST in the end to save time,
    but here's the code to do it manually if I had raw Landsat thermal data.
    """
    lam = 10.895e-6       # thermal band wavelength
    rho = 1.438e-2        
    ln_e = np.log(emissivity)

    lst = thermal_band / (1.0 + (lam * thermal_band / rho) * ln_e)
    return lst.astype(np.float32)


# ──────────────────────────────────────────────
#  STEP 3: Topographic Wetness Index (TWI)
# ──────────────────────────────────────────────
def compute_twi(dem, pixel_size=250.0):
    """
    Calculates TWI = ln(catchment_area / slope).
    I used a simple neighbourhood mean filter as a rough proxy for
    flow accumulation, instead of fighting with complex D8 routing algorithms
    in Python. It works well enough for a 250m resolution grid!
    """
    # get the slope in radians
    dy, dx = np.gradient(dem, pixel_size)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_rad = np.maximum(slope_rad, 0.001)  # avoid log(0) crash

    # simplified flow accumulation trick
    inverted = dem.max() - dem
    flow_acc = uniform_filter(inverted, size=7)
    flow_acc = np.maximum(flow_acc, 1.0)

    # local catchment area
    a = flow_acc * pixel_size

    twi = np.log(a / np.tan(slope_rad))
    return twi.astype(np.float32)


# ──────────────────────────────────────────────
#  STEP 4: TVDI computation
# ──────────────────────────────────────────────
def compute_tvdi(ndvi, lst, n_bins=None):
    """
    This is the core of the proxy method.
    I bin the NDVI values into slices, and find the max LST (dry edge)
    and min LST (wet edge) for each slice. Then I fit two lines.

    TVDI = (LST_observed - LST_wet) / (LST_dry - LST_wet)
    """
    if n_bins is None:
        n_bins = TVDI_CONFIG["n_bins"]

    ndvi_flat = ndvi.ravel()
    lst_flat = lst.ravel()

    # bin NDVI to find edges
    ndvi_min, ndvi_max = 0.05, 0.90  
    bin_edges = np.linspace(ndvi_min, ndvi_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    lst_max_per_bin = []
    lst_min_per_bin = []
    valid_centers = []

    for i in range(n_bins):
        mask = (ndvi_flat >= bin_edges[i]) & (ndvi_flat < bin_edges[i + 1])
        if mask.sum() > 10:
            lst_in_bin = lst_flat[mask]
            # using 95th/5th percentiles instead of true min/max to drop outliers
            lst_max_per_bin.append(np.percentile(lst_in_bin, 95))
            lst_min_per_bin.append(np.percentile(lst_in_bin, 5))
            valid_centers.append(bin_centers[i])

    valid_centers = np.array(valid_centers)
    lst_max_per_bin = np.array(lst_max_per_bin)
    lst_min_per_bin = np.array(lst_min_per_bin)

    # find the dry and wet edges using linear regression
    b_dry, a_dry = np.polyfit(valid_centers, lst_max_per_bin, 1)
    b_wet, a_wet = np.polyfit(valid_centers, lst_min_per_bin, 1)

    # calculate pixel TVDI
    lst_max_surface = a_dry + b_dry * ndvi
    lst_min_surface = a_wet + b_wet * ndvi

    denom = lst_max_surface - lst_min_surface
    denom[denom == 0] = 1e-10

    tvdi = (lst - lst_min_surface) / denom
    tvdi = np.clip(tvdi, 0.0, 1.0)

    edge_params = {
        "dry_slope": float(b_dry),
        "dry_intercept": float(a_dry),
        "wet_slope": float(b_wet),
        "wet_intercept": float(a_wet),
    }

    return tvdi.astype(np.float32), edge_params


# ──────────────────────────────────────────────
#  STEP 5: Final Composite Map
# ──────────────────────────────────────────────
def compute_soil_moisture_index(tvdi, twi, w_tvdi=0.6, w_twi=0.4):
    """
    SMI = 0.6 * (1 - normal_TVDI) + 0.4 * normal_TWI
    
    I use 1 - TVDI because high TVDI means dry, but I want my SMI 
    map to show high numbers for wet areas!
    """
    tvdi_norm = normalise(tvdi, 0, 1)
    twi_norm = normalise(twi, 0, 1)

    smi = w_tvdi * (1.0 - tvdi_norm) + w_twi * twi_norm
    smi = normalise(smi, 0, 1)
    return smi.astype(np.float32)


def run_soil_moisture_workflow(layers):
    """
    Runs the whole TVDI+TWI pipeline and saves the numpy arrays.
    """
    print("\n" + "=" * 70)
    print("  PHASE 2: SOIL MOISTURE PROXY METHODOLOGY")
    print("=" * 70)

    ndvi = layers["ndvi"]
    lst = layers["lst"]
    dem = layers["dem"]

    print("\n  [Step 1] Computing Topographic Wetness Index (TWI) ...")
    twi = compute_twi(dem)
    print(f"    TWI range: [{twi.min():.2f}, {twi.max():.2f}]")

    print("  [Step 2] Computing TVDI (Temperature-Vegetation Dryness Index) ...")
    tvdi, edge_params = compute_tvdi(ndvi, lst)
    print(f"    TVDI range: [{tvdi.min():.3f}, {tvdi.max():.3f}]")
    print(f"    Dry edge:  LST_max = {edge_params['dry_intercept']:.1f} "
          f"+ {edge_params['dry_slope']:.2f} × NDVI")
    print(f"    Wet edge:  LST_min = {edge_params['wet_intercept']:.1f} "
          f"+ {edge_params['wet_slope']:.2f} × NDVI")

    print("  [Step 3] Computing composite Soil Moisture Index ...")
    smi = compute_soil_moisture_index(tvdi, twi)
    print(f"    SMI range: [{smi.min():.3f}, {smi.max():.3f}]")

    layers["twi"] = twi
    layers["tvdi"] = tvdi
    layers["soil_moisture"] = smi

    # save processed layers
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    for key in ["twi", "tvdi", "soil_moisture"]:
        np.save(os.path.join(DATA_PROCESSED, f"{key}.npy"), layers[key])
        print(f"    -> Saved {key}.npy")

    return layers
