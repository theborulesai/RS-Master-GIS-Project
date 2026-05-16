"""
utils/real_data_loader.py
--------------------------
Loads all real satellite and soil raster datasets for the Bidar
district GIS project. All data lives inside data/raw/ so the project
is fully self-contained — no external paths needed.

Datasets (data/raw/):
    ndvi_lst/  — MODIS MOD13Q1 NDVI + MOD11A2 LST (NASA AppEEARS, 2023)
    dem/       — SRTM GL1 30m DEM (OpenTopography)
    soil_ph/   — SoilGrids pH-water 5-15cm (ISRIC)
    rainfall/  — CHIRPS monthly 2023 .tif.gz (UCSB CHRS)
    swi/       — Soil Water Index (Copernicus CLMS, for validation)

Study Area: Bidar district, Karnataka, India
CRS target : EPSG:4326 (WGS84 geographic)
Resolution : ~250m (0.00225 degrees per pixel)
"""

import os
import glob
import gzip
import shutil
import warnings
import numpy as np
from scipy.ndimage import zoom
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import from_bounds
import tempfile

warnings.filterwarnings("ignore")

# ── All data is self-contained inside data/raw/ ─────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATA_RAW     = os.path.join(_PROJECT_ROOT, "data", "raw")

PATHS = {
    "ndvi_lst_dir": os.path.join(_DATA_RAW, "ndvi_lst"),
    "dem":          os.path.join(_DATA_RAW, "dem",     "dem_srtm_bidar.tif"),
    "soil_ph":      os.path.join(_DATA_RAW, "soil_ph", "soil_ph_bidar.tif"),
    "rainfall_dir": os.path.join(_DATA_RAW, "rainfall"),
    "swi":          os.path.join(_DATA_RAW, "swi",     "soil_water_index_bidar.tif"),
}

# MODIS scale factors (from LP DAAC product documentation)
NDVI_SCALE  = 0.0001   # stored as int16 × 0.0001 → actual NDVI
LST_SCALE   = 0.02     # stored as uint16 × 0.02  → Kelvin
NDVI_NODATA = -3000
LST_NODATA  = 0

# Target grid — Bidar district, EPSG:4326
TARGET_BOUNDS = (76.800, 17.600, 77.600, 18.300)   # west, south, east, north
TARGET_RES    = 0.00225                              # ~250 m in degrees


# ── Internal helpers ─────────────────────────────────────────────────

def _target_grid():
    """Return (width, height, affine_transform) for the common output grid."""
    west, south, east, north = TARGET_BOUNDS
    width  = int((east - west)  / TARGET_RES)
    height = int((north - south) / TARGET_RES)
    transform = from_bounds(west, south, east, north, width, height)
    return width, height, transform


def _scale_and_reproject(src, scaled_array, nodata=np.nan):
    """
    Write a scaled numpy array back into a temp GeoTIFF keeping the
    original CRS/transform, then reproject it to the target grid.
    Returns a (height, width) float32 array in EPSG:4326.
    """
    width, height, transform = _target_grid()

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    with rasterio.open(
        tmp_path, 'w', driver='GTiff',
        height=src.height, width=src.width,
        count=1, dtype='float32',
        crs=src.crs, transform=src.transform,
        nodata=np.nan
    ) as tmp_ds:
        tmp_ds.write(scaled_array.astype(np.float32), 1)

    out = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(tmp_path) as tmp_src:
        reproject(
            source        = rasterio.band(tmp_src, 1),
            destination   = out,
            src_transform = tmp_src.transform,
            src_crs       = tmp_src.crs,
            dst_transform = transform,
            dst_crs       = CRS.from_epsg(4326),
            resampling    = Resampling.bilinear,
            dst_nodata    = np.nan,
        )
    os.unlink(tmp_path)
    return out


# ── NDVI ─────────────────────────────────────────────────────────────

def load_ndvi_clean(verbose=True):
    """
    Load all MODIS MOD13Q1 NDVI 16-day composites for 2023,
    apply scale factor (×0.0001), cloud-mask with pixel_reliability flag,
    and return the annual mean NDVI for Bidar in [-0.2, 1.0].
    """
    ndvi_files = sorted(glob.glob(
        os.path.join(PATHS["ndvi_lst_dir"], "MOD13Q1.061__250m_16_days_NDVI_doy2023*.tif")
    ))
    qa_files = sorted(glob.glob(
        os.path.join(PATHS["ndvi_lst_dir"], "MOD13Q1.061__250m_16_days_pixel_reliability_doy2023*.tif")
    ))
    if verbose:
        print(f"  Found {len(ndvi_files)} NDVI files, {len(qa_files)} QA files")

    stack = []
    for i, ndvi_path in enumerate(ndvi_files):
        with rasterio.open(ndvi_path) as src:
            raw = src.read(1).astype(np.float32)
            raw[raw == NDVI_NODATA] = np.nan
            scaled = np.clip(raw * NDVI_SCALE, -0.2, 1.0)

            # mask pixels flagged as cloudy/snow (reliability > 1)
            if i < len(qa_files):
                with rasterio.open(qa_files[i]) as qa_src:
                    qa = qa_src.read(1)
                    scaled[qa > 1] = np.nan

            stack.append(_scale_and_reproject(src, scaled))

    annual = np.nanmean(stack, axis=0)
    if verbose:
        print(f"    Annual mean NDVI: [{np.nanmin(annual):.3f}, {np.nanmax(annual):.3f}]")
    return annual.astype(np.float32)


# ── LST ──────────────────────────────────────────────────────────────

def load_lst_clean(verbose=True):
    """
    Load all MODIS MOD11A2 LST 8-day composites for 2023,
    apply scale factor (×0.02 → Kelvin), reproject to target grid.
    Returns annual mean LST in Kelvin.
    """
    lst_files = sorted(glob.glob(
        os.path.join(PATHS["ndvi_lst_dir"], "MOD11A2.061_LST_Day_1km_doy2023*.tif")
    ))
    if verbose:
        print(f"  Found {len(lst_files)} LST files")

    stack = []
    for lst_path in lst_files:
        with rasterio.open(lst_path) as src:
            raw = src.read(1).astype(np.float32)
            raw[raw == LST_NODATA] = np.nan
            lst_k = raw * LST_SCALE
            lst_k = np.where((lst_k > 200) & (lst_k < 360), lst_k, np.nan)
            stack.append(_scale_and_reproject(src, lst_k))

    annual = np.nanmean(stack, axis=0)
    if verbose:
        print(f"    Annual mean LST: [{np.nanmin(annual):.1f} K, {np.nanmax(annual):.1f} K]"
              f" = [{np.nanmin(annual)-273.15:.1f}°C, {np.nanmax(annual)-273.15:.1f}°C]")
    return annual.astype(np.float32)


# ── DEM ──────────────────────────────────────────────────────────────

def load_dem(verbose=True):
    """
    Load SRTM 30m DEM. Already in EPSG:4326 — just clip + resample to grid.
    """
    width, height, transform = _target_grid()
    out = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(PATHS["dem"]) as src:
        nodata_val = src.nodata if src.nodata is not None else -32768
        reproject(
            source        = rasterio.band(src, 1),
            destination   = out,
            src_transform = src.transform,
            src_crs       = src.crs,
            dst_transform = transform,
            dst_crs       = CRS.from_epsg(4326),
            resampling    = Resampling.bilinear,
            dst_nodata    = np.nan,
        )
    out[out == nodata_val] = np.nan
    if verbose:
        print(f"    DEM range: [{np.nanmin(out):.1f} m, {np.nanmax(out):.1f} m]")
    return out.astype(np.float32)


# ── Soil pH ───────────────────────────────────────────────────────────

def load_soil_ph(verbose=True):
    """
    Load SoilGrids pH-water at 5-15cm depth.
    SoilGrids stores pH × 10 as int16, so divide by 10 to get actual pH.
    If spatial coverage of Bidar is low, fills with the district typical value (7.6).
    """
    width, height, transform = _target_grid()
    out = np.full((height, width), np.nan, dtype=np.float32)

    with rasterio.open(PATHS["soil_ph"]) as src:
        raw = src.read(1).astype(np.float32)
        raw_scaled = raw / 10.0
        raw_scaled[(raw <= 0) | (raw_scaled > 14)] = np.nan
        out = _scale_and_reproject(src, raw_scaled)

    # if coverage is < 10%, fill with Bidar typical value (alkaline Vertisols)
    coverage = np.count_nonzero(~np.isnan(out)) / out.size
    if coverage < 0.10:
        out = np.where(np.isnan(out), 7.6, out)
        if verbose:
            print("    Soil pH: low coverage — filled with Bidar typical value (7.6, Vertisols)")
    else:
        out = np.where(np.isnan(out), np.nanmean(out), out)

    if verbose:
        print(f"    Soil pH range: [{np.nanmin(out):.2f}, {np.nanmax(out):.2f}]")
    return out.astype(np.float32)


# ── Rainfall (CHIRPS) ─────────────────────────────────────────────────

def load_rainfall(verbose=True):
    """
    Load 12 monthly CHIRPS .tif.gz rainfall files for 2023,
    decompress on-the-fly, and sum to annual total (mm/year).
    """
    gz_files = sorted(glob.glob(os.path.join(PATHS["rainfall_dir"], "*.tif.gz")))
    if verbose:
        print(f"  Found {len(gz_files)} CHIRPS monthly files")

    width, height, transform = _target_grid()
    annual = np.zeros((height, width), dtype=np.float32)

    for gz_path in gz_files:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        with gzip.open(gz_path, 'rb') as f_in:
            with open(tmp_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        out = np.full((height, width), np.nan, dtype=np.float32)
        with rasterio.open(tmp_path) as src:
            reproject(
                source        = rasterio.band(src, 1),
                destination   = out,
                src_transform = src.transform,
                src_crs       = src.crs,
                dst_transform = transform,
                dst_crs       = CRS.from_epsg(4326),
                resampling    = Resampling.bilinear,
                dst_nodata    = np.nan,
            )
        os.unlink(tmp_path)
        out[out < -100] = np.nan          # CHIRPS nodata = -9999
        annual += np.nan_to_num(out, nan=0.0)

    annual[annual <= 0] = np.nan
    if verbose:
        print(f"    Annual rainfall: [{np.nanmin(annual):.0f} mm, {np.nanmax(annual):.0f} mm]")
    return annual.astype(np.float32)


# ── Soil Water Index (Validation) ────────────────────────────────────

def load_swi(target_shape, verbose=True):
    """
    Load Copernicus SWI for validation.
    The file lacks embedded georeferencing, so we resize it to match
    the study area grid and treat it as a relative moisture indicator (0-1).
    """
    with rasterio.open(PATHS["swi"]) as src:
        raw = src.read(1).astype(np.float32)

    normalized = raw / 255.0
    normalized[normalized <= 0] = np.nan

    h_t, w_t = target_shape
    swi = zoom(normalized, (h_t / raw.shape[0], w_t / raw.shape[1]), order=1)
    swi = np.clip(swi, 0, 1)

    if verbose:
        print(f"    SWI range: [{np.nanmin(swi):.3f}, {np.nanmax(swi):.3f}] "
              f"(resized to {swi.shape})")
    return swi.astype(np.float32)


# ── Master Loader ─────────────────────────────────────────────────────

def load_all_real_layers(verbose=True):
    """
    Load all real datasets onto a common grid (Bidar, EPSG:4326, ~250m).
    Returns a dict of numpy arrays keyed by layer name.
    """
    print("=" * 60)
    print("  Loading Real GIS Datasets — Bidar, Karnataka, India")
    print("=" * 60)

    layers = {}

    print("\n  [1/6] NDVI  — MODIS MOD13Q1, 250m, annual mean 2023")
    layers["ndvi"] = load_ndvi_clean(verbose)

    print("\n  [2/6] LST   — MODIS MOD11A2, 1km, annual mean 2023")
    layers["lst"] = load_lst_clean(verbose)

    print("\n  [3/6] DEM   — SRTM GL1, 30m resampled to 250m")
    layers["dem"] = load_dem(verbose)

    print("\n  [4/6] Soil pH — SoilGrids, pH-water 5-15cm, 250m")
    layers["soil_ph"] = load_soil_ph(verbose)

    print("\n  [5/6] Rainfall — CHIRPS 2023, monthly sum")
    layers["rainfall"] = load_rainfall(verbose)

    print("\n  [6/6] Slope — derived from SRTM DEM")
    dy, dx = np.gradient(np.nan_to_num(layers["dem"], nan=0.0), 250.0)
    layers["slope"] = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)
    print(f"    Slope range: [{layers['slope'].min():.2f}°, {layers['slope'].max():.2f}°]")

    print("\n  [Validation] SWI — Copernicus Soil Water Index, 1km")
    layers["swi"] = load_swi(layers["ndvi"].shape, verbose)

    print(f"\n  Grid shape: {layers['ndvi'].shape}  |  "
          f"Coverage: {TARGET_BOUNDS}  |  "
          f"Resolution: ~{TARGET_RES*111000:.0f} m\n")

    return layers
