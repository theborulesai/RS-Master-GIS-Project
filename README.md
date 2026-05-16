# Soil Moisture Proxy Mapping and Crop Suitability Analysis Using Remote Sensing and Machine Learning

**Course:** Remote Sensing and GIS   
**Study Area:** Bidar District, Karnataka, India  
**Year of Study:** 2023  
**Author:** Sainath

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Objectives](#2-objectives)
3. [Study Area](#3-study-area)
4. [Datasets Used](#4-datasets-used)
5. [How I Acquired the Data](#5-how-i-acquired-the-data)
6. [Methodology](#6-methodology)
7. [Software and Tools](#7-software-and-tools)
8. [How to Run the Code](#8-how-to-run-the-code)
9. [Project Structure](#9-project-structure)
10. [Results and Discussion](#10-results-and-discussion)
11. [Limitations](#11-limitations)
12. [Conclusion](#12-conclusion)
13. [References](#13-references)

---

## 1. Introduction

Soil moisture is one of the most critical variables in agriculture. It directly controls seed germination, root development, nutrient transport, and ultimately crop yield. However, measuring soil moisture across large areas is expensive and impractical with field sensors alone. Remote sensing offers a scalable alternative by deriving soil moisture proxies from freely available satellite imagery.

This project develops a complete geospatial analysis pipeline that:
- Estimates soil moisture spatially using the **Temperature Vegetation Dryness Index (TVDI)** derived from MODIS NDVI and Land Surface Temperature (LST) data.
- Incorporates terrain-based moisture indicators through the **Topographic Wetness Index (TWI)** computed from SRTM Digital Elevation Model data.
- Evaluates crop suitability for **Maize (Zea mays)** using a **Multi-Criteria Evaluation (MCE)** framework based on FAO land evaluation guidelines.
- Validates the results using a **Random Forest machine learning classifier** trained on real Copernicus Soil Water Index (SWI) ground-truth data.

The entire workflow uses **real satellite datasets** downloaded from NASA, USGS, ISRIC, and Copernicus platforms — no synthetic data is used anywhere in this project.

---

## 2. Objectives

1. To acquire and preprocess multi-source remote sensing data (MODIS NDVI, MODIS LST, SRTM DEM, CHIRPS rainfall, SoilGrids pH) for Bidar district.
2. To compute a soil moisture proxy map using the TVDI triangle method and TWI.
3. To perform Multi-Criteria Evaluation (MCE) for Maize crop suitability using Weighted Linear Combination (WLC).
4. To train and evaluate a Random Forest classifier for automated suitability prediction.
5. To generate thematic maps and accuracy assessment reports.

---

## 3. Study Area

**Bidar District** is located in the northernmost part of Karnataka state, India. It lies between latitudes 17.6°N to 18.3°N and longitudes 76.8°E to 77.6°E. The district is characterised by:

- **Terrain:** Deccan Plateau region with elevations ranging from 451m to 613m above sea level (flat to gently undulating terrain).
- **Climate:** Semi-arid tropical, with annual rainfall of approximately 734–1037 mm (data from CHIRPS 2023).
- **Soils:** Predominantly **Vertisols** (black cotton soil) with alkaline pH around 7.5–8.0. These soils have high clay content and are known for their capacity to swell when wet and crack when dry.
- **Agriculture:** Major crops include Jowar (Sorghum), Maize, Tur (Pigeon Pea), and Sunflower. Agriculture is largely rain-fed.

**Bounding Box Used:**
| Parameter | Value |
|-----------|-------|
| West | 76.800°E |
| East | 77.600°E |
| South | 17.600°N |
| North | 18.300°N |
| CRS | EPSG:4326 (WGS84) |
| Resolution | ~250 m |

---

## 4. Datasets Used

All datasets are real, downloaded manually from open-access satellite data portals. They are stored inside the project under `data/raw/`.

| Dataset | Product | Source | Year | Native Resolution | Files |
|---------|---------|--------|------|-------------------|-------|
| NDVI | MOD13Q1 v061 (16-day composite) | NASA AppEEARS | 2023 | 250 m | 23 GeoTIFFs + 23 QA |
| LST | MOD11A2 v061 (8-day composite) | NASA AppEEARS | 2023 | 1 km | 45 GeoTIFFs |
| DEM | SRTM GL1 (1 arc-second) | OpenTopography | Static | 30 m | 1 GeoTIFF |
| Rainfall | CHIRPS v2.0 (monthly) | UCSB CHRS | 2023 | ~5 km | 12 .tif.gz files |
| Soil pH | SoilGrids (pH-water, 5–15 cm) | ISRIC | 2020 | 250 m | 1 GeoTIFF |
| Soil Water Index | SWI (validation only) | Copernicus CLMS | 2025 | 1 km | 1 GeoTIFF |

**Total raw files:** 163+ files across all datasets.

---

## 5. 🛠️ How I Acquired the Data

All data was manually sourced and downloaded for the year 2023.

#### 5.1 Vegetation & Temperature (NASA AppEEARS)
* **Action Taken**: Submitted an extraction request titled "Soil_Moisture_Project_1" on the [AppEEARS](https://appeears.earthdatacloud.nasa.gov/) platform.
* **Data Specifics**: Requested 163 files covering the full year of 2023 for the Bidar bounding box.
* **The Result**: Received GeoTIFF files for NDVI (MOD13Q1, 250m resolution) to measure plant health, and LST (MOD11A2, 1km resolution) to measure surface heat.
* **Direct Access**: AppEEARS Download Area *(NASA Earthdata login required)*.

#### 5.2 Elevation Data (OpenTopography)
* **Action Taken**: Searched for "Bidar, India" in the [OpenTopography](https://opentopography.org/) portal and selected the SRTM GL1 (Global 30m) dataset.
* **Data Specifics**: Exported the spatial extent as a single GeoTIFF raster.
* **The Result**: A high-resolution Digital Elevation Model (DEM) used to calculate slope and model water drainage (TWI).

#### 5.3 Soil pH (ISRIC SoilGrids)
* **Action Taken**: Navigated the global soil map on the [SoilGrids Map Viewer](https://soilgrids.org/) and selected the "Chemical Soil" (flask icon) menu.
* **Data Specifics**: Filtered for "pH water" at a depth of 5–15 cm.
* **The Result**: A 250m resolution map detailing soil acidity/alkalinity of Bidar (mostly alkaline Vertisols).
* **Note**: SoilGrids stores pH as pH × 10 (integer), so values need to be divided by 10 in the code.

#### 5.4 Rainfall (UCSB CHIRPS)
* **Action Taken**: Accessed the [CHIRPS data repository](https://data.chc.ucsb.edu/products/CHIRPS-2.0/) and downloaded 12 monthly rainfall files for 2023.
* **Data Specifics**: Files are compressed as `.tif.gz` — my Python code automatically decompresses them during processing.
* **The Result**: Monthly precipitation totals, summed to get annual rainfall (~5km resolution).

#### 5.5 Validation Data (Copernicus CLMS)
* **Action Taken**: Searched for Bidar in the [Copernicus Global Land Service](https://land.copernicus.eu/) Data Viewer, selected "Bio-geophysical Variables," and checked "Soil Water Index".
* **Data Specifics**: Used the Export button (yellow/orange circle icon) to order the 1km version.
* **The Result**: Real-world moisture data used to validate my calculated proxy maps and train the Machine Learning model.

---

## 6. Methodology

### 6.1 Data Preprocessing
All raw satellite data was loaded using the `rasterio` library and aligned to a common spatial grid:
- **Reprojection:** MODIS data (Sinusoidal CRS) was reprojected to EPSG:4326 using `rasterio.warp.reproject()`.
- **Resampling:** All datasets were resampled to a uniform ~250m resolution grid covering the Bidar extent.
- **Scale Factors:** NDVI values multiplied by 0.0001; LST values multiplied by 0.02 K.
- **Quality Masking:** NDVI pixels with reliability flag > 1 (Snow/Ice or Cloudy) were masked as NaN.
- **Temporal Averaging:** Annual mean computed from 23 NDVI composites and 45 LST composites.
- **CHIRPS Decompression:** Monthly `.tif.gz` files decompressed on-the-fly and summed to annual rainfall (mm/year).

### 6.2 Topographic Wetness Index (TWI)
Computed from the SRTM DEM to identify areas where water naturally accumulates:

```
TWI = ln(a / tan(β))
```

Where:
- `a` = specific upstream catchment area (approximated using neighbourhood mean filter)
- `β` = local slope angle in radians (computed from `np.gradient`)

Higher TWI values indicate flatter terrain with larger contributing areas — these locations tend to be wetter.

### 6.3 Temperature Vegetation Dryness Index (TVDI)
Based on the LST/NDVI feature space triangle method (Sandholt et al., 2002):

```
TVDI = (LST_observed - LST_wet_edge) / (LST_dry_edge - LST_wet_edge)
```

The procedure:
1. NDVI values are binned into 50 equal intervals between 0.05 and 0.90.
2. For each bin, the 95th percentile LST (dry edge) and 5th percentile LST (wet edge) are extracted.
3. Linear regression is fitted to both edges as functions of NDVI.
4. TVDI is computed pixel-by-pixel. Values range from 0 (wet) to 1 (dry).

From my real data, the fitted edges were:
- **Dry edge:** LST_max = 306.7 + 5.49 × NDVI
- **Wet edge:** LST_min = 299.2 + 14.04 × NDVI

### 6.4 Composite Soil Moisture Index (SMI)
The TVDI and TWI were combined into a single normalised moisture proxy:

```
SMI = 0.6 × (1 - TVDI_normalised) + 0.4 × TWI_normalised
```

The `1 - TVDI` inversion is necessary because high TVDI indicates dryness, but we want high SMI to indicate wetness.

### 6.5 Multi-Criteria Evaluation (MCE)
Crop suitability for Maize was assessed using Weighted Linear Combination (WLC) with four criteria:

| Criterion | Weight | Rationale |
|-----------|--------|-----------|
| Soil Moisture (SMI) | 40% | Primary growth-limiting factor |
| Slope | 20% | Steep terrain increases erosion and prevents mechanisation |
| Rainfall | 20% | Maize requires 500–800 mm during the growing season |
| Soil pH | 20% | Maize prefers pH 5.8–7.0; Bidar soils are alkaline (~7.6) |

Each criterion was reclassified to a **1–5 suitability scale** using thresholds derived from FAO guidelines. The final suitability score is the weighted sum, classified into five categories: Not Suitable (S5), Marginally Suitable (S4), Moderately Suitable (S3), Suitable (S2), and Highly Suitable (S1).

### 6.6 Machine Learning Classification (3 Algorithms)
To move beyond rule-based MCE, I trained **3 different ML classifiers** and compared their performance:

- **Features (X):** NDVI, LST, TWI, Slope, Rainfall, Soil pH (6 features per pixel)
- **Labels (y):** Copernicus SWI discretised into 5 moisture classes
- **Train/Test Split:** 70/30 with stratified sampling (same split for all 3 models)

#### Algorithm 1: Random Forest
- 150 decision trees, max depth 12
- Uses `class_weight="balanced"` to handle imbalanced classes
- Provides feature importance scores

#### Algorithm 2: Support Vector Machine (SVM)
- RBF (Radial Basis Function) kernel
- Features standardised with `StandardScaler` (SVM is sensitive to feature magnitude)
- Subsampled to 50,000 points for training efficiency (O(n²) complexity)
- Uses `class_weight="balanced"`

#### Algorithm 3: Gradient Boosting
- 200 sequential estimators with learning rate 0.1
- Each tree corrects errors of the previous one
- Subsample rate of 0.8 to reduce overfitting
- Also provides feature importance scores

### 6.7 Accuracy Assessment
- **Overall Accuracy (OA):** Percentage of correctly classified pixels
- **Cohen's Kappa (κ):** Measures agreement above chance. Kappa > 0.6 = substantial agreement.
- **Confusion Matrix:** Shows per-class prediction errors
- **Model Comparison:** Side-by-side accuracy and kappa chart for all 3 algorithms

---

## 7. Software and Tools

| Tool | Purpose |
|------|---------|
| Python 3.x | All processing and analysis |
| rasterio | Reading, reprojecting, and writing GeoTIFF rasters |
| numpy / scipy | Numerical computation, gradient calculation, zoom/resize |
| scikit-learn | Random Forest classifier, train/test split, metrics |
| matplotlib | Map generation and confusion matrix plots |
| QGIS 3.x (optional) | Visual inspection of raw rasters |

---

## 8. How to Run the Code

```bash
# 1. Set up the virtual environment
cd ~/Documents/Master-GIS-Project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run the pipeline
python run_pipeline.py
```

The pipeline takes about **90–120 seconds** to complete (training 3 ML models) and outputs:
- 18 raster maps (`.png`) in `outputs/maps/`
- 3 text reports in `outputs/reports/`

---

## 9. Project Structure

```
Master-GIS-Project/
├── run_pipeline.py           # main script (runs everything)
├── config.py                 # study area, weights, thresholds
├── requirements.txt          # python dependencies
├── README.md                 # this file
├── .gitignore
│
├── src/                      # core analysis modules
│   ├── soil_moisture_proxy.py    # TVDI, TWI, SMI computation
│   ├── multi_criteria_eval.py    # MCE weighted overlay
│   ├── ml_training.py            # RF, SVM, Gradient Boosting
│   └── validation.py             # confusion matrix, kappa
│
├── utils/                    # helper modules
│   ├── real_data_loader.py       # loads all raw GeoTIFFs
│   ├── raster_utils.py           # normalise, reclassify
│   └── visualization.py          # map plotting functions
│
├── data/
│   ├── raw/                  # original downloaded datasets
│   │   ├── ndvi_lst/             # 163 MODIS files
│   │   ├── dem/                  # SRTM DEM
│   │   ├── rainfall/             # 12 CHIRPS .tif.gz
│   │   ├── soil_ph/              # SoilGrids pH
│   │   └── swi/                  # Copernicus SWI
│   └── processed/            # intermediate numpy arrays
│
└── outputs/
    ├── maps/                 # generated map images
    │   ├── real_ndvi.png
    │   ├── real_lst.png
    │   ├── real_dem.png
    │   ├── real_slope.png
    │   ├── real_rainfall.png
    │   ├── real_soil_ph.png
    │   ├── real_twi.png
    │   ├── real_tvdi.png
    │   ├── real_soil_moisture.png
    │   ├── mce_suitability_bidar.png
    │   ├── ml_suitability_rf.png         # Random Forest
    │   ├── ml_suitability_svm.png        # SVM
    │   ├── ml_suitability_gb.png         # Gradient Boosting
    │   ├── ml_confusion_rf.png
    │   ├── ml_confusion_svm.png
    │   ├── ml_confusion_gb.png
    │   ├── ml_comparison_chart.png       # accuracy/kappa comparison
    │   └── mce_vs_ml_matrix.png
    └── reports/
        ├── mce_report.txt
        ├── ml_comparison_report.txt      # 3-model comparison
        └── validation_report.txt
```

---

## 10. Results and Discussion

### 10.1 Satellite Data Summary
| Layer | Range | Interpretation |
|-------|-------|----------------|
| NDVI (annual mean) | -0.103 to 0.703 | Moderate vegetation cover; seasonal agriculture |
| LST (annual mean) | 299.0 K to 311.3 K (25.8°C to 38.1°C) | Typical semi-arid temperatures |
| DEM | 451 m to 613 m | Flat Deccan Plateau terrain |
| Slope | 0.0° to 57.4° | 95.5% of area is flat (<2°) |
| Annual Rainfall | 734 mm to 1037 mm | Moderate; suitable for rain-fed crops |
| Soil pH | ~7.6 (uniform) | Alkaline Vertisols, slightly above Maize optimum |

### 10.2 TVDI Results
- Dry edge: LST_max = 306.7 + 5.49 × NDVI
- Wet edge: LST_min = 299.2 + 14.04 × NDVI
- TVDI range: 0.000 to 1.000 (full range of moisture conditions present)

### 10.3 MCE Suitability Results
| Suitability Class | Area (%) |
|-------------------|----------|
| Marginally Suitable (S4) | 1.3% |
| Moderately Suitable (S3) | 38.1% |
| Suitable (S2) | 60.6% |
| Highly Suitable (S1) | 0.1% |

**Interpretation:** Over 60% of Bidar district is classified as **Suitable** for Maize cultivation. The limiting factor preventing higher scores is the alkaline soil pH (7.6), which is slightly above the optimal range for Maize (5.8–7.0). The flat terrain and adequate rainfall are favourable.

### 10.4 Machine Learning Results (3 Algorithms Compared)

All 3 models were trained on the same 70/30 stratified train/test split.

| Algorithm | Overall Accuracy | Cohen's Kappa |
|-----------|-----------------|---------------|
| Random Forest | ~95% | ~0.17 |
| SVM (RBF) | ~95% | ~0.17 |
| Gradient Boosting | ~95% | ~0.17 |

*(Exact values are printed when the pipeline runs — they may vary slightly.)*

**Random Forest Feature Importances:**
| Feature | Importance |
|---------|------------|
| Rainfall | ~0.30 |
| LST | ~0.23 |
| NDVI | ~0.22 |
| TWI | ~0.14 |
| Slope | ~0.11 |
| Soil pH | ~0.00 |

**Key Observation:** All 3 models achieve similar high accuracy (~95%) but low Kappa (~0.17). This is because of the severe class imbalance — 98% of pixels fall into Class 5 (Highly Suitable) using SWI as ground truth, since Bidar's SWI values are uniformly high. All models correctly predict the dominant class but struggle with the rare minority classes. This is a data distribution limitation, not an algorithm limitation.

The comparison chart (`ml_comparison_chart.png`) visually shows the performance of all 3 algorithms side by side.

---

## 11. Limitations

1. **SWI Ground Truth:** The Copernicus SWI data lacks georeferencing metadata, so it was resized to fit the study area rather than properly georeferenced. This introduces spatial uncertainty.
2. **Soil pH Coverage:** The ISRIC SoilGrids data had limited spatial coverage over Bidar, so gaps were filled with the regional average (7.6). In reality, pH may vary locally.
3. **Simplified TWI:** The flow accumulation was approximated using a neighbourhood mean filter instead of proper D8 flow routing. This could be improved using SAGA GIS or GRASS.
4. **Class Imbalance:** The ML model suffers from extreme class imbalance (98% Class 5). Techniques like SMOTE or collecting more diverse training data could improve minority class performance.
5. **Temporal Mismatch:** MODIS data is from 2023 while SWI validation data is from 2025. Ideally both should be from the same year.

---

## 12. Conclusion

This project successfully demonstrated a complete remote sensing workflow for soil moisture proxy mapping and crop suitability analysis. Using real satellite data from multiple sources (NASA, OpenTopography, SoilGrids, CHIRPS, Copernicus), I was able to:

- Generate spatially continuous soil moisture proxy maps for Bidar district using the TVDI approach.
- Classify over 60% of the district as suitable for Maize cultivation using multi-criteria evaluation.
- Train a Random Forest model achieving 95% overall accuracy for automated suitability prediction.

The combination of traditional GIS methods (MCE) with machine learning provides complementary perspectives — MCE offers interpretability through explicit weights, while Random Forest captures non-linear relationships in the data.

---

## 13. References

1. Sandholt, I., Rasmussen, K., & Andersen, J. (2002). A simple interpretation of the surface temperature/vegetation index space for assessment of surface moisture status. *Remote Sensing of Environment*, 79(2-3), 213-224.
2. Beven, K.J. & Kirkby, M.J. (1979). A physically based, variable contributing area model of basin hydrology. *Hydrological Sciences Bulletin*, 24(1), 43-69.
3. FAO (1976). A Framework for Land Evaluation. *FAO Soils Bulletin No. 32*, Rome.
4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
5. Congalton, R.G. (1991). A review of assessing the accuracy of classifications of remotely sensed data. *Remote Sensing of Environment*, 37(1), 35-46.
6. Landis, J.R. & Koch, G.G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159-174.
7. Didan, K. (2021). MODIS/Terra Vegetation Indices 16-Day L3 Global 250m SIN Grid V061. NASA EOSDIS Land Processes DAAC.
8. Wan, Z., Hook, S., & Hulley, G. (2021). MODIS/Terra Land Surface Temperature/Emissivity 8-Day L3 Global 1km SIN Grid V061. NASA EOSDIS Land Processes DAAC.
9. Funk, C., et al. (2015). The climate hazards infrared precipitation with stations — a new environmental record for monitoring extremes. *Scientific Data*, 2, 150066.
10. Hengl, T., et al. (2017). SoilGrids250m: Global gridded soil information based on machine learning. *PLOS ONE*, 12(2), e0169748.
