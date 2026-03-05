# 🚗 Predicting Ride-Sharing Prices in Boston

A Harvard APCOMP209A final project investigating the key drivers of Uber and Lyft pricing across the Boston metropolitan area using machine learning and statistical modeling.

**Authors:** Xiaotong (Cindy) Sun · Junzhi (Molly) Han · Hanzhen (Jenny) Zhu · Xiaoman (Nicole) Xu  
**Course:** APCOMP209A — Introduction to Data Science, Harvard University, Fall 2025  
**Instructors:** Pavlos Protopapas and Kevin Rader  
**Video Presentation:** [Watch on Google Drive](https://drive.google.com/file/d/1sP1VH12GFEORt9SywnqHCbP_XPtbo8uG/view?usp=drive_link)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Research Question](#research-question)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)

---

## Overview

This project builds predictive models for ride-hailing fares using a large-scale dataset of trip-level observations from Boston. While distance is theoretically the primary determinant of ride cost, real-world pricing is also shaped by dynamic surge behavior, service tier, platform differences, temporal patterns, spatial context, and weather conditions.

We adopt a structured modeling strategy progressing from simple to flexible frameworks, evaluating performance using R² and RMSE across consistent train/validation/test splits.

---

## Research Question

> *To what extent can ride-hailing prices in Boston be predicted using trip distance alone, compared with models that incorporate surge pricing, weather conditions, temporal factors, spatial context, and service platform?*

---

## Dataset

**Source:** [Uber and Lyft Dataset — Boston, MA (Kaggle)](https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma/data)

Download `rideshare_kaggle.csv` and place it in a `data/` directory at the project root.

| Property | Value |
|---|---|
| Total observations | 693,071 |
| Features used | 16 (after engineering: 31) |
| Missing prices | 55,095 rows (dropped for modeling) |
| Coverage | November – December (Boston, MA) |

### Feature Summary

**Numerical:** `distance`, `surge_multiplier`, `temperature`, `precipIntensity`, `precipProbability`, `cloudCover`

**Cyclical (sine-cosine encoded):** `hour` → `hour_sin`, `hour_cos` · `day_of_month` → `day_sin`, `day_cos`

**Categorical:**
- Service: `name` (UberX, Lyft Lux, etc.), `cab_type` (Uber/Lyft)
- Temporal: `month_name`, `day_of_week`
- Spatial: `source`, `destination` (12 Boston neighborhoods each)

---

## Repository Structure

```
.
├── data/
│   └── rideshare_kaggle.csv          # Download from Kaggle (see link above)
├── DS209A_final_project_group86-FINAL.ipynb   # Main notebook
├── final_PDF.pdf                     # Exported PDF report
└── README.md
```

---

## Setup & Installation

### Requirements

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Install Dependencies

All required packages are auto-installed at the top of the notebook. You can also install them manually:

```bash
pip install numpy pandas matplotlib seaborn statsmodels scikit-learn category_encoders
```

### Run the Notebook

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Download the dataset from Kaggle and place it in data/
mkdir data
# Move rideshare_kaggle.csv into data/

jupyter notebook DS209A_final_project_group86-FINAL.ipynb
```

---

## Methodology

### 1. Data Cleaning
- Selected 16 relevant columns; parsed timestamps into `month_name`, `day_of_week`, `day_of_month`
- Dropped 55,095 rows with missing `price` values; confirmed zero duplicates

### 2. Exploratory Data Analysis (EDA)
- **Outcome:** Log-transformed `price` exhibits a more symmetric, near-normal distribution
- **Categorical:** `name` and `cab_type` show the strongest price variation; temporal categories show weaker effects
- **Numerical:** `distance` and `surge_multiplier` are the only predictors with clear positive linear associations with price
- **Correlation:** `precipIntensity` and `precipProbability` are highly collinear (r = 0.84)

### 3. Feature Engineering

| Technique | Applied To | Reason |
|---|---|---|
| Sine-cosine (cyclical) encoding | `hour`, `day_of_month` | Preserves periodic structure; no leakage risk |
| Target encoding | `source`, `destination` | Compresses 24 sparse dummies into 2 continuous spatial price signals |
| One-hot encoding | `name`, `cab_type`, `month_name`, `day_of_week` | Preserves discrete service/temporal structure |
| Interaction terms | `distance × name`, `distance × cab_type` | Captures service-specific per-mile pricing differences |

All encoding (except cyclical) was fit **exclusively on training data** to prevent leakage.

**Split:** 60% train · 20% validation · 20% test (stratified, `random_state=42`)

### 4. Models

| Model | Description |
|---|---|
| **Baseline** | Simple linear regression: `log(price) ~ distance` |
| **MLR (LASSO-reduced)** | Multiple linear regression with 33 LASSO-selected features + interaction terms |
| **Decision Tree** | `DecisionTreeRegressor` with 5-fold CV hyperparameter tuning |
| **Polynomial MLR** | MLR extended with `distance²` and `distance³` terms |

---

## Results

| Model | Train R² | Val R² | Test R² | Test RMSE |
|---|---|---|---|---|
| Baseline (distance only) | 0.1135 | 0.1169 | 0.1149 | 0.5352 |
| MLR (LASSO-reduced) | 0.9421 | 0.9425 | **0.9427** | 0.1362 |
| Decision Tree | 0.9557 | 0.9476 | **0.9483** | **0.1293** |
| Polynomial-Enhanced MLR | 0.9439 | 0.9443 | 0.9445 | 0.1340 |

All models show near-zero train/validation gaps, indicating strong generalization with no meaningful overfitting.

---

## Key Findings

- **Distance alone is a poor predictor** — the baseline model explains only ~11% of fare variation (R² ≈ 0.11)
- **Service type and surge multiplier dominate pricing** — `name_Shared`, `name_Black SUV`, `name_Lux Black XL`, and `surge_multiplier` are the most influential features across all models
- **The distance–price relationship is largely linear** — adding polynomial distance terms (degree 2 and 3) yields no meaningful improvement once other features are included
- **The Decision Tree achieves the best test performance** (R² = 0.9483, RMSE = 0.1293), slightly outperforming the linear models by automatically capturing nonlinear service-price thresholds
- **Weather variables contribute minimally** — `precipIntensity`, `precipProbability`, and `cloudCover` were among the features eliminated by LASSO or ranked lowest in decision tree importance
- **Spatial effects are meaningful but secondary** — target-encoded `source` and `destination` provide consistent predictive signal, with Boston University and Fenway commanding the highest expected fares
