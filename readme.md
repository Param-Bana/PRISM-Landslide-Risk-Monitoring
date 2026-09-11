# PRISM — AI-Based Landslide Risk Monitoring System

**PRISM (Predictive Risk Intelligence & Slope Monitoring)** is an AI-based landslide risk monitoring and early-warning prototype designed for the **Northeastern Region (NER) of India**.

The system combines terrain, rainfall, environmental and geological information with machine-learning models to estimate landslide susceptibility and support risk-aware decision making.

---

## Problem Statement

The Northeastern Region of India is highly vulnerable to landslides due to:

- Heavy and prolonged rainfall
- Steep and fragile terrain
- Soil and geological instability
- Deforestation and land-use changes
- Unplanned hill cutting and infrastructure development
- Road and settlement exposure

PRISM aims to provide an AI-assisted platform for identifying areas with elevated landslide susceptibility and supporting early intervention.

---

## Current Prototype

The current deployed prototype uses a **Random Forest Classifier** to estimate landslide susceptibility from environmental and terrain features.

### Model Inputs

#### Terrain
- Elevation
- Slope
- Aspect
- Curvature
- Distance to river

#### Environmental
- Rainfall over 24 hours
- Rainfall over 72 hours
- Rainfall over 7 days
- NDVI

#### Categorical
- Soil type
- Land cover / terrain
- State

The prototype currently covers the eight Northeastern states:

- Arunachal Pradesh
- Assam
- Manipur
- Meghalaya
- Mizoram
- Nagaland
- Sikkim
- Tripura

---

## Machine Learning Model

**Algorithm:** Random Forest Classifier

Current prototype configuration:

- 500 decision trees
- Balanced class weighting
- Minimum samples per leaf: 2
- Stratified train/test split
- Scikit-learn 1.9.0

### Prototype Evaluation

The current random holdout evaluation produced:

| Metric | Result |
|---|---:|
| Accuracy | **97.08%** |
| Precision | **90.77%** |
| Recall | **98.33%** |
| F1 Score | **94.40%** |
| ROC-AUC | **99.53%** |
| Test Samples | **1,200** |

> **Important:** These are prototype random-holdout results and should not be interpreted as operationally validated landslide prediction accuracy for the entire Northeastern Region. Further temporal, spatial and event-based validation is required.

---

## Risk Classification

The model score is mapped to four prototype risk levels:

| Model Score | Risk Level |
|---:|---|
| < 25% | 🟢 GREEN — Low |
| 25–50% | 🟡 YELLOW — Moderate |
| 50–75% | 🟠 ORANGE — High |
| ≥ 75% | 🔴 RED — Very High |

The displayed score is currently a **prototype model score**, not a calibrated operational warning probability.

---

## System Architecture

```text
Terrain & Environmental Data
            │
            ▼
     Data Preprocessing
            │
            ▼
     Machine Learning Model
       Random Forest
            │
            ▼
      Risk Score Generation
            │
            ▼
   GREEN / YELLOW / ORANGE / RED
            │
            ▼
       PRISM Dashboard
