# Code and Data Description

This repository contains the scripts and data used to calculate the Defense-Repair-Adjustment mechanism scores, spectral endpoint, Stem Stress Index, joint latent alignment, robustness analyses, and Cd prediction model described in the study.

## 1. Overview

The code supports three main analytical steps:
1. Construction of the Defense-Repair-Adjustment mechanism scores from metabolite data.
2. Extraction of the mechanism-associated optical readout, defined as Endpoint S, from the trained spectral model.
3. Cross-domain alignment between the biological mechanism structure and tissue-scale spectral information, including SSI construction, JLM analysis, leave-one-out sensitivity analysis, and robustness testing.

The complete workflow links metabolomic mechanism scores, hyperspectral data, and model-derived optical readouts to evaluate whether the internal stem Cd stress mechanism can be traced by tissue-scale optical signals.

## 2. File structure

### 2.1 Code files

| Placeholder file name | Description |
|---|---|
| `3modules_score.py` | Calculates the three mechanism scores for Defense, Repair, and Adjustment from the curated metabolite sets. |
| `3modules_score_robustness.py` | Calculates the three mechanism scores for robustness analysis using alternative or expanded metabolite sets. |
| `JLM.py` | Performs Joint Latent Model analysis to evaluate the alignment between spectral variation and the Defense-Repair-Adjustment mechanism score structure. |
| `Preprocessing_package.py` | Contains preprocessing functions used for spectral data processing and model input preparation. |
| `best_cnn.h5` | Stores the optimized parameters of the best trained prediction model. |
| `deleteone.py` | Performs leave-one-out sensitivity analysis for the association between Endpoint S and SSI. |
| `endpoint.py` | Calculates Endpoint S and related endpoint outputs from the trained spectral model. |
| `prediction.py` | Trains the spectral prediction model for stem Cd concentration. |
| `ssi.py` | Calculates the Stem Stress Index based on the three mechanism scores. |

### 2.2 Data files

| Placeholder file name | Description |
|---|---|
| `stem_data.csv` | Full hyperspectral dataset used for spectral model training and prediction. |
| `data.csv` | Hyperspectral data of the 24 samples used for mechanism-optical alignment analysis. |
| `meta.csv` | Metabolite abundance data of the 24 samples used for Defense-Repair-Adjustment score calculation. |
