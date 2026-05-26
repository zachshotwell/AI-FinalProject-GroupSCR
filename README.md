# AI Final Project - Group 9
**Members:** Alan Ray · Zach Shotwell · Miguel Moreno Coin

**Course:** Artificial Intelligence (CS 4440-101), Appalachian State University

Two end-to-end deep learning pipelines submitted as the final project: bias correction of streamflow forecasts, and building damage classification from UAV aerial imagery. The repository contains both notebooks, the paired papers each project replicates, and the data needed to reproduce the headline results from a clean Colab runtime.

## Runoff Forecasting

A deep learning post-processor for NOAA National Water Model (NWM) short-range runoff forecasts. The model predicts the residual error `USGS_obs − NWM_forecast` over an 18-hour horizon, using the previous 6 hours of observed errors, NWM forecasts, and USGS observations as input. The corrected forecast is reconstructed as `NWM + predicted_error`. Three recurrent architectures are trained per station - GRU, simple LSTM, stacked LSTM - and compared against the raw NWM forecast and a persistence baseline using the standard hydrologic metric panel (CC, NSE, PBIAS, RMSE) at each lead time 1–18 h.

The two stations behave very differently. At USGS 11266500 (San Joaquin River, CA; NHD reach 21609641), where NWM is already well-calibrated, the GRU halves test RMSE at lead = 1 h (1.73 → 0.86 cms) and tracks the 100+ cms atmospheric river peaks of December 2022 / January 2023 with R² = 0.998 - even though no comparable peaks are present in the validation period. At USGS 09520500 (lower Gila River near Dome, AZ; reach 20380357), NWM overforecasts by roughly 16× because the gauge sits below Painted Rock Dam and the Ashurst–Hayden irrigation diversion, neither of which is represented in NWM's natural-flow simulation. The post-processor cleanly removes the mean bias and cuts RMSE roughly in half, but Pearson CC stays near zero: there is no dynamic skill to recover when the input forecast and the observed flow are physically decoupled by upstream regulation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zachshotwell/AI-FinalProject-GroupSCR/blob/main/RunoffForecastingProject/RunoffForecasting.ipynb)

## DoriaNET

Six-class FEMA HAZUS damage state classification of post-Hurricane-Dorian buildings from UAV aerial imagery, using the DoriaNET dataset (Cheng, Behzadan & Noshadravan 2021): 271 video frames from Marsh Harbour, Bahamas, yielding 1,458 (frame, building) instances across 147 distinct physical buildings. Six architectures are benchmarked - an MLP baseline, a from-scratch CNN, modernized versions of each (MLP v2, CNN v2) applying Géron (2022) Ch. 11 training improvements, MobileNetV2 with two-step transfer learning, and MobileNetV1 (the paper's chosen backbone) under the same recipe.

The split is building-stratified: all frame appearances of a given physical building stay in the same partition. A row-level shuffle would put the same building into all three splits at slightly different angles, so the model could memorize building-specific texture rather than learn damage patterns. Our absolute numbers therefore sit closer to Cheng et al.'s unseen-data generalization results (Dataset 2: 30% / 64%) than to their within-split numbers (Dataset 1: 61% / 90%) - the expected consequence of removing the leakage.

The MLP and from-scratch CNN both collapse to majority-class prediction (31.1% test accuracy, equal to the class-3 share in the test set) - a clean reproduction of the data-scarcity regime that motivates transfer learning, given only ~960 training instances spread across six ordinal classes. The modernized v2 variants partially break the collapse but do not beat the majority baseline on exact accuracy. MobileNetV2 reaches 34% exact / 67% ±1-class accuracy. MobileNetV1 wins outright at 47% exact / 82% ±1-class accuracy, leveraging ImageNet-pretrained features under the same two-step fine-tune recipe Cheng et al. published. An end-to-end pipeline couples a YOLO11n building detector to MobileNetV1, evaluating the full frame-in / damage-out flow on held-out test frames.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zachshotwell/AI-FinalProject-GroupSCR/blob/main/DoriaNETProject/DoriaNET.ipynb)
