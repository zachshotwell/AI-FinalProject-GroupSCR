# AI Final Project

**Members:** Alan Ray · Zach Shotwell · Miguel Moreno Coin

**Course:** Artificial Intelligence, Appalachian State University

## Projects

### Runoff Forecasting

Deep learning post-processor (GRU / LSTM simple / stacked LSTM) for NOAA National Water Model short-range streamflow forecasts at two USGS stations. The model predicts the residual error `USGS_obs − NWM_forecast` at each of 18 lead times; the corrected forecast is reconstructed as `NWM + predicted_error`.

Results differ sharply between the two stations. At the well-calibrated station (large California river), the post-processor cuts test RMSE in half at short leads and holds R² ≈ 0.99 across all 18 lead times, including the December 2022 / January 2023 atmospheric river events. At the second station (lower Gila River near Dome, AZ), it removes most of the mean bias but does not demonstrate dynamic forecasting skill — the gauge sits on a heavily regulated reach where NWM's natural-flow simulation and the observed managed flow are mechanically decoupled by upstream dams and irrigation diversions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zachshotwell/AI-FinalProject-GroupSCR/blob/main/RunoffForecastingProject/RunoffForecasting.ipynb)

### DoriaNET

Image classification of post-hurricane building damage (FEMA HAZUS states 0–5) from UAV aerial imagery, using the DoriaNET dataset (Cheng, Behzadan & Noshadravan 2021): 271 frames from Marsh Harbor, Bahamas, after Hurricane Dorian, with 1,458 (frame, building) instances across 147 distinct buildings.

Three architectures are benchmarked along the standard course progression: an MLP baseline, a standard CNN trained from scratch, and a transfer-learning model based on MobileNetV2 with two-step fine-tuning. The MLP and from-scratch CNN both collapse to majority-class prediction (22% test accuracy) — a clean reproduction of the data-scarcity regime that motivates transfer learning, given only ~800 training instances spread across 6 classes. MobileNetV2 reaches 40% test accuracy and 69% ±1 accuracy by leveraging ImageNet-pretrained features. Our absolute numbers sit closer to the paper's *unseen-data* generalization results (Dataset 2: 30% / 64%) than to their *within-split* numbers (Dataset 1: 61% / 90%), reflecting our deliberate use of a building-stratified split rather than the paper's row-level split, which leaks the same building across train and test.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zachshotwell/AI-FinalProject-GroupSCR/blob/main/DoriaNETProject/DoriaNET.ipynb)
