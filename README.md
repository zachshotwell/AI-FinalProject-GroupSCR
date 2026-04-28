# AI Final Project

**Members:** Alan Ray · Zach Shotwell · Miguel Moreno Coin

**Course:** Artificial Intelligence, Appalachian State University

## Projects

### Runoff Forecasting

Deep learning post-processor (GRU / LSTM simple / stacked LSTM) for NOAA National Water Model short-range streamflow forecasts at two USGS stations. The model predicts the residual error `USGS_obs − NWM_forecast` at each of 18 lead times; the corrected forecast is reconstructed as `NWM + predicted_error`.

Results differ sharply between the two stations. At the well-calibrated station (large California river), the post-processor cuts test RMSE in half at short leads and holds R² ≈ 0.99 across all 18 lead times, including the December 2022 / January 2023 atmospheric river events. At the second station (lower Gila River near Dome, AZ), it removes most of the mean bias but does not demonstrate dynamic forecasting skill — the gauge sits on a heavily regulated reach where NWM's natural-flow simulation and the observed managed flow are mechanically decoupled by upstream dams and irrigation diversions.

See `RunoffForecastingProject/` for the notebook and methodology writeup.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zachshotwell/AI-FinalProject-GroupSCR/blob/main/RunoffForecastingProject/RunoffForecasting.ipynb)

### DoriaNET

CNN-based image classification of post-hurricane building damage (FEMA HAZUS states 0–5) from UAV aerial imagery.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zachshotwell/AI-FinalProject-GroupSCR/blob/main/DoriaNETProject/DoriaNET.ipynb)
