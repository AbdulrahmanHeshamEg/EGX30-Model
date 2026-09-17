# 🧭 Smart EGX Guide

An AI-powered market scanner that predicts the next-day price direction (up/down) for stocks listed on the Egyptian Exchange (EGX), using an XGBoost classifier trained on price, volume, and technical-indicator features.

## Overview

Smart EGX Guide scans a curated list of EGX-listed tickers and, for a selected date, predicts whether each stock is likely to close **higher** or **lower** the next trading day — along with the model's confidence for each prediction. Results are ranked and displayed in an interactive dashboard.

## Features

- One-click scan across ~27 major EGX tickers
- Automatic data fetch via `yfinance` (OHLCV + USD/EGP exchange rate)
- Technical indicators computed on the fly: RSI, MACD, Bollinger Bands, SMA distances, volatility, volume ratio, and multi-horizon returns
- Bullish / Bearish signal with a model confidence score
- Simple, interactive Streamlit dashboard

## Tech Stack

| Layer | Tool |
|---|---|
| Modeling | Python, scikit-learn, XGBoost |
| Data | yfinance |
| App / UI | Streamlit |
| Persistence | joblib |

## Model

- **Task:** binary classification — next-day direction (up = 1, down = 0)
- **Features:** OHLCV, USD/EGP, return over multiple horizons, RSI(14), MACD, Bollinger %B, SMA distances (20/50/100), volatility, volume ratio, and a one-hot ticker identity
- **Algorithm:** XGBoost (`XGBClassifier`) inside a scikit-learn pipeline (`StandardScaler` + model), selected after comparing Logistic Regression, KNN, Random Forest, and Gradient Boosting
- **Validation:** time-series cross-validation (no shuffling, no look-ahead)
- **Artifact:** the trained pipeline is saved as `xgb_model_pipeline.pkl` and loaded directly by the app

> Note: short-horizon direction prediction on individual equities is inherently noisy — the model's edge over a random baseline is modest. This tool is meant as a decision-support signal, not financial advice.

## Project Structure

```
├── EGX30_Model_final_final.ipynb   # Data collection, feature engineering, model training & evaluation
├── xgb_model_pipeline.pkl          # Trained model pipeline (StandardScaler + XGBoost)
├── app.py                          # Streamlit dashboard
└── requirements.txt                # Python dependencies
```

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app locally
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. Pick a date in the sidebar and click **Scan Market**.

## Deployment

The app is a standard Streamlit app, so it can be deployed with **Streamlit Community Cloud** in a few steps:

1. Push this repo to GitHub (include `app.py`, `xgb_model_pipeline.pkl`, and `requirements.txt`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub repo
3. Set `app.py` as the entry point and deploy

Any other platform that runs Streamlit apps (e.g., a basic VM or container with `streamlit run app.py`) works the same way.

## Disclaimer

This project is for educational and research purposes only. Predictions are not investment advice, and past performance does not guarantee future results.
