import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import date, timedelta

st.set_page_config(page_title="EGX Stock Direction Predictor", page_icon="\U0001F4C8", layout="centered")

st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {background-color: #f5f7fa; border-radius: 10px; padding: 12px;}
    </style>
""", unsafe_allow_html=True)

FORECAST_HORIZON = 60  # trading days ahead the model predicts

@st.cache_resource
def load_pipeline():
    return joblib.load("xgb_model_pipeline.pkl")

pipeline = load_pipeline()
FEATURE_ORDER = list(pipeline.named_steps["scaler"].feature_names_in_)

TICKERS = [
    "ABUK.CA", "ADIB.CA", "ALCN.CA", "CANA.CA", "CIEB.CA", "COMI.CA", "EAST.CA",
    "EFID.CA", "EFIH.CA", "EGAL.CA", "EMFD.CA", "ETEL.CA", "FWRY.CA", "GBCO.CA",
    "GPPL.CA", "HDBK.CA", "HELI.CA", "HRHO.CA", "JUFO.CA", "MFPC.CA", "OCDI.CA",
    "ORAS.CA", "ORHD.CA", "PHDC.CA", "SCTS.CA", "SWDY.CA", "TMGH.CA",
]


def create_features(df):
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    df["usd_return_1d"] = df["USD_EGP"].pct_change(1)
    df["usd_return_5d"] = df["USD_EGP"].pct_change(5)
    df["usd_return_20d"] = df["USD_EGP"].pct_change(20)

    sma_20 = df["Close"].rolling(window=20).mean()
    sma_50 = df["Close"].rolling(window=50).mean()
    sma_100 = df["Close"].rolling(window=100).mean()

    df["dist_sma_20"] = (df["Close"] / sma_20) - 1
    df["dist_sma_50"] = (df["Close"] / sma_50) - 1
    df["dist_sma_100"] = (df["Close"] / sma_100) - 1

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    df["macd_pct"] = macd / df["Close"]
    df["macd_signal_pct"] = df["macd_pct"].ewm(span=9, adjust=False).mean()

    rolling_std = df["Close"].rolling(window=20).std()
    bollinger_high = sma_20 + (rolling_std * 2)
    bollinger_low = sma_20 - (rolling_std * 2)
    df["bollinger_pb"] = (df["Close"] - bollinger_low) / (bollinger_high - bollinger_low + 1e-8)

    df["volatility_20d"] = df["return_1d"].rolling(window=20).std()
    vol_sma_20 = df["Volume"].rolling(window=20).mean()
    df["vol_ratio"] = df["Volume"] / (vol_sma_20 + 1e-8)

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_features(ticker, as_of_date):
    lookback_start = as_of_date - timedelta(days=400)  # buffer for 100d SMA + weekends/holidays

    stock = yf.download(ticker, start=lookback_start.isoformat(), end=(as_of_date + timedelta(days=1)).isoformat(),
                         progress=False, auto_adjust=True)
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.droplevel(1)
    if stock.empty:
        return None, "No price data returned for this ticker/date from Yahoo Finance."

    usd_egp = yf.download("EGP=X", start=lookback_start.isoformat(), end=(as_of_date + timedelta(days=1)).isoformat(),
                           progress=False, auto_adjust=True)
    if isinstance(usd_egp.columns, pd.MultiIndex):
        usd_egp.columns = usd_egp.columns.droplevel(1)
    usd_egp = usd_egp[["Close"]].rename(columns={"Close": "USD_EGP"})

    stock = stock.merge(usd_egp, left_index=True, right_index=True, how="left")
    stock["USD_EGP"] = stock["USD_EGP"].ffill()
    stock = stock.dropna(subset=["USD_EGP"])

    stock = create_features(stock)
    stock = stock.dropna()

    if stock.empty:
        return None, "Not enough historical data before this date to compute all indicators (need ~100+ trading days)."

    last_row = stock.iloc[-1]
    actual_date = stock.index[-1].date()
    return last_row, actual_date


st.title("\U0001F4C8 EGX Stock Direction Predictor")
st.caption(f"Predicts price direction {FORECAST_HORIZON} trading days ahead for Egyptian Exchange stocks using a trained XGBoost pipeline.")

with st.sidebar:
    st.header("\u2699\ufe0f Settings")
    ticker = st.selectbox("Ticker", TICKERS, index=TICKERS.index("COMI.CA"))
    selected_date = st.date_input("Date", value=date.today() - timedelta(days=1), max_value=date.today())
    st.markdown("---")
    st.caption("Historical OHLCV & USD/EGP data is fetched automatically via yfinance.")

if st.button("\U0001F52E Predict", use_container_width=True, type="primary"):
    with st.spinner("Fetching market data and computing indicators..."):
        last_row, info = fetch_features(ticker, selected_date)

    if last_row is None:
        st.error(info)
    else:
        if info != selected_date:
            st.info(f"Using last available trading day: {info} (market closed on {selected_date}).")

        row = {feat: last_row[feat] for feat in [
            "Close", "High", "Low", "Open", "Volume", "USD_EGP",
            "return_1d", "return_5d", "return_20d",
            "usd_return_1d", "usd_return_5d", "usd_return_20d",
            "dist_sma_20", "dist_sma_50", "dist_sma_100",
            "rsi_14", "macd_pct", "macd_signal_pct", "bollinger_pb",
            "volatility_20d", "vol_ratio",
        ]}
        for t in TICKERS:
            row[f"ticker_{t}"] = 1.0 if t == ticker else 0.0

        X = pd.DataFrame([row])[FEATURE_ORDER]

        pred = pipeline.predict(X)[0]
        proba = pipeline.predict_proba(X)[0]

        st.subheader("Prediction Result")
        target_date = info + timedelta(days=int(FORECAST_HORIZON * 7 / 5))  # rough calendar estimate
        st.caption(f"Direction expected around {FORECAST_HORIZON} trading days after {info} (~{target_date}).")

        if pred == 1:
            st.success(f"\U0001F4C8 UP - {ticker} is predicted to be higher in {FORECAST_HORIZON} trading days")
        else:
            st.error(f"\U0001F4C9 DOWN - {ticker} is predicted to be lower in {FORECAST_HORIZON} trading days")

        p1, p2 = st.columns(2)
        p1.metric("Probability Down", f"{proba[0]:.1%}")
        p2.metric("Probability Up", f"{proba[1]:.1%}")
        st.progress(float(proba[1]), text="Confidence (Up)")

        with st.expander("Show computed features"):
            st.dataframe(X.T.rename(columns={0: "value"}))
