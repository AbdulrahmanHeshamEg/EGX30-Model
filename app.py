import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import timedelta, date

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smart EGX Guide",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #16213e 0%, #0b0c1a 45%, #0b0c1a 100%);
    }

    .main .block-container {
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        max-width: 1100px;
    }

    /* Hero header */
    .hero-wrap {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    .hero-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(56,189,248,0.15), rgba(167,139,250,0.15));
        border: 1px solid rgba(148,163,184,0.25);
        color: #a5b4fc;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f8fafc 20%, #93c5fd 60%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.1;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.02rem;
        max-width: 560px;
        margin: 0.9rem auto 0 auto;
        line-height: 1.6;
    }

    /* Centered stat cards */
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 1.25rem;
        flex-wrap: wrap;
        margin: 2.2rem 0 2.6rem 0;
    }
    .stat-card {
        flex: 0 1 220px;
        text-align: center;
        padding: 1.6rem 1.2rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(148,163,184,0.15);
        backdrop-filter: blur(6px);
    }
    .stat-label {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.5rem;
    }
    .stat-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .stat-bullish .stat-value { color: #4ade80; }
    .stat-bearish .stat-value { color: #f87171; }
    .stat-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.3rem;
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #f1f5f9;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .section-sub {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    section[data-testid="stSidebar"] {
        background: #0f1226;
        border-right: 1px solid rgba(148,163,184,0.12);
    }

    .stButton > button {
        font-weight: 600;
        border-radius: 10px;
        border: none;
        background: linear-gradient(90deg, #38bdf8, #a78bfa);
        color: #0b0c1a;
    }
    .stButton > button:hover {
        opacity: 0.9;
        color: #0b0c1a;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,0.15);
    }
    </style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
EGX_TICKERS = [
    "ABUK.CA", "ADIB.CA", "ALCN.CA", "CANA.CA", "CIEB.CA", "COMI.CA",
    "EAST.CA", "EFID.CA", "EFIH.CA", "EGAL.CA", "EMFD.CA", "ETEL.CA",
    "FWRY.CA", "GBCO.CA", "GPPL.CA", "HDBK.CA", "HELI.CA", "HRHO.CA",
    "JUFO.CA", "MFPC.CA", "OCDI.CA", "ORAS.CA", "ORHD.CA", "PHDC.CA",
    "SCTS.CA", "SWDY.CA", "TMGH.CA"
]

EXPECTED_FEATURES = [
    'Close', 'High', 'Low', 'Open', 'Volume', 'USD_EGP',
    'return_1d', 'return_5d', 'return_20d',
    'usd_return_1d', 'usd_return_5d', 'usd_return_20d',
    'dist_sma_20', 'dist_sma_50', 'dist_sma_100',
    'rsi_14', 'macd_pct', 'macd_signal_pct',
    'bollinger_pb', 'volatility_20d', 'vol_ratio'
] + [f"ticker_{t}" for t in EGX_TICKERS]

@st.cache_resource
def load_model():
    return joblib.load('model_pipeline.pkl')

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(target_date):
    start_date = target_date - timedelta(days=200)
    end_date = target_date + timedelta(days=1)
    
    all_stocks_data = []
    for ticker in EGX_TICKERS:
        try:
            df_stock = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not df_stock.empty:
                if isinstance(df_stock.columns, pd.MultiIndex):
                    df_stock.columns = df_stock.columns.droplevel(1)
                df_stock['Ticker'] = ticker
                all_stocks_data.append(df_stock)
        except:
            continue
            
    if not all_stocks_data:
        return None
        
    df_final = pd.concat(all_stocks_data).reset_index()
    
    try:
        usd_egp = yf.download("EGP=X", start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(usd_egp.columns, pd.MultiIndex):
            usd_egp.columns = usd_egp.columns.droplevel(1)
        usd_egp = usd_egp[['Close']].rename(columns={'Close': 'USD_EGP'})
        df_final = pd.merge(df_final, usd_egp, left_on='Date', right_index=True, how='left')
        df_final['USD_EGP'] = df_final['USD_EGP'].ffill().bfill()
    except:
        df_final['USD_EGP'] = 48.0 
        
    df_final = df_final.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    return df_final

def create_features(df):
    df = df.copy()
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_20d'] = df['Close'].pct_change(20)
    df['usd_return_1d'] = df['USD_EGP'].pct_change(1)
    df['usd_return_5d'] = df['USD_EGP'].pct_change(5)
    df['usd_return_20d'] = df['USD_EGP'].pct_change(20)
    
    sma_20 = df['Close'].rolling(window=20).mean()
    sma_50 = df['Close'].rolling(window=50).mean()
    sma_100 = df['Close'].rolling(window=100).mean()
    
    df['dist_sma_20'] = (df['Close'] / sma_20) - 1
    df['dist_sma_50'] = (df['Close'] / sma_50) - 1
    df['dist_sma_100'] = (df['Close'] / sma_100) - 1
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    df['macd_pct'] = macd / df['Close']
    df['macd_signal_pct'] = df['macd_pct'].ewm(span=9, adjust=False).mean()
    
    rolling_std = df['Close'].rolling(window=20).std()
    bollinger_high = sma_20 + (rolling_std * 2)
    bollinger_low = sma_20 - (rolling_std * 2)
    df['bollinger_pb'] = (df['Close'] - bollinger_low) / (bollinger_high - bollinger_low + 1e-8)
    
    df['volatility_20d'] = df['return_1d'].rolling(window=20).std()
    vol_sma_20 = df['Volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['Volume'] / (vol_sma_20 + 1e-8)
    
    return df

def process_data(df_final):
    df_features = df_final.groupby('Ticker', group_keys=False).apply(create_features)
    return df_features.dropna().reset_index(drop=True)

# --- UI LAYOUT ---
st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">🧭 Market Intelligence</div>
        <div class="hero-title">Smart EGX Guide</div>
        <div class="hero-subtitle">
            A one-click market scan across the Egyptian Exchange, ranking covered
            stocks by predicted next-day direction and model confidence.
        </div>
    </div>
""", unsafe_allow_html=True)

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the prediction model: {e}")
    st.stop()

with st.sidebar:
    st.markdown("#### 🧭 Smart EGX Guide")
    st.caption("Configure and run the scan.")
    st.divider()
    selected_date = st.date_input("Prediction Date", value=date.today(), max_value=date.today())
    scan_button = st.button("Scan Market", type="primary", use_container_width=True)
    st.divider()
    st.caption(f"Coverage: {len(EGX_TICKERS)} EGX-listed tickers")

if scan_button:
    with st.spinner("Fetching market data and calculating technical indicators..."):
        raw_data = fetch_market_data(selected_date)
        
        if raw_data is None or raw_data.empty:
            st.error("No market data could be found for the selected date range. Please try a different date.")
        else:
            processed_data = process_data(raw_data)
            
            df_latest = processed_data.groupby('Ticker').tail(1).copy()
            
            if df_latest.empty:
                st.error("Not enough historical data available to compute the 100-day moving average. Please select a later date.")
            else:
                for t in EGX_TICKERS:
                    df_latest[f"ticker_{t}"] = (df_latest['Ticker'] == t).astype(int)
                
                X_input = df_latest[EXPECTED_FEATURES]
                
                preds = model.predict(X_input)
                probs = model.predict_proba(X_input)[:, 1]
                
                df_latest['Prediction'] = preds
                df_latest['Confidence (%)'] = np.where(preds == 1, probs * 100, (1 - probs) * 100)
                df_latest['Signal'] = df_latest['Prediction'].map({1: '🟢 Bullish', 0: '🔴 Bearish'})
                
                bullish_count = (preds == 1).sum()
                bearish_count = (preds == 0).sum()
                
                total = len(df_latest)
                bullish_pct = (bullish_count / total * 100) if total else 0
                bearish_pct = (bearish_count / total * 100) if total else 0

                st.markdown(f"""
                    <div class="stats-row">
                        <div class="stat-card">
                            <div class="stat-label">Stocks Analyzed</div>
                            <div class="stat-value">{total}</div>
                            <div class="stat-sub">EGX tickers covered</div>
                        </div>
                        <div class="stat-card stat-bullish">
                            <div class="stat-label">🟢 Bullish Signals</div>
                            <div class="stat-value">{int(bullish_count)}</div>
                            <div class="stat-sub">{bullish_pct:.0f}% of scan</div>
                        </div>
                        <div class="stat-card stat-bearish">
                            <div class="stat-label">🔴 Bearish Signals</div>
                            <div class="stat-value">{int(bearish_count)}</div>
                            <div class="stat-sub">{bearish_pct:.0f}% of scan</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-title">Scan Results</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-sub">Sorted by model confidence.</div>', unsafe_allow_html=True)
                
                # Sort by model confidence, highest first
                df_latest = df_latest.sort_values(by='Confidence (%)', ascending=False)
                
                display_cols = ['Ticker', 'Close', 'rsi_14', 'vol_ratio', 'Signal', 'Confidence (%)']
                display_df = df_latest[display_cols].rename(columns={'rsi_14': 'RSI (14)', 'vol_ratio': 'Volume Ratio'})
                
                st.dataframe(
                    display_df,
                    column_config={
                        "Ticker": st.column_config.TextColumn("Stock"),
                        "Close": st.column_config.NumberColumn("Last Close (EGP)", format="%.2f"),
                        "RSI (14)": st.column_config.NumberColumn("RSI", format="%.1f"),
                        "Volume Ratio": st.column_config.NumberColumn("Vol Ratio", format="%.2f"),
                        "Signal": st.column_config.TextColumn("Predicted Signal"),
                        "Confidence (%)": st.column_config.ProgressColumn(
                            "Model Confidence",
                            format="%.2f%%",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                    hide_index=True,
                    use_container_width=True
                )
else:
    st.markdown("""
        <div style="text-align:center; color:#64748b; padding: 2rem 0;">
            Choose a prediction date in the sidebar and run <b>Scan Market</b> to generate the latest signals.
        </div>
    """, unsafe_allow_html=True)
