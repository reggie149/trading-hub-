import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import requests
import json
import os
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Quant Trading Dashboard", 
    layout="wide"
)
st.title("📈 Quantitative Strategy Backtester")
st.markdown(
    "Analyze historical market data, simulate "
    "trades, and execute on Robinhood."
)

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🛠️ Strategy Parameters")
ticker = st.sidebar.text_input(
    "Asset Ticker", 
    value="BTC-USD"
)
timeline = st.sidebar.selectbox(
    "History Period", 
    ["1mo", "3mo", "6mo", "1y"], 
    index=0
)
time_frame = st.sidebar.selectbox(
    "Candle Interval", 
    ["1h", "1d"], 
    index=0
)
st.sidebar.markdown("---")
initial_capital = st.sidebar.number_input(
    "Starting Capital ($)", 
    value=10000.00, 
    step=1000.00
)
fast_period = st.sidebar.slider(
    "Fast EMA Period", 
    min_value=2, 
    max_value=50, 
    value=12
)
slow_period = st.sidebar.slider(
    "Slow EMA Period", 
    min_value=5, 
    max_value=100, 
    value=26
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Mode")
app_mode = st.sidebar.radio(
    "Select Mode", 
    [
        "📊 Backtest", 
        "🧪 Simulation", 
        "🤖 Robinhood Live", 
        "📋 Rules Manager"
    ]
)

# --- Volume Profile Settings ---
st.sidebar.markdown("---")
st.sidebar.header("📊 Volume Profile")
vp_bins = st.sidebar.slider(
    "Volume Profile Bins", 
    min_value=10, 
    max_value=100, 
    value=40
)
show_vp = st.sidebar.toggle(
    "Show Volume Profile", 
    value=True
)
show_poc = st.sidebar.toggle(
    "Show POC Line", 
    value=True
)
show_value_area = st.sidebar.toggle(
    "Show Value Area (70%)", 
    value=True
)

# --- Fair Value Gap Settings ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Price Imbalances")
show_fvg = st.sidebar.toggle(
    "Show Fair Value Gaps (FVG)", 
    value=True
)

# ============================================================
# COINGECKO CRYPTO ID MAP
# ============================================================
CRYPTO_MAP = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "SOL-USD": "solana",
    "DOGE-USD": "dogecoin",
    "ADA-USD": "cardano",
    "XRP-USD": "ripple",
    "LTC-USD": "litecoin",
    "MATIC-NETWORK": "matic-network",
    "AVAX-USD": "avalanche-2",
    "DOT-USD": "polkadot",
}

PERIOD_DAYS = {
    "1mo": 30, 
    "3mo": 90, 
    "6mo": 180, 
    "1y": 365
}

def is_crypto(symbol):
    clean_sym = symbol.upper()
    return (
        clean_sym in CRYPTO_MAP or 
        clean_sym.endswith("-USD")
    )

def load_crypto_coingecko(symbol, period):
    coin_id = CRYPTO_MAP.get(symbol.upper())
    if not coin_id:
        return pd.DataFrame()
    days = PERIOD_DAYS.get(period, 30)
    
    base_url = "https://api.coingecko.com"
    endpoint = f"/api/v3/coins/{coin_id}/ohlc"
    url = f"{base_url}{endpoint}"
    
    params = {
        "vs_currency": "usd", 
        "days": days
    }
    try:
        resp = requests.get(
            url, 
            params=params, 
            timeout=10
        )
        data = resp.json()
        if not isinstance(data, list):
            return pd.DataFrame()
        
        cols = [
            "Datetime", "Open", 
            "High", "Low", "Close"
        ]
        df = pd.DataFrame(data, columns=cols)
        df["Datetime"] = pd.to_datetime(
            df["Datetime"], 
            unit="ms"
        )
        df = df.sort_values("Datetime")
        df = df.reset_index(drop=True)
        df["Volume"] = (
            (df["High"] - df["Low"]) * 1000
        )
        return df
    except Exception as e:
        return pd.DataFrame()

def get_live_price_crypto(symbol):
    coin_id = CRYPTO_MAP.get(symbol.upper())
    if not coin_id:
        return None
    try:
        base_url = "https://api.coingecko.com"
        endpoint = "/api/v3/simple/price"
        url = f"{base_url}{endpoint}"
        p_dict = {
            "ids": coin_id, 
            "vs_currencies": "usd"
        }
        resp = requests.get(
            url, 
            params=p_dict, 
            timeout=5
        )
        return float(
            resp.json()[coin_id]["usd"]
        )
    except:
        return None

@st.cache_data(ttl=300)
def load_data(symbol, per, inter):
    if is_crypto(symbol):
        return load_crypto_coingecko(symbol, per)
    else:
        df = yf.download(
            symbol, 
            period=per, 
            interval=inter, 
            progress=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = 'Datetime'
        df.reset_index(inplace=True)
        return df

def get_live_price(symbol):
    if is_crypto(symbol):
        return get_live_price_crypto(symbol)
    try:
        data = yf.download(
            symbol, 
            period="1d", 
            interval="1m", 
            progress=False
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return float(data['Close'].iloc[-1])
    except:
        return None

def compute_emas(df, fast, slow):
    df = df.copy()
    df['Fast_EMA'] = (
        df['Close']
        .ewm(span=fast, adjust=False)
        .mean()
    )
    df['Slow_EMA'] = (
        df['Close']
        .ewm(span=slow, adjust=False)
        .mean()
    )
    return df

# ============================================================
# FAIR VALUE GAP CALCULATION
# ============================================================
def find_fair_value_gaps(df):
    fvg_list = []
    if len(df) < 3:
        return fvg_list

    for i in range(1, len(df) - 1):
        c1_high = float(df.loc[i-1, 'High'])
        c1_low = float(df.loc[i-1, 'Low'])
        c3_high = float(df.loc[i+1, 'High'])
        c3_low = float(df.loc[i+1, 'Low'])
        
        t2 = df.loc[i, 'Datetime']
        t_end = df.iloc[-1]['Datetime']

        # Bullish FVG
        if c3_low > c1_high:
            fvg_list.append({
                "type": "bullish",
                "top": c3_low,
                "bottom": c1_high,
                "start_time": t2,
                "end_time": t_end
            })
        # Bearish FVG
        elif c1_low > c3_high:
            fvg_list.append({
                "type": "bearish",
                "top": c1_low,
                "bottom": c3_high,
                "start_time": t2,
                "end_time": t_end
            })
            
    return fvg_list

# ============================================================
# VOLUME PROFILE CALCULATION
# ============================================================
def compute_volume_profile(df, bins=40):
    if df.empty or "Volume" not in df.columns:
        return None
    price_min = df["Low"].min()
    price_max = df["High"].max()
    if price_min == price_max:
        return None
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    volume_at_level = np
