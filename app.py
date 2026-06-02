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
st.set_page_config(page_title="Quant Trading Dashboard", layout="wide")
st.title("📈 Quantitative Strategy Backtester")
st.markdown("Analyze historical market data, simulate trades, and execute on Robinhood.")

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🛠️ Strategy Parameters")
ticker = st.sidebar.text_input("Asset Ticker", value="BTC-USD")
timeline = st.sidebar.selectbox("History Period", ["1mo", "3mo", "6mo", "1y"], index=0)
time_frame = st.sidebar.selectbox("Candle Interval", ["1h", "1d"], index=0)
st.sidebar.markdown("---")
initial_capital = st.sidebar.number_input("Starting Capital ($)", value=10000.00, step=1000.00)
fast_period = st.sidebar.slider("Fast EMA Period", min_value=2, max_value=50, value=12)
slow_period = st.sidebar.slider("Slow EMA Period", min_value=5, max_value=100, value=26)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Mode")
app_mode = st.sidebar.radio("Select Mode", ["📊 Backtest", "🧪 Simulation", "🤖 Robinhood Live", "📋 Rules Manager"])

# --- Volume Profile Settings ---
st.sidebar.markdown("---")
st.sidebar.header("📊 Volume Profile")
vp_bins = st.sidebar.slider("Volume Profile Bins", min_value=10, max_value=100, value=40)
show_vp = st.sidebar.toggle("Show Volume Profile", value=True)
show_poc = st.sidebar.toggle("Show POC Line", value=True)
show_value_area = st.sidebar.toggle("Show Value Area (70%)", value=True)

# --- Fair Value Gap Settings ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Price Imbalances")
show_fvg = st.sidebar.toggle("Show Fair Value Gaps (FVG)", value=True)

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

PERIOD_DAYS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}

def is_crypto(symbol):
    return symbol.upper() in CRYPTO_MAP or symbol.upper().endswith("-USD")

def load_crypto_coingecko(symbol, period):
    coin_id = CRYPTO_MAP.get(symbol.upper())
    if not coin_id:
        return pd.DataFrame()
    days = PERIOD_DAYS.get(period, 30)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if not isinstance(data, list):
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=["Datetime", "Open", "High", "Low", "Close"])
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit="ms")
        df = df.sort_values("Datetime").reset_index(drop=True)
        df["Volume"] = (df["High"] - df["Low"]) * 1000
        return df
    except Exception as e:
        return pd.DataFrame()

def get_live_price_crypto(symbol):
    coin_id = CRYPTO_MAP.get(symbol.upper())
    if not coin_id:
        return None
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        resp = requests.get(url, params={"ids": coin_id, "vs_currencies": "usd"}, timeout=5)
        return float(resp.json()[coin_id]["usd"])
    except:
        return None

@st.cache_data(ttl=300)
def load_data(symbol, per, inter):
    if is_crypto(symbol):
        return load_crypto_coingecko(symbol, per)
    else:
        df = yf.download(symbol, period=per, interval=inter, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = 'Datetime'
        df.reset_index(inplace=True)
        return df

def get_live_price(symbol):
    if is_crypto(symbol):
        return get_live_price_crypto(symbol)
    try:
        data = yf.download(symbol, period="1d", interval="1m", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return float(data['Close'].iloc[-1])
    except:
        return None

def compute_emas(df, fast, slow):
    df = df.copy()
    df['Fast_EMA'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['Slow_EMA'] = df['Close'].ewm(span=slow, adjust=False).mean()
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
    volume_at_level = np.zeros(bins)
    for _, row in df.iterrows():
        vol = float(row["Volume"])
        if vol <= 0 or pd.isna(vol):
            continue
        low, high = float(row["Low"]), float(row["High"])
        lo_idx = np.searchsorted(bin_edges, low, side="left")
        hi_idx = np.searchsorted(bin_edges, high, side="right")
        lo_idx = max(0, lo_idx - 1)
        hi_idx = min(bins, hi_idx)
        n_bins = hi_idx - lo_idx
        if n_bins <= 0:
            continue
        volume_at_level[lo_idx:hi_idx] += vol / n_bins
    poc_idx = int(np.argmax(volume_at_level))
    poc_price = bin_centres[poc_idx]
    total_vol = volume_at_level.sum()
    target_vol = total_vol * 0.70
    lo_ptr, hi_ptr = poc_idx, poc_idx
    area_vol = volume_at_level[poc_idx]
    while area_vol < target_vol:
        expand_lo = volume_at_level[lo_ptr - 1] if lo_ptr > 0 else 0
        expand_hi = volume_at_level[hi_ptr + 1] if hi_ptr < bins - 1 else 0
        if expand_lo == 0 and expand_hi == 0:
            break
        if expand_hi >= expand_lo:
            hi_ptr += 1
            area_vol += volume_at_level[hi_ptr]
        else:
            lo_ptr -= 1
            area_vol += volume_at_level[lo_ptr]
    return {
        "price_levels": bin_centres,
        "volumes": volume_at_level,
        "poc_price": poc_price,
        "vah_price": bin_centres[hi_ptr],
        "val_price": bin_centres[lo_ptr],
    }

# ============================================================
# CHART RENDERING WITH VOLUME PROFILE & FVG
# ============================================================
def render_chart(df, buy_x, buy_y, sell_x, sell_y, fast_period, slow_period):
    fig = go.Figure()

    # --- Draw Fair Value Gaps (Behind the candlesticks) ---
    if show_fvg:
        fvgs = find_fair_value_gaps(df)
        for fvg in fvgs:
            color = "rgba(0, 255, 1
