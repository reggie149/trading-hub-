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
        url = "https://api.
