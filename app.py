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
    
    # Broken up to prevent line truncation
    base_url = "https://api.coingecko.com"
    endpoint = f"/api/v3/coins/{coin_id}/ohlc"
    url = f"{base_url}{endpoint}"
    
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
        base_url = "https://api.coingecko.com"
        endpoint = "/api/v3/simple/price"
        url = f"{base_url}{endpoint}"
        p_dict = {"ids": coin_id, "vs_currencies": "usd"}
        resp = requests.get(url, params=p_dict, timeout=5)
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
            if fvg["type"] == "bullish":
                color = "rgba(0, 255, 120, 0.12)"
                line_color = "rgba(0, 255, 120, 0.25)"
            else:
                color = "rgba(255, 50, 50, 0.12)"
                line_color = "rgba(255, 50, 50, 0.25)"
            
            fig.add_shape(
                type="rect",
                x0=fvg["start_time"], x1=fvg["end_time"], xref="x",
                y0=fvg["bottom"], y1=fvg["top"], yref="y",
                fillcolor=color,
                line=dict(color=line_color, width=1),
                layer="below"
            )

    fig.add_trace(go.Candlestick(
        x=df['Datetime'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name="Price Action",
        xaxis="x", yaxis="y"
    ))
    fig.add_trace(go.Scatter(
        x=df['Datetime'], y=df['Fast_EMA'],
        line=dict(color='orange', width=1.5),
        name=f'{fast_period} Fast EMA',
        xaxis="x", yaxis="y"
    ))
    fig.add_trace(go.Scatter(
        x=df['Datetime'], y=df['Slow_EMA'],
        line=dict(color='#4da6ff', width=1.5),
        name=f'{slow_period} Slow EMA',
        xaxis="x", yaxis="y"
    ))
    if buy_x:
        fig.add_trace(go.Scatter(
            x=buy_x, y=buy_y, mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='green',
                        line=dict(width=2, color='black')),
            name='BUY Entry', xaxis="x", yaxis="y"
        ))
    if sell_x:
        fig.add_trace(go.Scatter(
            x=sell_x, y=sell_y, mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='red',
                        line=dict(width=2, color='black')),
            name='SELL Exit', xaxis="x", yaxis="y"
        ))

    if show_vp:
        vp = compute_volume_profile(df, bins=vp_bins)
        if vp is not None:
            prices = vp["price_levels"]
            volumes = vp["volumes"]
            poc = vp["poc_price"]
            vah = vp["vah_price"]
            val = vp["val_price"]
            max_vol = volumes.max() if volumes.max() > 0 else 1

            bar_colours = []
            for p, v in zip(prices, volumes):
                if abs(p - poc) < (prices[1] - prices[0]) * 0.6:
                    bar_colours.append("rgba(255, 220, 50, 0.90)")
                elif val <= p <= vah:
                    bar_colours.append("rgba(50, 200, 180, 0.55)")
                else:
                    bar_colours.append("rgba(160, 160, 160, 0.35)")

            price_span = prices[-1] - prices[0]
            bar_height = price_span / vp_bins * 0.85
            norm_vols = volumes / max_vol

            # Shortened hovertemplate to avoid string clipping anomalies
            h_tpl = "Price: %{y:,.2f}<br>Vol: %{customdata:,.0f}<extra></extra>"

            fig.add_trace(go.Bar(
                x=norm_vols,
                y=prices,
                orientation='h',
                width=bar_height,
                marker_color=bar_colours,
                name="Volume Profile",
                xaxis="x2",
                yaxis="y",
                hovertemplate=h_tpl,
                customdata=volumes,
                showlegend=True,
            ))

            if show_poc:
                fig.add_shape(
                    type="line",
                    x0=0, x1=1, xref="paper",
                    y0=poc, y1=poc, yref="y",
                    line=dict(color="rgba(255,220,50,0.85)", width=1.5, dash="dot"),
                )
                fig.add_annotation(
                    x=1, xref="paper",
                    y=poc, yref="y",
                    text=f" POC {poc:,.2f}",
                    showarrow=False,
                    font=dict(color="rgba(255,220,50,0.95)", size=11),
                    xanchor="left",
                )

            if show_value_area:
                fig.add_shape(
                    type="rect",
                    x0=0, x1=1, xref="paper",
                    y0=val, y1=vah, yref="y",
                    fillcolor="rgba(50,200,180,0.07)",
                    line=dict(color="rgba(50,200,180,0.45)", width=1, dash="dash"),
                )
                fig.add_annotation(
                    x=1, xref="paper",
                    y=vah, yref="y",
                    text=f" VAH {vah:,.2f}",
                    showarrow=False,
                    font=dict(color="rgba(50,200,180,0.9)", size=10),
                    xanchor="left",
                )
                fig.add_annotation(
                    x=1, xref="paper",
                    y=val, yref="y",
                    text=f" VAL {val:,.2f}",
                    showarrow=False,
                    font=dict(color="rgba(50,200,180,0.9)", size=10),
                    xanchor="left",
                )

    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=False),
            domain=[0, 0.82],
        ),
        xaxis2=dict(
            domain=[0.83, 1.0],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[0, 1.05],
            fixedrange=True,
            autorange="reversed",
        ),
        yaxis=dict(
            side="right",
        ),
        height=580,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(r=120),
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# BACKTEST MODE
# ============================================================
if app_mode == "📊 Backtest":
    df_raw = load_data(ticker, timeline, time_frame)

    if df_raw.empty:
        err_msg = "No data found. Try crypto tickers like BTC-USD or stocks like AAPL."
        st.error(err_msg)
        st.info("Supported crypto: " + ", ".join(CRYPTO_MAP.keys()))
    else:
        df = compute_emas(df_raw, fast_period, slow_period)
        cash = initial_capital
        position = 0.0
        is_invested = False
        trade_log = []
        buy_signals_x, buy_signals_y = [], []
        sell_signals_x, sell_signals_y = [], []

        for idx, row in df.iterrows():
            if idx < slow_period:
                continue
            current_price = float(row['Close'])
            fast_ma = float(row['Fast_EMA'])
            slow_ma = float(row['Slow_EMA'])
            timestamp = row['Datetime']

            if not is_invested and fast_ma > slow_ma:
                position = cash / current_price
                cash = 0.0
                is_invested = True
                t_val = round(position * current_price, 2)
                trade_log.append({"Action": "BUY", "Time": timestamp, "Price": round(current_price, 2), "Net Worth": t_val})
                buy_signals_x.append(timestamp)
                buy_signals_y.append(current_price)
            elif is_invested and fast_ma < slow_ma:
                cash = position * current_price
                position = 0.0
                is_invested = False
                trade_log.append({"Action": "SELL", "Time": timestamp, "Price": round(current_price, 2), "Net Worth": round(cash, 2)})
                sell_signals_x.append(timestamp)
                sell_signals_y.append(current_price)

        final_price = float(df.iloc[-1]['Close'])
        final_value = cash if not is_invested else (position * final_price)
        total_return = ((final_value - initial_capital) / initial_capital) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Starting Capital", f"${initial_capital:,.2f}")
        col2.metric("Ending Net Worth", f"${final_value:,.2f}")
        col3.metric("Total Return", f"{total_return:+.2f}%", delta_color="normal")
        col4.metric("Total Trades Executed", len(trade_log))
        st.markdown("---")
        st.subheader("📊 Interactive Market Chart & Execution Flags")
        render_chart(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y, fast_period, slow_period)
        st.markdown("---")
        st.subheader("📜 Complete Trade Ledger Logs")
        if trade_log:
            st.dataframe(pd.DataFrame(trade_log), use_container_width=True)
        else:
            st.info("No trades executed. Try tightening your EMA horizons.")

# ============================================================
# SIMULATION MODE
# ============================================================
elif app_mode == "🧪 Simulation":
    st.header("🧪 Simulation Module")
    sim_mode = st.sidebar.radio("Simulation Type", ["📡 Paper Trade (Live Prices)", "🔁 Historical Replay"])
    auto_trade = st.toggle("🤖 Auto-trade on EMA signals", value=False)

    if 'sim_cash' not in st.session_state:
        st.session_state.sim_cash = initial_capital
        st.session_state.sim_position = 0.0
        st.session_state.sim_invested = False
        st.session_state.sim_trades = []
        st.session_state.sim_buy_x = []
        st.session_state.sim_buy_y = []
        st.session_state.sim_sell_x = []
        st.session_state.sim_sell_y = []

    def sim_buy(price, timestamp):
        if not st.session_state.sim_invested and st.session_state.sim_cash > 0:
            st.session_state.sim_position = st.session_state.sim_cash / price
            st.session_state.sim_cash = 0.0
            st.session_state.sim_invested = True
            n_w = round(st.session_state.sim_position * price, 2)
            st.session_state.sim_trades.append({"Action": "BUY", "Time": str(timestamp), "Price": round(price, 2), "Net Worth": n_w})
            st.session_state.sim_buy_x.append(timestamp)
            st.session_state.sim_buy_y.append(price)

    def sim_sell(price, timestamp):
        if st.session_state.sim_invested:
            st.session_state.sim_cash = st.session_state.sim_position * price
            st.session_state.sim_position = 0.0
            st.session_state.sim_invested = False
            n_w = round(st.session_state.sim_cash, 2)
            st.session_state.sim_trades.append({"Action": "SELL", "Time": str(timestamp), "Price": round(price, 2),
