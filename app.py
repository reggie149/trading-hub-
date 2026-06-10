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

timeline = st.sidebar.selectbox(
    "History Period",
    ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=2,
)

time_frame = st.sidebar.radio(
    "Candle Interval",
    ["1h", "4h", "1d"],
    horizontal=True,
)

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
st.sidebar.header("🔲 Fair Value Gaps")
show_fvg        = st.sidebar.toggle("Show Fair Value Gaps", value=True)
fvg_max         = st.sidebar.slider("Max Auto-FVGs to Display", min_value=1, max_value=50, value=10)
show_bull_fvg   = st.sidebar.toggle("Show Bullish FVGs", value=True)
show_bear_fvg   = st.sidebar.toggle("Show Bearish FVGs", value=True)
show_filled     = st.sidebar.toggle("Show Filled FVGs", value=False)
show_fvg_labels = st.sidebar.toggle("Show FVG Labels", value=True)
fvg_min_pct     = st.sidebar.slider("Min Gap Size (% of price)", min_value=0.0, max_value=2.0, value=0.05, step=0.01,
                                     help="Ignore gaps smaller than this % of current price — filters out noise")
fvg_max_pct     = st.sidebar.slider("Max Gap Size (% of price)", min_value=0.5, max_value=20.0, value=2.0, step=0.1,
                                     help="Ignore gaps larger than this % of current price — removes chart-swallowing boxes")

# --- Market Hours Settings ---
st.sidebar.markdown("---")
st.sidebar.header("🕐 Market Hours Filter")
show_closed_gaps = st.sidebar.toggle("Show Closed Market Annotations", value=True)

# --- Risk / Reward Settings ---
st.sidebar.markdown("---")
st.sidebar.header("⚖️ Risk / Reward Tool")
show_rr          = st.sidebar.toggle("Show R:R Tool", value=False)
show_rr_on_chart = st.sidebar.toggle("Show R:R Label on Chart", value=True)
rr_account_size  = st.sidebar.number_input("Account Size ($)", value=10000.0, step=500.0)
rr_risk_pct      = st.sidebar.slider("Max Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

# ============================================================
# INTERVAL / PERIOD MAPS & VALIDATION
# ============================================================

YF_INTERVAL_MAP = {"1h": "1h", "4h": "1h", "1d": "1d"}

YF_PERIOD_LIMITS = {
    "1h": ["1d", "5d", "1mo", "3mo"],
    "1d": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
}


def clamp_period(yf_inter: str, per: str) -> str:
    allowed = YF_PERIOD_LIMITS.get(yf_inter)
    if allowed and per not in allowed:
        return allowed[-1]
    return per


# ============================================================
# COINGECKO CRYPTO ID MAP
# ============================================================
CRYPTO_MAP = {
    "BTC-USD": "bitcoin",    "ETH-USD": "ethereum",   "SOL-USD": "solana",
    "DOGE-USD": "dogecoin",  "ADA-USD": "cardano",     "XRP-USD": "ripple",
    "LTC-USD": "litecoin",   "MATIC-USD": "matic-network",
    "AVAX-USD": "avalanche-2", "DOT-USD": "polkadot",
}

PERIOD_DAYS = {
    "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
    "6mo": 180, "1y": 365, "2y": 730, "5y": 1825,
}


def is_crypto(symbol: str) -> bool:
    return symbol.upper() in CRYPTO_MAP or symbol.upper().endswith("-USD")


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = df.set_index("Datetime")
    ohlc_dict = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    df = df.resample("4h").agg(ohlc_dict).dropna(subset=["Close"])
    df = df.reset_index()
    return df


def load_crypto_coingecko(symbol: str, period: str, interval: str = "1h") -> pd.DataFrame:
    coin_id = CRYPTO_MAP.get(symbol.upper())
    if not coin_id:
        return pd.DataFrame()
    days = PERIOD_DAYS.get(period, 30)
    url  = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    try:
        resp = requests.get(url, params={"vs_currency": "usd", "days": days}, timeout=10)
        data = resp.json()
        if not isinstance(data, list):
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=["Datetime", "Open", "High", "Low", "Close"])
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit="ms")
        df = df.sort_values("Datetime").reset_index(drop=True)
        df["Volume"] = (df["High"] - df["Low"]) * 1000
        if interval == "4h":
            df = resample_4h(df)
        elif interval == "1d":
            df = df.set_index("Datetime")
            ohlc_dict = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            df = df.resample("1D").agg(ohlc_dict).dropna(subset=["Close"]).reset_index()
        return df
    except Exception:
        return pd.DataFrame()


def get_live_price_crypto(symbol: str):
    coin_id = CRYPTO_MAP.get(symbol.upper())
    if not coin_id:
        return None
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": coin_id, "vs_currencies": "usd"},
            timeout=5,
        )
        return float(resp.json()[coin_id]["usd"])
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_data(symbol: str, per: str, inter: str) -> pd.DataFrame:
    if is_crypto(symbol):
        return load_crypto_coingecko(symbol, per, interval=inter)

    yf_inter = YF_INTERVAL_MAP.get(inter, inter)
    safe_per = clamp_period(yf_inter, per)

    df = yf.download(symbol, period=safe_per, interval=yf_inter, progress=False)

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Datetime"
    df.reset_index(inplace=True)

    if pd.api.types.is_datetime64tz_dtype(df["Datetime"]):
        df["Datetime"] = df["Datetime"].dt.tz_convert(None)

    if inter == "4h":
        df = resample_4h(df)

    return df


def get_live_price(symbol: str):
    if is_crypto(symbol):
        return get_live_price_crypto(symbol)
    try:
        data = yf.download(symbol, period="1d", interval="1m", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return float(data["Close"].iloc[-1])
    except Exception:
        return None


def compute_emas(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    df = df.copy()
    df["Fast_EMA"] = df["Close"].ewm(span=fast, adjust=False).mean()
    df["Slow_EMA"] = df["Close"].ewm(span=slow, adjust=False).mean()
    return df


# ============================================================
# PERIOD-CLAMPING WARNING HELPER
# ============================================================
def warn_if_clamped(symbol: str, inter: str, requested_per: str) -> None:
    if is_crypto(symbol):
        return
    yf_inter = YF_INTERVAL_MAP.get(inter, inter)
    safe = clamp_period(yf_inter, requested_per)
    if safe != requested_per:
        st.warning(
            f"⚠️ yfinance only supports **{inter}** data up to **{safe}**. "
            f"History period auto-adjusted from `{requested_per}` → `{safe}`."
        )


# ============================================================
# TIMESTAMP HELPER
# ============================================================
def safe_isoformat(ts) -> str:
    t = pd.Timestamp(ts)
    try:
        t = t.tz_localize(None)
    except TypeError:
        try:
            t = t.tz_convert(None)
        except Exception:
            pass
    return t.isoformat()


# ============================================================
# TRADING HOURS FILTER
# ============================================================

# NYSE / NASDAQ regular session in UTC: 14:30 – 21:00
MARKET_OPEN_UTC  = 14 * 60 + 30   # minutes since midnight
MARKET_CLOSE_UTC = 21 * 60 + 0


def filter_trading_hours(df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """
    Drop rows that fall outside regular NYSE/NASDAQ trading hours (14:30–21:00 UTC).
    - Crypto (24/7) and daily candles are returned unchanged.
    - 1h and 4h stock/ETF data is filtered to session hours only.
    """
    if interval == "1d" or is_crypto(symbol):
        return df
    if df.empty:
        return df
    minutes = df["Datetime"].dt.hour * 60 + df["Datetime"].dt.minute
    mask = (minutes >= MARKET_OPEN_UTC) & (minutes < MARKET_CLOSE_UTC)
    return df[mask].reset_index(drop=True)


def get_closed_gaps(df_raw: pd.DataFrame, df_filtered: pd.DataFrame,
                    symbol: str, interval: str) -> list:
    """
    Compare raw vs filtered DataFrames to find contiguous market-closed blocks.
    Returns a list of dicts: { x0, x1, label } for use as chart annotations.
    Returns [] for crypto or daily intervals.
    """
    if interval == "1d" or is_crypto(symbol):
        return []
    if df_raw.empty or df_filtered.empty:
        return []

    raw_times      = set(df_raw["Datetime"].astype(str))
    filtered_times = set(df_filtered["Datetime"].astype(str))
    removed        = sorted(raw_times - filtered_times)

    if not removed:
        return []

    delta_map = {"1h": pd.Timedelta("1h"), "4h": pd.Timedelta("4h")}
    candle_td = delta_map.get(interval, pd.Timedelta("1h"))

    gaps        = []
    block_start = removed[0]
    prev        = removed[0]

    for ts in removed[1:]:
        if pd.Timestamp(ts) - pd.Timestamp(prev) > candle_td * 1.5:
            gaps.append((block_start, prev))
            block_start = ts
        prev = ts
    gaps.append((block_start, prev))

    result = []
    for x0_str, x1_str in gaps:
        t0 = pd.Timestamp(x0_str)
        t1 = pd.Timestamp(x1_str) + candle_td
        label = f"Market closed  {t0.strftime('%H:%M')}–{t1.strftime('%H:%M')} UTC"
        result.append({"x0": t0.isoformat(), "x1": t1.isoformat(), "label": label})
    return result


# ============================================================
# FAIR VALUE GAP CALCULATION
# ============================================================
def find_fair_value_gaps(df: pd.DataFrame) -> list:
    fvg_list = []
    if len(df) < 3:
        return fvg_list

    datetimes = df["Datetime"].tolist()
    last_iso  = safe_isoformat(datetimes[-1])

    for i in range(1, len(df) - 1):
        c1_high = float(df["High"].iloc[i - 1])
        c1_low  = float(df["Low"].iloc[i - 1])
        c3_high = float(df["High"].iloc[i + 1])
        c3_low  = float(df["Low"].iloc[i + 1])

        if any(pd.isna(v) for v in [c1_high, c1_low, c3_high, c3_low]):
            continue

        is_bullish = c3_low > c1_high
        is_bearish = c1_low > c3_high

        if not is_bullish and not is_bearish:
            continue

        gap_top    = c3_low  if is_bullish else c1_low
        gap_bottom = c1_high if is_bullish else c3_high

        mid_price = (gap_top + gap_bottom) / 2.0
        gap_pct   = ((gap_top - gap_bottom) / mid_price) * 100.0
        if gap_pct < fvg_min_pct or gap_pct > fvg_max_pct:
            continue

        start_iso = safe_isoformat(datetimes[i])
        fvg_type  = "bullish" if is_bullish else "bearish"

        filled  = False
        end_iso = last_iso
        for j in range(i + 2, len(df)):
            future_low  = float(df["Low"].iloc[j])
            future_high = float(df["High"].iloc[j])
            if future_low <= gap_top and future_high >= gap_bottom:
                end_iso = safe_isoformat(datetimes[j])
                filled  = True
                break

        fvg_list.append({
            "type":       fvg_type,
            "top":        gap_top,
            "bottom":     gap_bottom,
            "start_time": start_iso,
            "end_time":   end_iso,
            "filled":     filled,
            "source":     "auto",
        })

    return fvg_list


# ============================================================
# VOLUME PROFILE
# ============================================================
def compute_volume_profile(df: pd.DataFrame, bins: int = 40):
    if df.empty or "Volume" not in df.columns:
        return None
    price_min = df["Low"].min()
    price_max = df["High"].max()
    if price_min == price_max:
        return None
    bin_edges       = np.linspace(price_min, price_max, bins + 1)
    bin_centres     = (bin_edges[:-1] + bin_edges[1:]) / 2
    volume_at_level = np.zeros(bins)
    for _, row in df.iterrows():
        vol = float(row["Volume"])
        if vol <= 0 or pd.isna(vol):
            continue
        lo_idx = max(0, np.searchsorted(bin_edges, float(row["Low"]),  side="left")  - 1)
        hi_idx = min(bins, np.searchsorted(bin_edges, float(row["High"]), side="right"))
        n_bins = hi_idx - lo_idx
        if n_bins > 0:
            volume_at_level[lo_idx:hi_idx] += vol / n_bins
    poc_idx   = int(np.argmax(volume_at_level))
    total_vol = volume_at_level.sum()
    lo_ptr, hi_ptr = poc_idx, poc_idx
    area_vol  = volume_at_level[poc_idx]
    while area_vol < total_vol * 0.70:
        expand_lo = volume_at_level[lo_ptr - 1] if lo_ptr > 0        else 0
        expand_hi = volume_at_level[hi_ptr + 1] if hi_ptr < bins - 1 else 0
        if expand_lo == 0 and expand_hi == 0:
            break
        if expand_hi >= expand_lo:
            hi_ptr += 1; area_vol += volume_at_level[hi_ptr]
        else:
            lo_ptr -= 1; area_vol += volume_at_level[lo_ptr]
    return {
        "price_levels": bin_centres,
        "volumes":      volume_at_level,
        "poc_price":    bin_centres[poc_idx],
        "vah_price":    bin_centres[hi_ptr],
        "val_price":    bin_centres[lo_ptr],
    }


# ============================================================
# CHART RENDERING
# ============================================================
def render_chart(df, buy_x, buy_y, sell_x, sell_y, fast_period, slow_period,
                 extra_fvgs=None, closed_gaps=None):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["Datetime"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price Action", xaxis="x", yaxis="y"
    ))
    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["Fast_EMA"],
        line=dict(color="orange", width=1.5),
        name=f"{fast_period} Fast EMA", xaxis="x", yaxis="y"
    ))
    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["Slow_EMA"],
        line=dict(color="#4da6ff", width=1.5),
        name=f"{slow_period} Slow EMA", xaxis="x", yaxis="y"
    ))

    if "Volume" in df.columns:
        vol_colors = [
            "rgba(0,200,100,0.6)" if float(df["Close"].iloc[i]) >= float(df["Open"].iloc[i])
            else "rgba(255,60,60,0.6)"
            for i in range(len(df))
        ]
        fig.add_trace(go.Bar(
            x=df["Datetime"], y=df["Volume"],
            marker_color=vol_colors, name="Volume",
            xaxis="x3", yaxis="y3", showlegend=True,
            hovertemplate="%{x}<br>Volume: %{y:,.0f}<extra></extra>",
        ))

    if buy_x:
        fig.add_trace(go.Scatter(
            x=buy_x, y=buy_y, mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="green", line=dict(width=2, color="black")),
            name="BUY Entry", xaxis="x", yaxis="y"
        ))
    if sell_x:
        fig.add_trace(go.Scatter(
            x=sell_x, y=sell_y, mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="red", line=dict(width=2, color="black")),
            name="SELL Exit", xaxis="x", yaxis="y"
        ))

    # ---- MARKET CLOSED ANNOTATIONS ----
    if show_closed_gaps and closed_gaps:
        for gap in closed_gaps:
            # Subtle shaded column behind the price pane
            fig.add_vrect(
                x0=gap["x0"], x1=gap["x1"],
                fillcolor="rgba(120,120,160,0.07)",
                line=dict(color="rgba(120,120,160,0.20)", width=0.5, dash="dot"),
                layer="below",
            )
            # Small label pinned just above the chart area
            fig.add_annotation(
                x=gap["x0"],
                xref="x",
                y=1.01,
                yref="paper",
                text=gap["label"],
                showarrow=False,
                font=dict(size=9, color="rgba(180,180,210,0.85)"),
                xanchor="left",
                bgcolor="rgba(30,30,50,0.50)",
                borderpad=3,
            )

    # ---- RISK / REWARD ZONES ----
    if show_rr and "rr_state" in st.session_state:
        rr = st.session_state.rr_state
        entry = rr["entry"]
        sl    = rr["sl"]
        tp    = rr["tp"]
        x0    = df["Datetime"].iloc[0]
        x1    = df["Datetime"].iloc[-1]

        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        ratio  = reward / risk if risk > 0 else 0

        y_min = float(df["Low"].min())
        y_max = float(df["High"].max())
        y_pad = (y_max - y_min) * 0.05

        # SL zone (red)
        fig.add_shape(type="rect",
            x0=df["Datetime"].iloc[0], x1=df["Datetime"].iloc[-1], xref="x",
            y0=min(entry, sl), y1=max(entry, sl), yref="y",
            fillcolor="rgba(255,60,60,0.18)", line=dict(width=0), layer="below")

        # TP zone (green)
        fig.add_shape(type="rect",
            x0=df["Datetime"].iloc[0], x1=df["Datetime"].iloc[-1], xref="x",
            y0=min(entry, tp), y1=max(entry, tp), yref="y",
            fillcolor="rgba(0,200,100,0.18)", line=dict(width=0), layer="below")

        # Entry line
        fig.add_shape(type="line",
            x0=df["Datetime"].iloc[0], x1=df["Datetime"].iloc[-1], xref="x",
            y0=entry, y1=entry, yref="y",
            line=dict(color="rgba(255,255,255,0.85)", width=1.5, dash="dash"))

        # SL line
        fig.add_shape(type="line",
            x0=df["Datetime"].iloc[0], x1=df["Datetime"].iloc[-1], xref="x",
            y0=sl, y1=sl, yref="y",
            line=dict(color="rgba(255,60,60,0.95)", width=1.5, dash="dot"))

        # TP line
        fig.add_shape(type="line",
            x0=df["Datetime"].iloc[0], x1=df["Datetime"].iloc[-1], xref="x",
            y0=tp, y1=tp, yref="y",
            line=dict(color="rgba(0,200,100,0.95)", width=1.5, dash="dot"))

        # Labels on right
        fig.add_annotation(x=1, xref="paper", y=entry, yref="y",
            text=f" Entry {entry:,.2f}", showarrow=False,
            font=dict(color="rgba(255,255,255,0.9)", size=10), xanchor="left")
        fig.add_annotation(x=1, xref="paper", y=sl, yref="y",
            text=f" SL {sl:,.2f}", showarrow=False,
            font=dict(color="rgba(255,80,80,0.9)", size=10), xanchor="left")
        fig.add_annotation(x=1, xref="paper", y=tp, yref="y",
            text=f" TP {tp:,.2f}", showarrow=False,
            font=dict(color="rgba(0,220,100,0.9)", size=10), xanchor="left")

        # R:R label on chart
        if show_rr_on_chart:
            fig.add_annotation(
                x=0.01, xref="paper", y=0.97, yref="paper",
                text=f"R:R  1:{ratio:.2f}",
                showarrow=False,
                font=dict(color="rgba(255,220,50,0.95)", size=12),
                bgcolor="rgba(0,0,0,0.45)",
                borderpad=4, xanchor="left")

        fig.update_layout(yaxis=dict(range=[y_min - y_pad, y_max + y_pad], side="right", domain=[0.25, 1.0]))

    if show_fvg:
        auto_fvgs   = find_fair_value_gaps(df)[-fvg_max:]
        all_fvgs    = auto_fvgs + (extra_fvgs or [])
        bull_legend = bear_legend = filled_legend = False

        for fvg in all_fvgs:
            is_bull   = fvg["type"] == "bullish"
            is_filled = fvg.get("filled", False)
            is_manual = fvg.get("source", "auto") == "manual"

            if is_filled and not show_filled: continue
            if is_bull and not show_bull_fvg: continue
            if not is_bull and not show_bear_fvg: continue

            if is_filled:
                fill_col, border_col, dash_style = "rgba(180,180,180,0.10)", "rgba(180,180,180,0.30)", "dot"
            elif is_bull:
                fill_col   = "rgba(0,200,100,0.15)"  if not is_manual else "rgba(0,230,120,0.22)"
                border_col = "rgba(0,200,100,0.60)"  if not is_manual else "rgba(0,255,140,0.85)"
                dash_style = "solid"
            else:
                fill_col   = "rgba(255,60,60,0.15)"  if not is_manual else "rgba(255,80,80,0.22)"
                border_col = "rgba(255,60,60,0.60)"  if not is_manual else "rgba(255,100,100,0.85)"
                dash_style = "solid"

            fig.add_shape(
                type="rect",
                x0=fvg["start_time"], x1=fvg["end_time"], xref="x",
                y0=fvg["bottom"],     y1=fvg["top"],       yref="y",
                fillcolor=fill_col,
                line=dict(color=border_col, width=1 if not is_manual else 1.5, dash=dash_style),
            )
            if show_fvg_labels:
                mid   = (fvg["top"] + fvg["bottom"]) / 2
                label = ("✏️ " if is_manual else "") + ("Bull FVG" if is_bull else "Bear FVG") + (" ✓" if is_filled else "")
                fig.add_annotation(
                    x=fvg["start_time"], xref="x", y=mid, yref="y",
                    text=label, showarrow=False,
                    font=dict(color="rgba(0,230,120,0.9)" if is_bull else "rgba(255,100,100,0.9)", size=9),
                    xanchor="left", bgcolor="rgba(0,0,0,0.35)",
                )

            if is_filled and not filled_legend:
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color="rgba(180,180,180,0.4)", symbol="square"),
                    name="Filled FVG", xaxis="x", yaxis="y")); filled_legend = True
            elif is_bull and not bull_legend and not is_filled:
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color="rgba(0,200,100,0.6)", symbol="square"),
                    name="Bullish FVG", xaxis="x", yaxis="y")); bull_legend = True
            elif not is_bull and not bear_legend and not is_filled:
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color="rgba(255,60,60,0.6)", symbol="square"),
                    name="Bearish FVG", xaxis="x", yaxis="y")); bear_legend = True

    if show_vp:
        vp = compute_volume_profile(df, bins=vp_bins)
        if vp is not None:
            prices  = vp["price_levels"]
            volumes = vp["volumes"]
            poc, vah, val = vp["poc_price"], vp["vah_price"], vp["val_price"]
            max_vol = volumes.max() if volumes.max() > 0 else 1

            bar_colours = []
            for p in prices:
                if abs(p - poc) < (prices[1] - prices[0]) * 0.6:
                    bar_colours.append("rgba(255,220,50,0.90)")
                elif val <= p <= vah:
                    bar_colours.append("rgba(50,200,180,0.55)")
                else:
                    bar_colours.append("rgba(160,160,160,0.35)")

            fig.add_trace(go.Bar(
                x=volumes / max_vol, y=prices,
                orientation="h", width=(prices[-1] - prices[0]) / vp_bins * 0.85,
                marker_color=bar_colours, name="Volume Profile",
                xaxis="x2", yaxis="y",
                hovertemplate="Price: %{y:,.2f}<br>Volume: %{customdata:,.0f}<extra></extra>",
                customdata=volumes, showlegend=True,
            ))

            if show_poc:
                fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                    y0=poc, y1=poc, yref="y",
                    line=dict(color="rgba(255,220,50,0.85)", width=1.5, dash="dot"))
                fig.add_annotation(x=1, xref="paper", y=poc, yref="y",
                    text=f" POC {poc:,.2f}", showarrow=False,
                    font=dict(color="rgba(255,220,50,0.95)", size=11), xanchor="left")

            if show_value_area:
                fig.add_annotation(x=1, xref="paper", y=vah, yref="y",
                    text=f" VAH {vah:,.2f}", showarrow=False,
                    font=dict(color="rgba(50,200,180,0.9)", size=10), xanchor="left")
                fig.add_annotation(x=1, xref="paper", y=val, yref="y",
                    text=f" VAL {val:,.2f}", showarrow=False,
                    font=dict(color="rgba(50,200,180,0.9)", size=10), xanchor="left")

    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.04),
            domain=[0, 0.82],
            showticklabels=True,
        ),
        xaxis2=dict(domain=[0.83, 1.0], showgrid=False, showticklabels=False,
                    zeroline=False, range=[0, 1.05], fixedrange=True, autorange="reversed"),
        xaxis3=dict(domain=[0, 0.82], matches="x", showgrid=False),
        yaxis=dict(side="right", domain=[0.25, 1.0], fixedrange=False),
        yaxis3=dict(domain=[0.0, 0.20], showgrid=False, showticklabels=False, zeroline=False, fixedrange=True),
        height=680, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(r=120, b=60, t=40), bargap=0,
        dragmode="pan",
    )

    chart_config = dict(
        scrollZoom=True,
        displayModeBar=True,
        modeBarButtonsToRemove=["select2d", "lasso2d", "autoScale2d"],
        modeBarButtonsToAdd=["pan2d", "zoom2d", "resetScale2d", "resetViews"],
        displaylogo=False,
    )

    # Streamlit reset button above chart — bumping the key forces a full re-render
    # which resets pan/zoom back to the default view
    rc1, rc2 = st.columns([8, 1])
    with rc2:
        if st.button("↺ Reset view", use_container_width=True, key="reset_chart_view"):
            st.session_state["chart_render_key"] = st.session_state.get("chart_render_key", 0) + 1

    chart_key = f"main_chart_{st.session_state.get('chart_render_key', 0)}"

    # Use on_select when R:R mode is active so clicks set price levels
    rr_mode = st.session_state.get("rr_state", {}).get("mode")
    if show_rr and rr_mode:
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                                config=chart_config,
                                key=f"rr_chart_{st.session_state.get('rr_chart_key',0)}")
        if event and event.selection and event.selection.get("points"):
            clicked_y = event.selection["points"][0].get("y")
            if clicked_y is not None:
                st.session_state.rr_state[rr_mode] = float(clicked_y)
                st.session_state.rr_state["mode"]  = None
                st.session_state["rr_chart_key"]   = st.session_state.get("rr_chart_key", 0) + 1
                st.rerun()
    else:
        st.plotly_chart(fig, use_container_width=True, config=chart_config, key=chart_key)


# ============================================================
# RISK / REWARD TOOL
# ============================================================
def rr_tool_ui(df: pd.DataFrame) -> dict:
    mid = float(df["Close"].iloc[-1])
    price_range = float(df["High"].max()) - float(df["Low"].min())
    nudge_range = price_range * 0.02

    if "rr_state" not in st.session_state:
        st.session_state.rr_state = {
            "entry": mid,
            "sl":    mid * 0.98,
            "tp":    mid * 1.04,
            "mode":  None,
        }
    if "rr_chart_key" not in st.session_state:
        st.session_state.rr_chart_key = 0

    state = st.session_state.rr_state
    price_fmt = "%.2f" if mid > 10 else "%.6f"

    st.markdown("#### ⚖️ Risk / Reward Tool")
    st.caption("Click a mode button, then click the chart OR type a price directly.")

    # ---- Mode buttons ----
    b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
    with b1:
        entry_active = state["mode"] == "entry"
        if st.button("🎯 Set Entry" if not entry_active else "🎯 Entry ✓",
                     use_container_width=True,
                     type="primary" if entry_active else "secondary"):
            state["mode"] = None if entry_active else "entry"
            st.rerun()
    with b2:
        sl_active = state["mode"] == "sl"
        if st.button("🔴 Set SL" if not sl_active else "🔴 SL ✓",
                     use_container_width=True,
                     type="primary" if sl_active else "secondary"):
            state["mode"] = None if sl_active else "sl"
            st.rerun()
    with b3:
        tp_active = state["mode"] == "tp"
        if st.button("🟢 Set TP" if not tp_active else "🟢 TP ✓",
                     use_container_width=True,
                     type="primary" if tp_active else "secondary"):
            state["mode"] = None if tp_active else "tp"
            st.rerun()
    with b4:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.rr_state = {
                "entry": mid, "sl": mid * 0.98, "tp": mid * 1.04, "mode": None
            }
            st.session_state.rr_chart_key += 1
            st.rerun()

    if state["mode"]:
        st.info(f"Click anywhere on the chart to set **{state['mode'].upper()}**, or type a price below.")

    # ---- Direct price inputs (always visible) ----
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        new_entry = st.number_input("Entry price", value=float(state["entry"]),
                                     format=price_fmt, key="rr_entry_input",
                                     label_visibility="visible")
        if new_entry != state["entry"]:
            state["entry"] = new_entry
            if state["mode"] == "entry":
                state["mode"] = None

    with pc2:
        new_sl = st.number_input("Stop loss", value=float(state["sl"]),
                                  format=price_fmt, key="rr_sl_input",
                                  label_visibility="visible")
        if new_sl != state["sl"]:
            state["sl"] = new_sl
            if state["mode"] == "sl":
                state["mode"] = None

    with pc3:
        new_tp = st.number_input("Take profit", value=float(state["tp"]),
                                  format=price_fmt, key="rr_tp_input",
                                  label_visibility="visible")
        if new_tp != state["tp"]:
            state["tp"] = new_tp
            if state["mode"] == "tp":
                state["mode"] = None

    # ---- Fine-tune sliders ----
    with st.expander("Fine-tune with sliders", expanded=False):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            state["entry"] = st.slider(
                "Entry", min_value=float(state["entry"] - nudge_range),
                max_value=float(state["entry"] + nudge_range),
                value=float(state["entry"]), format=price_fmt, key="rr_entry_slider")
        with sc2:
            state["sl"] = st.slider(
                "SL", min_value=float(state["sl"] - nudge_range),
                max_value=float(state["sl"] + nudge_range),
                value=float(state["sl"]), format=price_fmt, key="rr_sl_slider")
        with sc3:
            state["tp"] = st.slider(
                "TP", min_value=float(state["tp"] - nudge_range),
                max_value=float(state["tp"] + nudge_range),
                value=float(state["tp"]), format=price_fmt, key="rr_tp_slider")

    # ---- Metrics ----
    entry  = state["entry"]
    sl     = state["sl"]
    tp     = state["tp"]
    risk   = abs(entry - sl)
    reward = abs(tp - entry)
    rr     = reward / risk if risk > 0 else 0
    risk_dollars   = (rr_risk_pct / 100) * rr_account_size
    position_size  = risk_dollars / risk if risk > 0 else 0
    potential_gain = reward * position_size

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R:R Ratio",      f"1 : {rr:.2f}",
              delta="Good" if rr >= 2 else "Low",
              delta_color="normal" if rr >= 2 else "inverse")
    m2.metric("Risk $",         f"${risk_dollars:,.2f}")
    m3.metric("Position Size",  f"{position_size:,.4f} units")
    m4.metric("Potential Gain", f"${potential_gain:,.2f}")

    st.session_state.rr_state = state
    return state


# ============================================================
# MANUAL FVG MANAGER
# ============================================================
def manual_fvg_ui(df: pd.DataFrame) -> list:
    if "manual_fvgs" not in st.session_state:
        st.session_state.manual_fvgs = []

    st.markdown("#### ✏️ Manual Fair Value Gaps")

    price_min = float(df["Low"].min())
    price_max = float(df["High"].max())
    mid_price = (price_min + price_max) / 2.0
    gap_size  = (price_max - price_min) * 0.01

    with st.expander("➕ Add Manual FVG", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            m_type   = st.selectbox("Type", ["bullish", "bearish"], key="mfvg_type")
            m_top    = st.number_input("Top of gap",    value=round(mid_price + gap_size, 4), format="%.4f", key="mfvg_top")
            m_bottom = st.number_input("Bottom of gap", value=round(mid_price,             4), format="%.4f", key="mfvg_bottom")
        with c2:
            total_candles = len(df)
            start_idx, end_idx = st.select_slider(
                "Candle range (start → end)",
                options=list(range(total_candles)),
                value=(max(0, total_candles - 20), total_candles - 1),
                key="mfvg_range"
            )
            st.caption(f"Start: {df['Datetime'].iloc[start_idx].strftime('%Y-%m-%d %H:%M')}")
            st.caption(f"End:   {df['Datetime'].iloc[end_idx].strftime('%Y-%m-%d %H:%M')}")

        if st.button("Add FVG", key="mfvg_add"):
            if m_top <= m_bottom:
                st.error("Top must be greater than Bottom.")
            elif start_idx >= end_idx:
                st.error("Start candle must be before end candle.")
            else:
                st.session_state.manual_fvgs.append({
                    "type":       m_type,
                    "top":        m_top,
                    "bottom":     m_bottom,
                    "start_time": safe_isoformat(df["Datetime"].iloc[start_idx]),
                    "end_time":   safe_isoformat(df["Datetime"].iloc[end_idx]),
                    "filled":     False,
                    "source":     "manual",
                })
                st.success("Manual FVG added!")
                st.rerun()

    if st.session_state.manual_fvgs:
        st.markdown("**Existing Manual FVGs**")
        for i, fvg in enumerate(st.session_state.manual_fvgs):
            with st.expander(
                f"{'🟢' if fvg['type']=='bullish' else '🔴'} FVG #{i+1}  "
                f"| {fvg['bottom']:,.4f} – {fvg['top']:,.4f}  "
                f"{'✓ filled' if fvg['filled'] else ''}",
                expanded=False
            ):
                ec1, ec2 = st.columns(2)
                with ec1:
                    new_type   = st.selectbox("Type", ["bullish", "bearish"],
                                              index=0 if fvg["type"] == "bullish" else 1,
                                              key=f"edit_type_{i}")
                    new_top    = st.number_input("Top",    value=float(fvg["top"]),
                                                 format="%.4f", key=f"edit_top_{i}")
                    new_bottom = st.number_input("Bottom", value=float(fvg["bottom"]),
                                                 format="%.4f", key=f"edit_bot_{i}")
                with ec2:
                    total_candles = len(df)
                    def closest_idx(iso_str):
                        target = pd.Timestamp(iso_str)
                        diffs  = (df["Datetime"] - target).abs()
                        return int(diffs.argmin())
                    cur_start = closest_idx(fvg["start_time"])
                    cur_end   = closest_idx(fvg["end_time"])
                    new_start_idx, new_end_idx = st.select_slider(
                        "Candle range",
                        options=list(range(total_candles)),
                        value=(cur_start, cur_end),
                        key=f"edit_range_{i}"
                    )
                    st.caption(f"Start: {df['Datetime'].iloc[new_start_idx].strftime('%Y-%m-%d %H:%M')}")
                    st.caption(f"End:   {df['Datetime'].iloc[new_end_idx].strftime('%Y-%m-%d %H:%M')}")
                    new_filled = st.checkbox("Mark as filled", value=fvg["filled"], key=f"edit_filled_{i}")

                col_save, col_del = st.columns(2)
                with col_save:
                    if st.button("💾 Save Changes", key=f"save_fvg_{i}", use_container_width=True):
                        if new_top <= new_bottom:
                            st.error("Top must be greater than Bottom.")
                        else:
                            st.session_state.manual_fvgs[i] = {
                                "type":       new_type,
                                "top":        new_top,
                                "bottom":     new_bottom,
                                "start_time": safe_isoformat(df["Datetime"].iloc[new_start_idx]),
                                "end_time":   safe_isoformat(df["Datetime"].iloc[new_end_idx]),
                                "filled":     new_filled,
                                "source":     "manual",
                            }
                            st.success("Updated!")
                            st.rerun()
                with col_del:
                    if st.button("🗑️ Delete", key=f"del_fvg_{i}", use_container_width=True):
                        st.session_state.manual_fvgs.pop(i)
                        st.rerun()
    else:
        st.caption("No manual FVGs yet.")

    return st.session_state.manual_fvgs


# ============================================================
# BACKTEST MODE
# ============================================================
if app_mode == "📊 Backtest":
    df_raw = load_data(ticker, timeline, time_frame)
    warn_if_clamped(ticker, time_frame, timeline)

    if df_raw.empty:
        st.error("No data found. For crypto use tickers like BTC-USD, ETH-USD, SOL-USD. For stocks use AAPL, TSLA etc.")
        st.info("Supported crypto: " + ", ".join(CRYPTO_MAP.keys()))
    else:
        df_filtered = filter_trading_hours(df_raw, ticker, time_frame)
        closed_gaps = get_closed_gaps(df_raw, df_filtered, ticker, time_frame)
        df = compute_emas(df_filtered, fast_period, slow_period)

        cash, position, is_invested = initial_capital, 0.0, False
        trade_log = []
        buy_signals_x, buy_signals_y   = [], []
        sell_signals_x, sell_signals_y = [], []

        for idx, row in df.iterrows():
            if idx < slow_period:
                continue
            current_price = float(row["Close"])
            fast_ma, slow_ma = float(row["Fast_EMA"]), float(row["Slow_EMA"])
            timestamp = row["Datetime"]

            if not is_invested and fast_ma > slow_ma:
                position = cash / current_price; cash = 0.0; is_invested = True
                trade_log.append({"Action": "BUY",  "Time": timestamp, "Price": round(current_price, 2), "Net Worth": round(position * current_price, 2)})
                buy_signals_x.append(timestamp); buy_signals_y.append(current_price)
            elif is_invested and fast_ma < slow_ma:
                cash = position * current_price; position = 0.0; is_invested = False
                trade_log.append({"Action": "SELL", "Time": timestamp, "Price": round(current_price, 2), "Net Worth": round(cash, 2)})
                sell_signals_x.append(timestamp); sell_signals_y.append(current_price)

        final_price  = float(df.iloc[-1]["Close"])
        final_value  = cash if not is_invested else position * final_price
        total_return = ((final_value - initial_capital) / initial_capital) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Starting Capital",     f"${initial_capital:,.2f}")
        col2.metric("Ending Net Worth",     f"${final_value:,.2f}")
        col3.metric("Total Return",         f"{total_return:+.2f}%", delta_color="normal")
        col4.metric("Total Trades Executed", len(trade_log))
        st.markdown("---")
        st.subheader("📊 Interactive Market Chart & Execution Flags")

        manual_fvgs = manual_fvg_ui(df)
        if show_rr:
            rr_tool_ui(df)
        render_chart(df, buy_signals_x, buy_signals_y, sell_signals_x, sell_signals_y,
                     fast_period, slow_period, extra_fvgs=manual_fvgs, closed_gaps=closed_gaps)

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
    sim_mode   = st.radio("Simulation Type", ["📡 Paper Trade (Live Prices)", "🔁 Historical Replay"], horizontal=True)
    auto_trade = st.toggle("🤖 Auto-trade on EMA signals", value=False)

    if "sim_cash" not in st.session_state:
        st.session_state.sim_cash     = initial_capital
        st.session_state.sim_position = 0.0
        st.session_state.sim_invested = False
        st.session_state.sim_trades   = []
        st.session_state.sim_buy_x    = []; st.session_state.sim_buy_y  = []
        st.session_state.sim_sell_x   = []; st.session_state.sim_sell_y = []

    def sim_buy(price, timestamp):
        if not st.session_state.sim_invested and st.session_state.sim_cash > 0:
            st.session_state.sim_position = st.session_state.sim_cash / price
            st.session_state.sim_cash = 0.0; st.session_state.sim_invested = True
            st.session_state.sim_trades.append({"Action": "BUY", "Time": str(timestamp), "Price": round(price, 2), "Net Worth": round(st.session_state.sim_position * price, 2)})
            st.session_state.sim_buy_x.append(timestamp); st.session_state.sim_buy_y.append(price)

    def sim_sell(price, timestamp):
        if st.session_state.sim_invested:
            st.session_state.sim_cash = st.session_state.sim_position * price
            st.session_state.sim_position = 0.0; st.session_state.sim_invested = False
            st.session_state.sim_trades.append({"Action": "SELL", "Time": str(timestamp), "Price": round(price, 2), "Net Worth": round(st.session_state.sim_cash, 2)})
            st.session_state.sim_sell_x.append(timestamp); st.session_state.sim_sell_y.append(price)

    if sim_mode == "📡 Paper Trade (Live Prices)":
        st.subheader("📡 Paper Trading with Live Prices")
        live_price = get_live_price(ticker)
        if live_price:
            st.metric("Current Live Price", f"${live_price:,.2f}")
            df_raw_sim   = load_data(ticker, "1mo", time_frame)
            df_filt_sim  = filter_trading_hours(df_raw_sim, ticker, time_frame)
            closed_gaps  = get_closed_gaps(df_raw_sim, df_filt_sim, ticker, time_frame)
            df_chart     = compute_emas(df_filt_sim, fast_period, slow_period)
            warn_if_clamped(ticker, time_frame, "1mo")
            if auto_trade and len(df_chart) > slow_period:
                fast_now, slow_now = float(df_chart["Fast_EMA"].iloc[-1]), float(df_chart["Slow_EMA"].iloc[-1])
                now = datetime.now()
                if not st.session_state.sim_invested and fast_now > slow_now:
                    sim_buy(live_price, now); st.success(f"🤖 Auto-BUY triggered at ${live_price:,.2f}")
                elif st.session_state.sim_invested and fast_now < slow_now:
                    sim_sell(live_price, now); st.success(f"🤖 Auto-SELL triggered at ${live_price:,.2f}")
            if not auto_trade:
                col_b, col_s, col_r = st.columns(3)
                with col_b:
                    if st.button("🟢 Manual BUY", use_container_width=True):
                        sim_buy(live_price, datetime.now()); st.success(f"Bought at ${live_price:,.2f}")
                with col_s:
                    if st.button("🔴 Manual SELL", use_container_width=True):
                        sim_sell(live_price, datetime.now()); st.success(f"Sold at ${live_price:,.2f}")
                with col_r:
                    if st.button("🔄 Reset", use_container_width=True):
                        for k in ["sim_cash","sim_position","sim_invested","sim_trades","sim_buy_x","sim_buy_y","sim_sell_x","sim_sell_y"]:
                            del st.session_state[k]
                        st.rerun()
            if not df_chart.empty:
                manual_fvgs = manual_fvg_ui(df_chart)
                if show_rr:
                    rr_tool_ui(df_chart)
                render_chart(df_chart, st.session_state.sim_buy_x, st.session_state.sim_buy_y,
                             st.session_state.sim_sell_x, st.session_state.sim_sell_y,
                             fast_period, slow_period, extra_fvgs=manual_fvgs, closed_gaps=closed_gaps)
        else:
            st.error("Could not fetch live price. Check ticker symbol.")

    else:
        st.subheader("🔁 Historical Data Replay")
        df_raw_rep  = load_data(ticker, timeline, time_frame)
        df_filt_rep = filter_trading_hours(df_raw_rep, ticker, time_frame)
        df_replay   = compute_emas(df_filt_rep, fast_period, slow_period)
        warn_if_clamped(ticker, time_frame, timeline)

        if "replay_idx" not in st.session_state:
            st.session_state.replay_idx = slow_period

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶️ Step Forward 1 Candle"):
                if st.session_state.replay_idx < len(df_replay) - 1:
                    st.session_state.replay_idx += 1
        with col_b:
            if st.button("⏩ Step Forward 10 Candles"):
                st.session_state.replay_idx = min(st.session_state.replay_idx + 10, len(df_replay) - 1)

        idx           = st.session_state.replay_idx
        df_visible    = df_replay.iloc[:idx+1].copy()
        current_price = float(df_visible["Close"].iloc[-1])
        timestamp     = df_visible["Datetime"].iloc[-1]
        st.metric("Current Replay Price", f"${current_price:,.2f}", f"Candle {idx} of {len(df_replay)}")

        # Recompute gaps for the visible slice
        df_raw_vis   = df_raw_rep.iloc[:idx+1].copy() if not df_raw_rep.empty else pd.DataFrame()
        closed_gaps  = get_closed_gaps(df_raw_vis, df_visible, ticker, time_frame)

        if auto_trade and len(df_visible) > slow_period:
            fast_now, slow_now = float(df_visible["Fast_EMA"].iloc[-1]), float(df_visible["Slow_EMA"].iloc[-1])
            if not st.session_state.sim_invested and fast_now > slow_now:
                sim_buy(current_price, timestamp)
            elif st.session_state.sim_invested and fast_now < slow_now:
                sim_sell(current_price, timestamp)
        if not auto_trade:
            col_b2, col_s2, col_r2 = st.columns(3)
            with col_b2:
                if st.button("🟢 BUY at Current Price", use_container_width=True):
                    sim_buy(current_price, timestamp)
            with col_s2:
                if st.button("🔴 SELL at Current Price", use_container_width=True):
                    sim_sell(current_price, timestamp)
            with col_r2:
                if st.button("🔄 Reset", use_container_width=True):
                    for k in ["sim_cash","sim_position","sim_invested","sim_trades","sim_buy_x","sim_buy_y","sim_sell_x","sim_sell_y","replay_idx"]:
                        del st.session_state[k]
                    st.rerun()

        manual_fvgs = manual_fvg_ui(df_visible)
        if show_rr:
            rr_tool_ui(df_visible)
        render_chart(df_visible, st.session_state.sim_buy_x, st.session_state.sim_buy_y,
                     st.session_state.sim_sell_x, st.session_state.sim_sell_y,
                     fast_period, slow_period, extra_fvgs=manual_fvgs, closed_gaps=closed_gaps)

    st.markdown("---")
    live_val     = get_live_price(ticker) or 0
    net_worth    = st.session_state.sim_cash if not st.session_state.sim_invested else st.session_state.sim_position * live_val
    total_return = ((net_worth - initial_capital) / initial_capital) * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Starting Capital", f"${initial_capital:,.2f}")
    c2.metric("Sim Net Worth",    f"${net_worth:,.2f}")
    c3.metric("Sim Return",       f"{total_return:+.2f}%")
    c4.metric("Sim Trades",       len(st.session_state.sim_trades))
    if st.session_state.sim_trades:
        st.subheader("📜 Simulation Trade Log")
        st.dataframe(pd.DataFrame(st.session_state.sim_trades), use_container_width=True)


# ============================================================
# ROBINHOOD LIVE
# ============================================================
elif app_mode == "🤖 Robinhood Live":
    st.header("🤖 Robinhood Live Trading")
    try:
        import robin_stocks.robinhood as r
        rh_available = True
    except ImportError:
        rh_available = False

    if not rh_available:
        st.error("robin_stocks is not installed.")
        st.code("python -m pip install robin_stocks", language="bash")
    else:
        st.warning("⚠️ This mode executes REAL trades with REAL money. Use with caution.")
        with st.expander("🔐 Robinhood Login", expanded="rh_logged_in" not in st.session_state):
            rh_user = st.text_input("Robinhood Email")
            rh_pass = st.text_input("Robinhood Password", type="password")
            if st.button("Login to Robinhood"):
                try:
                    r.login(rh_user, rh_pass)
                    st.session_state.rh_logged_in = True
                    st.success("✅ Logged in successfully!")
                except Exception as e:
                    st.error(f"Login failed: {e}")

        if st.session_state.get("rh_logged_in"):
            st.subheader("💼 Portfolio Overview")
            try:
                profile      = r.load_portfolio_profile()
                equity       = float(profile.get("equity", 0))
                buying_power = float(r.load_account_profile().get("buying_power", 0))
                col1, col2 = st.columns(2)
                col1.metric("Total Equity", f"${equity:,.2f}")
                col2.metric("Buying Power", f"${buying_power:,.2f}")
            except Exception as e:
                st.error(f"Could not load portfolio: {e}")

            st.subheader("📦 Current Positions")
            try:
                positions = r.get_open_stock_positions()
                if positions:
                    pos_data = []
                    for p in positions:
                        instrument = r.get_instrument_by_url(p["instrument"])
                        pos_data.append({"Symbol": instrument.get("symbol","N/A"), "Quantity": float(p.get("quantity",0)), "Avg Buy Price": float(p.get("average_buy_price",0))})
                    st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
                else:
                    st.info("No open positions.")
            except Exception as e:
                st.error(f"Could not load positions: {e}")

            st.markdown("---")
            live_price = get_live_price(ticker)
            if live_price:
                st.metric(f"Live Price: {ticker}", f"${live_price:,.2f}")

            df_raw_rh   = load_data(ticker, "1mo", time_frame)
            df_filt_rh  = filter_trading_hours(df_raw_rh, ticker, time_frame)
            closed_gaps = get_closed_gaps(df_raw_rh, df_filt_rh, ticker, time_frame)
            df_live     = compute_emas(df_filt_rh, fast_period, slow_period)
            warn_if_clamped(ticker, time_frame, "1mo")
            fast_now = float(df_live["Fast_EMA"].iloc[-1])
            slow_now = float(df_live["Slow_EMA"].iloc[-1])
            signal   = "🟢 BUY Signal" if fast_now > slow_now else "🔴 SELL Signal"
            st.subheader(f"EMA Signal: {signal}")

            rh_auto       = st.toggle("🤖 Auto-execute trades on EMA signal", value=False)
            crypto_symbol = ticker.replace("-USD", "")

            if rh_auto:
                st.warning("Auto-trading is ON.")
                if fast_now > slow_now:
                    if st.button("⚡ Execute AUTO-BUY"):
                        try:
                            order = r.order_buy_crypto_by_price(crypto_symbol, buying_power * 0.99)
                            st.success(f"✅ BUY placed! ID: {order.get('id')}")
                        except Exception as e:
                            st.error(f"Order failed: {e}")
                else:
                    if st.button("⚡ Execute AUTO-SELL"):
                        try:
                            for h in r.get_crypto_positions():
                                if h["currency"]["code"] == crypto_symbol:
                                    order = r.order_sell_crypto_by_quantity(crypto_symbol, float(h["quantity_available"]))
                                    st.success(f"✅ SELL placed! ID: {order.get('id')}")
                        except Exception as e:
                            st.error(f"Order failed: {e}")
            else:
                col_b, col_s = st.columns(2)
                with col_b:
                    buy_amount = st.number_input("Buy Amount ($)", min_value=1.0, value=100.0, step=10.0)
                    if st.button("🟢 Place BUY Order", use_container_width=True):
                        try:
                            order = r.order_buy_crypto_by_price(crypto_symbol, buy_amount)
                            st.success(f"✅ BUY placed for ${buy_amount}! ID: {order.get('id')}")
                        except Exception as e:
                            st.error(f"Order failed: {e}")
                with col_s:
                    sell_pct = st.slider("Sell % of Holdings", 1, 100, 100)
                    if st.button("🔴 Place SELL Order", use_container_width=True):
                        try:
                            for h in r.get_crypto_positions():
                                if h["currency"]["code"] == crypto_symbol:
                                    qty   = float(h["quantity_available"]) * (sell_pct / 100)
                                    order = r.order_sell_crypto_by_quantity(crypto_symbol, qty)
                                    st.success(f"✅ SELL placed! ID: {order.get('id')}")
                        except Exception as e:
                            st.error(f"Order failed: {e}")

            st.markdown("---")
            st.subheader("📊 Live Chart")
            if not df_live.empty:
                manual_fvgs = manual_fvg_ui(df_live)
                if show_rr:
                    rr_tool_ui(df_live)
                render_chart(df_live, [], [], [], [], fast_period, slow_period,
                             extra_fvgs=manual_fvgs, closed_gaps=closed_gaps)


# ============================================================
# RULES MANAGER
# ============================================================
elif app_mode == "📋 Rules Manager":

    RULES_FILE = "trading_rules.json"

    def load_all(file):
        if os.path.exists(file):
            with open(file, "r") as f:
                data = json.load(f)
            if data and isinstance(data[0], dict) and "rule" in data[0]:
                return [{"name": "Trading Rules", "rules": data}]
            return data
        return [
            {"name": "Chart Setup",   "rules": []},
            {"name": "Trading Rules", "rules": []},
        ]

    def save_all(data, file):
        with open(file, "w") as f:
            json.dump(data, f, indent=2)

    st.header("📋 Trading Rules Manager")
    st.markdown("Organize your rules into custom sections. Toggle, edit, or delete rules anytime.")

    if "rm_sections" not in st.session_state:
        st.session_state.rm_sections = load_all(RULES_FILE)

    sections = st.session_state.rm_sections

    st.subheader("➕ Add New Section")
    col_sec, col_sec_btn = st.columns([4, 1])
    with col_sec:
        new_section_name = st.text_input("Section name", placeholder="e.g. Risk Management, Entry Criteria...",
                                         key="new_section_input")
    with col_sec_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add Section", use_container_width=True):
            if new_section_name.strip():
                sections.append({"name": new_section_name.strip(), "rules": []})
                save_all(sections, RULES_FILE)
                st.rerun()

    st.markdown("---")

    if not sections:
        st.info("No sections yet. Add your first section above.")
    else:
        if "rm_open" not in st.session_state:
            st.session_state.rm_open = {}

        for s_idx, section in enumerate(sections):
            total        = len(section["rules"])
            active_count = sum(1 for r in section["rules"] if r["active"])
            sec_key      = f"sec_open_{s_idx}"
            is_open      = st.session_state.rm_open.get(sec_key, False)

            with st.expander(f"📁 {section['name']}  —  {active_count}/{total} active", expanded=is_open):
                st.session_state.rm_open[sec_key] = True

                col_rename, col_rename_btn, col_del = st.columns([4, 1, 1])
                with col_rename:
                    new_name = st.text_input("Rename section", value=section["name"], key=f"rename_{s_idx}")
                with col_rename_btn:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Rename", key=f"rename_btn_{s_idx}", use_container_width=True):
                        if new_name.strip():
                            sections[s_idx]["name"] = new_name.strip()
                            save_all(sections, RULES_FILE)
                            st.rerun()
                with col_del:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🗑️ Delete Section", key=f"del_sec_{s_idx}", use_container_width=True):
                        sections.pop(s_idx)
                        st.session_state.rm_open.pop(sec_key, None)
                        save_all(sections, RULES_FILE)
                        st.rerun()

                st.markdown("---")

                new_rule = st.text_input(
                    "New rule (press Enter to add)",
                    placeholder="e.g. Only enter when volume is above 20-period average",
                    key=f"new_rule_{s_idx}"
                )
                if new_rule.strip() and st.session_state.get(f"new_rule_{s_idx}") == new_rule:
                    last_key = f"last_added_{s_idx}"
                    if st.session_state.get(last_key) != new_rule:
                        sections[s_idx]["rules"].append({"rule": new_rule.strip(), "active": True})
                        st.session_state[last_key] = new_rule
                        save_all(sections, RULES_FILE)
                        st.rerun()

                st.markdown("---")

                if not section["rules"]:
                    st.caption("No rules in this section yet.")
                else:
                    for r_idx, rule in enumerate(section["rules"]):
                        col_chk, col_rule, col_edit, col_del2 = st.columns([0.5, 5, 1, 0.8])

                        with col_chk:
                            active = st.checkbox("", value=rule["active"], key=f"toggle_{s_idx}_{r_idx}")
                            if active != rule["active"]:
                                sections[s_idx]["rules"][r_idx]["active"] = active
                                save_all(sections, RULES_FILE)

                        with col_rule:
                            edit_key = f"editing_{s_idx}_{r_idx}"
                            if st.session_state.get(edit_key, False):
                                edited = st.text_input("Edit rule", value=rule["rule"],
                                                       key=f"edit_input_{s_idx}_{r_idx}",
                                                       label_visibility="collapsed")
                                if st.button("💾 Save", key=f"save_rule_{s_idx}_{r_idx}"):
                                    if edited.strip():
                                        sections[s_idx]["rules"][r_idx]["rule"] = edited.strip()
                                        save_all(sections, RULES_FILE)
                                    st.session_state[edit_key] = False
                                    st.rerun()
                            else:
                                status = "🟢" if rule["active"] else "🔴"
                                st.markdown(f"{status} {rule['rule']}")

                        with col_edit:
                            edit_key = f"editing_{s_idx}_{r_idx}"
                            label = "✏️ Cancel" if st.session_state.get(edit_key, False) else "✏️ Edit"
                            if st.button(label, key=f"edit_btn_{s_idx}_{r_idx}", use_container_width=True):
                                st.session_state[edit_key] = not st.session_state.get(edit_key, False)
                                st.rerun()

                        with col_del2:
                            if st.button("🗑️", key=f"del_rule_{s_idx}_{r_idx}", use_container_width=True):
                                sections[s_idx]["rules"].pop(r_idx)
                                save_all(sections, RULES_FILE)
                                st.rerun()

        st.markdown("---")
        st.subheader("✅ All Active Rules")
        any_active = False
        for section in sections:
            active_rules = [r for r in section["rules"] if r["active"]]
            if active_rules:
                any_active = True
                st.markdown(f"**{section['name']}**")
                for rule in active_rules:
                    st.markdown(f"- {rule['rule']}")
        if not any_active:
            st.info("No active rules across any section.")

        st.markdown("---")
        st.download_button(
            label="📥 Export All Rules as JSON",
            data=json.dumps(sections, indent=2),
            file_name="trading_rules.json",
            mime="application/json"
        )
