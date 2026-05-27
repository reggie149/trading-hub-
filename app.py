import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import requests
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
app_mode = st.sidebar.radio("Select Mode", ["📊 Backtest", "🧪 Simulation", "🤖 Robinhood Live"])

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
    "MATIC-USD": "matic-network",
    "AVAX-USD": "avalanche-2",
    "DOT-USD": "polkadot",
}

PERIOD_DAYS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}

def is_crypto(symbol):
    return symbol.upper() in CRYPTO_MAP or symbol.upper().endswith("-USD")

def load_crypto_coingecko(symbol, period):
    coin_id = CRYPTO_MAP.get(symbol.upper())
    if not coin_id:
        # Try to guess from symbol like "BTC-USD" -> "bitcoin" won't work, show error
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

def render_chart(df, buy_x, buy_y, sell_x, sell_y, fast_period, slow_period):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['Datetime'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Price Action"
    ))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Fast_EMA'], line=dict(color='orange', width=1.5), name=f'{fast_period} Fast EMA'))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Slow_EMA'], line=dict(color='blue', width=1.5), name=f'{slow_period} Slow EMA'))
    if buy_x:
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=2, color='black')),
            name='BUY Entry'))
    if sell_x:
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=2, color='black')),
            name='SELL Exit'))
    fig.update_layout(xaxis_rangeslider_visible=False, height=550, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# BACKTEST MODE
# ============================================================
if app_mode == "📊 Backtest":
    df_raw = load_data(ticker, timeline, time_frame)

    if df_raw.empty:
        st.error("No data found. For crypto use tickers like BTC-USD, ETH-USD, SOL-USD. For stocks use AAPL, TSLA etc.")
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
                trade_log.append({"Action": "BUY", "Time": timestamp, "Price": round(current_price, 2), "Net Worth": round(position * current_price, 2)})
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
    sim_mode = st.radio("Simulation Type", ["📡 Paper Trade (Live Prices)", "🔁 Historical Replay"], horizontal=True)
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
            st.session_state.sim_trades.append({"Action": "BUY", "Time": str(timestamp), "Price": round(price, 2), "Net Worth": round(st.session_state.sim_position * price, 2)})
            st.session_state.sim_buy_x.append(timestamp)
            st.session_state.sim_buy_y.append(price)

    def sim_sell(price, timestamp):
        if st.session_state.sim_invested:
            st.session_state.sim_cash = st.session_state.sim_position * price
            st.session_state.sim_position = 0.0
            st.session_state.sim_invested = False
            st.session_state.sim_trades.append({"Action": "SELL", "Time": str(timestamp), "Price": round(price, 2), "Net Worth": round(st.session_state.sim_cash, 2)})
            st.session_state.sim_sell_x.append(timestamp)
            st.session_state.sim_sell_y.append(price)

    if sim_mode == "📡 Paper Trade (Live Prices)":
        st.subheader("📡 Paper Trading with Live Prices")
        live_price = get_live_price(ticker)

        if live_price:
            st.metric("Current Live Price", f"${live_price:,.2f}")
            df_chart = load_data(ticker, "1mo", "1h")
            df_chart = compute_emas(df_chart, fast_period, slow_period)

            if auto_trade and len(df_chart) > slow_period:
                fast_now = float(df_chart['Fast_EMA'].iloc[-1])
                slow_now = float(df_chart['Slow_EMA'].iloc[-1])
                now = datetime.now()
                if not st.session_state.sim_invested and fast_now > slow_now:
                    sim_buy(live_price, now)
                    st.success(f"🤖 Auto-BUY triggered at ${live_price:,.2f}")
                elif st.session_state.sim_invested and fast_now < slow_now:
                    sim_sell(live_price, now)
                    st.success(f"🤖 Auto-SELL triggered at ${live_price:,.2f}")

            if not auto_trade:
                col_b, col_s, col_r = st.columns(3)
                with col_b:
                    if st.button("🟢 Manual BUY", use_container_width=True):
                        sim_buy(live_price, datetime.now())
                        st.success(f"Bought at ${live_price:,.2f}")
                with col_s:
                    if st.button("🔴 Manual SELL", use_container_width=True):
                        sim_sell(live_price, datetime.now())
                        st.success(f"Sold at ${live_price:,.2f}")
                with col_r:
                    if st.button("🔄 Reset", use_container_width=True):
                        for key in ['sim_cash','sim_position','sim_invested','sim_trades','sim_buy_x','sim_buy_y','sim_sell_x','sim_sell_y']:
                            del st.session_state[key]
                        st.rerun()

            if not df_chart.empty:
                render_chart(df_chart, st.session_state.sim_buy_x, st.session_state.sim_buy_y,
                             st.session_state.sim_sell_x, st.session_state.sim_sell_y, fast_period, slow_period)
        else:
            st.error("Could not fetch live price. Check ticker symbol.")

    else:
        st.subheader("🔁 Historical Data Replay")
        df_replay = load_data(ticker, timeline, time_frame)
        df_replay = compute_emas(df_replay, fast_period, slow_period)

        if 'replay_idx' not in st.session_state:
            st.session_state.replay_idx = slow_period

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶️ Step Forward 1 Candle"):
                if st.session_state.replay_idx < len(df_replay) - 1:
                    st.session_state.replay_idx += 1
        with col_b:
            if st.button("⏩ Step Forward 10 Candles"):
                st.session_state.replay_idx = min(st.session_state.replay_idx + 10, len(df_replay) - 1)

        idx = st.session_state.replay_idx
        df_visible = df_replay.iloc[:idx+1].copy()
        current_price = float(df_visible['Close'].iloc[-1])
        timestamp = df_visible['Datetime'].iloc[-1]

        st.metric("Current Replay Price", f"${current_price:,.2f}", f"Candle {idx} of {len(df_replay)}")

        if auto_trade and len(df_visible) > slow_period:
            fast_now = float(df_visible['Fast_EMA'].iloc[-1])
            slow_now = float(df_visible['Slow_EMA'].iloc[-1])
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
                    for key in ['sim_cash','sim_position','sim_invested','sim_trades','sim_buy_x','sim_buy_y','sim_sell_x','sim_sell_y','replay_idx']:
                        del st.session_state[key]
                    st.rerun()

        render_chart(df_visible, st.session_state.sim_buy_x, st.session_state.sim_buy_y,
                     st.session_state.sim_sell_x, st.session_state.sim_sell_y, fast_period, slow_period)

    st.markdown("---")
    live_val = get_live_price(ticker) or 0
    net_worth = st.session_state.sim_cash if not st.session_state.sim_invested else (st.session_state.sim_position * live_val)
    total_return = ((net_worth - initial_capital) / initial_capital) * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Starting Capital", f"${initial_capital:,.2f}")
    c2.metric("Sim Net Worth", f"${net_worth:,.2f}")
    c3.metric("Sim Return", f"{total_return:+.2f}%")
    c4.metric("Sim Trades", len(st.session_state.sim_trades))
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
        with st.expander("🔐 Robinhood Login", expanded='rh_logged_in' not in st.session_state):
            rh_user = st.text_input("Robinhood Email")
            rh_pass = st.text_input("Robinhood Password", type="password")
            if st.button("Login to Robinhood"):
                try:
                    r.login(rh_user, rh_pass)
                    st.session_state.rh_logged_in = True
                    st.success("✅ Logged in successfully!")
                except Exception as e:
                    st.error(f"Login failed: {e}")

        if st.session_state.get('rh_logged_in'):
            st.subheader("💼 Portfolio Overview")
            try:
                profile = r.load_portfolio_profile()
                equity = float(profile.get('equity', 0))
                buying_power = float(r.load_account_profile().get('buying_power', 0))
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
                        instrument = r.get_instrument_by_url(p['instrument'])
                        pos_data.append({"Symbol": instrument.get('symbol', 'N/A'), "Quantity": float(p.get('quantity', 0)), "Avg Buy Price": float(p.get('average_buy_price', 0))})
                    st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
                else:
                    st.info("No open positions.")
            except Exception as e:
                st.error(f"Could not load positions: {e}")

            st.markdown("---")
            live_price = get_live_price(ticker)
            if live_price:
                st.metric(f"Live Price: {ticker}", f"${live_price:,.2f}")

            df_live = load_data(ticker, "1mo", "1h")
            df_live = compute_emas(df_live, fast_period, slow_period)
            fast_now = float(df_live['Fast_EMA'].iloc[-1])
            slow_now = float(df_live['Slow_EMA'].iloc[-1])
            signal = "🟢 BUY Signal" if fast_now > slow_now else "🔴 SELL Signal"
            st.subheader(f"EMA Signal: {signal}")

            rh_auto = st.toggle("🤖 Auto-execute trades on EMA signal", value=False)
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
                            holdings = r.get_crypto_positions()
                            for h in holdings:
                                if h['currency']['code'] == crypto_symbol:
                                    qty = float(h['quantity_available'])
                                    order = r.order_sell_crypto_by_quantity(crypto_symbol, qty)
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
                            holdings = r.get_crypto_positions()
                            for h in holdings:
                                if h['currency']['code'] == crypto_symbol:
                                    qty = float(h['quantity_available']) * (sell_pct / 100)
                                    order = r.order_sell_crypto_by_quantity(crypto_symbol, qty)
                                    st.success(f"✅ SELL placed! ID: {order.get('id')}")
                        except Exception as e:
                            st.error(f"Order failed: {e}")

            st.markdown("---")
            st.subheader("📊 Live Chart")
            if not df_live.empty:
                render_chart(df_live, [], [], [], [], fast_period, slow_period)
