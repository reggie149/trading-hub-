import pandas as pd
import yfinance as yf

class RealDataEngine:
    """Reaches out to the internet to grab real-world historical financial data."""
    @staticmethod
    def get_crypto_history(ticker="BTC-USD", period="1mo", interval="1h"):
        print(f"📡 Fetching real-world history for {ticker} from Yahoo Finance...")
        # Downloads data and forces a clean single-level column structure
        df = yf.download(ticker, period=period, interval=interval)
        
        # Flatten multi-level columns if Yahoo Finance sends them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.reset_index(inplace=True)
        df['Tick'] = df.index + 1
        return df

class BacktestEngine:
    """Runs our Exponential Moving Average Crossover over the real historical chart."""
    def __init__(self, initial_capital=10000.00, fast_period=12, slow_period=26):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0.0
        self.is_invested = False
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trade_log = []

    def run_backtest(self, df):
        # Calculate standard MACD style Exponential Moving Averages
        df['Fast_EMA'] = df['Close'].ewm(span=self.fast_period, adjust=False).mean()
        df['Slow_EMA'] = df['Close'].ewm(span=self.slow_period, adjust=False).mean()
        
        print("\n" + "="*60)
        print(f" PROCESSING REAL MARKET DATA ({len(df)} 1-Hour Candles) ")
        print("="*60)

        for idx, row in df.iterrows():
            # Safely extract float values from the series
            current_price = float(row['Close'].iloc[0]) if isinstance(row['Close'], pd.Series) else float(row['Close'])
            fast_ma = float(row['Fast_EMA'].iloc[0]) if isinstance(row['Fast_EMA'], pd.Series) else float(row['Fast_EMA'])
            slow_ma = float(row['Slow_EMA'].iloc[0]) if isinstance(row['Slow_EMA'], pd.Series) else float(row['Slow_EMA'])
            timestamp = str(row['Datetime']) if 'Datetime' in df.columns else str(row['Tick'])
            
            if idx < self.slow_period:
                continue

            # BUY TRIGGER: Fast EMA crosses above Slow EMA
            if not self.is_invested and fast_ma > slow_ma:
                self.position = self.cash / current_price
                self.cash = 0.0
                self.is_invested = True
                self.trade_log.append({"type": "BUY", "price": current_price})
                print(f"📈 [{timestamp[:16]}] BUY Bitcoin at ${current_price:,.2f}")

            # SELL TRIGGER: Fast EMA crosses below Slow EMA
            elif self.is_invested and fast_ma < slow_ma:
                revenue = self.position * current_price
                self.cash = revenue
                self.position = 0.0
                self.is_invested = False
                self.trade_log.append({"type": "SELL", "price": current_price})
                print(f"📉 [{timestamp[:16]}] SELL Bitcoin at ${current_price:,.2f}")

        # Final settlement value calculation
        final_row = df.iloc[-1]
        final_price = float(final_row['Close'].iloc[0]) if isinstance(final_row['Close'], pd.Series) else float(final_row['Close'])
        final_value = self.cash if not self.is_invested else (self.position * final_price)
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        print("\n" + "="*60)
        print("                     REAL-WORLD PERFORMANCE                 ")
        print("="*60)
        print(f"Asset Tested:       Bitcoin (BTC-USD)")
        print(f"Starting Balance:   ${self.initial_capital:,.2f}")
        print(f"Ending Net Worth:   ${final_value:,.2f}")
        print(f"Total Return:       {total_return:+.2f}%")
        print(f"Total Trades Run:   {len(self.trade_log)}")
        print("="*60 + "\n")

if __name__ == "__main__":
    # 1. Download real 1-hour candles for the past 1 month
    real_candles = RealDataEngine.get_crypto_history(ticker="BTC-USD", period="1mo", interval="1h")
    
    # 2. Initialize the backtester
    tester = BacktestEngine(initial_capital=10000.00, fast_period=12, slow_period=26)
    
    # 3. Execute the strategy over real history
    tester.run_backtest(real_candles)