# ============================================================
# FAIR VALUE GAP CALCULATION
# ============================================================
def find_fair_value_gaps(df):
    fvg_list = []
    if len(df) < 3:
        return fvg_list

    for i in range(1, len(df) - 1):
        # Use strictly positional .iloc to prevent index errors
        c1_high = float(df['High'].iloc[i-1])
        c1_low = float(df['Low'].iloc[i-1])
        c3_high = float(df['High'].iloc[i+1])
        c3_low = float(df['Low'].iloc[i+1])
        
        # Skip if yfinance returned missing data for this row
        if pd.isna(c1_high) or pd.isna(c3_low):
            continue
            
        # Convert Timestamps to strings to prevent Plotly JSON crash
        t2 = str(df['Datetime'].iloc[i])
        t_end = str(df['Datetime'].iloc[-1])

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
