import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import linregress

st.set_page_config(layout="wide")

# -- Inertia function
def inertia(close_series, length=252):
    values = [np.nan] * (length - 1)
    for i in range(length - 1, len(close_series)):
        y = close_series.iloc[i - length + 1:i + 1]
        if y.isna().any():
            values.append(np.nan)
            continue
        x = np.arange(length)
        slope, intercept, *_ = linregress(x, y)
        projected = intercept + slope * (length - 1)
        values.append(projected)
    return pd.Series(values, index=close_series.index)

# -- Ticker selection
tickers = ["SPY", "QQQ", "SCHD", "VICI", "KO", "TGT", "TSM", "O", "LOW", "HSY", 
           "AAPL", "TSLA", "SBUX", "ADC", "V", "CVX", "MSFT", "LMT", "MSTX"]
selected_ticker = st.selectbox("Choose a ticker", tickers)

# -- Data download
rolling_window = 252
df = yf.download(selected_ticker, period='2y', interval='1d', auto_adjust=False)
df.columns = [f"{col[1]}_{col[0]}" for col in df.columns]  # Flatten MultiIndex

# -- Column name for Close
close_col = f"{selected_ticker}_Close"
df["Mean"] = inertia(df[close_col], length=rolling_window)
df["STD"] = df[close_col].rolling(window=rolling_window).std(ddof=0)

# -- SE bands
for i in range(-4, 5):
    df[f"SE_{i}"] = df["Mean"] + i * df["STD"]

# -- EMAs
df["EMA_10"] = df[close_col].ewm(span=10).mean()
df["EMA_50"] = df[close_col].ewm(span=50).mean()

# -- Filter to 1 year
df = df[df.index >= df.index.max() - pd.DateOffset(years=1)]

# -- Plotting
fig = go.Figure()

# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df[f"{selected_ticker}_Open"],
    high=df[f"{selected_ticker}_High"],
    low=df[f"{selected_ticker}_Low"],
    close=df[f"{selected_ticker}_Close"],
    name="Candlestick"
))

# SE bands
colors = {
    -4: "purple", 4: "purple",
    -3: "turquoise", 3: "turquoise",
    -2: "limegreen", 2: "limegreen",
    -1: "goldenrod", 1: "goldenrod",
    0: "white"
}

for i in range(-4, 5):
    fig.add_trace(go.Scatter(
        x=df.index, y=df[f"SE_{i}"],
        mode="lines",
        name=f"SE {i}",
        line=dict(color=colors[i], width=2, dash="dash")
    ))

# EMAs
fig.add_trace(go.Scatter(
    x=df.index, y=df["EMA_10"],
    mode="lines",
    name="EMA 10",
    line=dict(color="red", width=2)
))
fig.add_trace(go.Scatter(
    x=df.index, y=df["EMA_50"],
    mode="lines",
    name="EMA 50",
    line=dict(color="skyblue", width=2)
))

fig.update_layout(
    title=f"{selected_ticker} Candlestick + Inertia Bands + EMAs",
    xaxis_title="Date",
    yaxis_title="Price",
    yaxis=dict(side="right"),
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color="white"),
    xaxis_rangeslider_visible=False,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    height=800,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)


















