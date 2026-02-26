"""
dashboard/components/market_overview.py

Market overview panel: price chart and volume.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def render(price_df: pd.DataFrame, asset: str) -> None:
    if price_df.empty:
        st.warning(f"No processed price data found for {asset}.")
        return

    lookback = st.selectbox("Lookback Period", ["1Y", "2Y", "5Y", "All"], index=0)
    lookback_map = {"1Y": 252, "2Y": 504, "5Y": 1260, "All": len(price_df)}
    n = lookback_map[lookback]
    df = price_df.tail(n)

    col1, col2, col3, col4 = st.columns(4)
    latest_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2] if len(df) > 1 else latest_close
    day_ret = (latest_close - prev_close) / prev_close * 100

    col1.metric("Latest Close", f"${latest_close:,.2f}", f"{day_ret:+.2f}%")

    ytd_start = df[df.index.year == df.index[-1].year]["Close"].iloc[0]
    ytd_ret = (latest_close - ytd_start) / ytd_start * 100
    col2.metric("YTD Return", f"{ytd_ret:+.2f}%")

    rolling_vol = df["Close"].pct_change().rolling(30).std() * (252 ** 0.5) * 100
    col3.metric("30d Realized Vol", f"{rolling_vol.iloc[-1]:.1f}%")

    from_peak = (latest_close - df["Close"].max()) / df["Close"].max() * 100
    col4.metric("From 52W High", f"{from_peak:+.2f}%")

    st.markdown("---")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.7, 0.3]
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=asset,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color="rgba(100, 149, 237, 0.5)",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"{asset} — Price & Volume",
        xaxis_rangeslider_visible=False,
        height=550,
        template="plotly_dark",
        showlegend=False,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)
