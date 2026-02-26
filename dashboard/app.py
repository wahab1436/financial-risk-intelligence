"""
dashboard/app.py

Streamlit dashboard for the Financial Risk Intelligence System.
Reads from stored predictions and reports — never recomputes models live.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from dashboard.components import (
    market_overview,
    regime_timeline,
    risk_monitor,
    sentiment_dashboard,
    volatility_forecast,
    report_interface,
)

st.set_page_config(
    page_title="Financial Risk Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=300)
def load_config() -> dict:
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=300)
def load_processed_data(asset: str, cfg: dict) -> pd.DataFrame:
    """Load the processed OHLCV data for the selected asset."""
    safe = asset.replace("^", "").replace("-", "_")
    path = Path(cfg["data"]["processed_dir"]) / f"{safe}_processed.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=300)
def load_features(asset: str, cfg: dict) -> pd.DataFrame:
    safe = asset.replace("^", "").replace("-", "_")
    features_dir = Path(cfg["data"]["features_dir"])
    files = sorted(features_dir.glob(f"{safe}_market_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])


@st.cache_data(ttl=300)
def load_risk_scores(asset: str) -> pd.DataFrame:
    safe = asset.replace("^", "").replace("-", "_")
    path = Path("data/predictions") / f"{safe}_risk_scores.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=300)
def load_vol_predictions(asset: str) -> pd.DataFrame:
    safe = asset.replace("^", "").replace("-", "_")
    path = Path("data/predictions") / f"{safe}_vol_predictions.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=300)
def load_regime_labels(asset: str) -> pd.Series:
    safe = asset.replace("^", "").replace("-", "_")
    path = Path("data/predictions") / f"{safe}_regimes.parquet"
    if not path.exists():
        return pd.Series(dtype=str)
    df = pd.read_parquet(path)
    return df.iloc[:, 0]


def sidebar(cfg: dict) -> tuple[str, str]:
    """Render sidebar controls and return selected asset and view."""
    st.sidebar.title("Financial Risk Intelligence")
    st.sidebar.markdown("---")

    asset = st.sidebar.selectbox(
        "Select Asset",
        options=cfg["data"]["assets"],
        index=0,
    )

    view = st.sidebar.radio(
        "View",
        options=[
            "Market Overview",
            "Regime Timeline",
            "Volatility Forecast",
            "Sentiment Dashboard",
            "Risk Monitor",
            "Report Interface",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"System: {cfg['system']['name']} v{cfg['system']['version']}")
    st.sidebar.caption("Dashboard reads stored predictions only.")
    return asset, view


def main():
    cfg = load_config()
    asset, view = sidebar(cfg)

    # Load data for selected asset
    price_df = load_processed_data(asset, cfg)
    features_df = load_features(asset, cfg)
    risk_df = load_risk_scores(asset)
    vol_df = load_vol_predictions(asset)
    regime_labels = load_regime_labels(asset)

    st.title(f"{view} — {asset}")
    st.markdown("---")

    if view == "Market Overview":
        market_overview.render(price_df, asset)

    elif view == "Regime Timeline":
        regime_timeline.render(price_df, regime_labels, asset)

    elif view == "Volatility Forecast":
        volatility_forecast.render(vol_df, asset)

    elif view == "Sentiment Dashboard":
        sentiment_dashboard.render(features_df, asset)

    elif view == "Risk Monitor":
        risk_monitor.render(risk_df, asset)

    elif view == "Report Interface":
        report_interface.render(asset, cfg)


if __name__ == "__main__":
    main()
