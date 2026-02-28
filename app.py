"""
dashboard/app.py

Streamlit dashboard — reads ONLY from stored predictions. Never reruns models live.

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Add repo root to sys.path so all backend modules resolve correctly
# whether running locally or on Streamlit Cloud
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import yaml

from dashboard.components import (
    market_overview,
    regime_timeline,
    risk_monitor,
    report_interface,
    sentiment_dashboard,
    volatility_forecast,
)

st.set_page_config(
    page_title="Financial Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=300)
def _load_config() -> dict:
    # Always resolve config relative to repo root, not cwd
    config_path = ROOT / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=300)
def _load_parquet(path: str) -> pd.DataFrame:
    p = ROOT / path
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def _safe(asset: str) -> str:
    return asset.replace("^", "").replace("-", "_")


def sidebar(cfg: dict) -> tuple[str, str]:
    st.sidebar.title("Risk Intelligence")
    st.sidebar.markdown("---")
    asset = st.sidebar.selectbox("Asset", cfg["data"]["assets"])
    view  = st.sidebar.radio("View", [
        "Market Overview",
        "Regime Timeline",
        "Volatility Forecast",
        "Sentiment Dashboard",
        "Risk Monitor",
        "Report Interface",
    ])
    st.sidebar.markdown("---")
    st.sidebar.caption(f"{cfg['system']['name']}  v{cfg['system']['version']}")
    return asset, view


def main() -> None:
    cfg        = _load_config()
    asset, view = sidebar(cfg)
    safe       = _safe(asset)
    pdir       = cfg["data"]["predictions_dir"]
    fdir       = cfg["data"]["features_dir"]
    proc_dir   = cfg["data"]["processed_dir"]

    # Load stored data — missing files return empty DataFrames gracefully
    price_df = _load_parquet(f"{proc_dir}/{safe}_processed.parquet")
    risk_df  = _load_parquet(f"{pdir}/{safe}_risk_scores.parquet")
    vol_df   = _load_parquet(f"{pdir}/{safe}_vol_predictions.parquet")

    # Latest feature file for this asset
    feat_dir   = ROOT / fdir
    feat_files = sorted(feat_dir.glob(f"{safe}_market_*.parquet")) if feat_dir.exists() else []
    feats_df   = pd.read_parquet(feat_files[-1]) if feat_files else pd.DataFrame()

    # Regime labels
    regime_path = ROOT / pdir / f"{safe}_regimes.parquet"
    if regime_path.exists():
        regime_labels = pd.read_parquet(regime_path).iloc[:, 0]
    else:
        regime_labels = pd.Series(dtype=str)

    st.title(f"{view}  —  {asset}")
    st.markdown("---")

    if view == "Market Overview":
        market_overview.render(price_df, asset)
    elif view == "Regime Timeline":
        regime_timeline.render(price_df, regime_labels, asset)
    elif view == "Volatility Forecast":
        volatility_forecast.render(vol_df, asset)
    elif view == "Sentiment Dashboard":
        sentiment_dashboard.render(feats_df, asset)
    elif view == "Risk Monitor":
        risk_monitor.render(risk_df, asset)
    elif view == "Report Interface":
        report_interface.render(asset, cfg)


if __name__ == "__main__":
    main()
