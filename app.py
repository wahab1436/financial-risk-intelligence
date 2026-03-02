"""
app.py  —  Root-level Streamlit entry point.

Streamlit Cloud:  set Main file path = app.py
Local:            streamlit run app.py

Architecture:
  - Imports all dashboard components from dashboard/components/
  - Imports backend modules for live pipeline triggers
  - Reads stored predictions from data/predictions/ for display
  - Can re-run the full pipeline from the sidebar
"""

import sys
from pathlib import Path

# ── Add repo root to sys.path so all submodule imports resolve ────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import json
import math
import traceback
import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yaml

# ── Dashboard components ──────────────────────────────────────────────────────
from dashboard.components import market_overview
from dashboard.components import regime_timeline
from dashboard.components import volatility_forecast
from dashboard.components import sentiment_dashboard
from dashboard.components import risk_monitor
from dashboard.components import report_interface

# ── Backend: data pipeline ────────────────────────────────────────────────────
from data_pipeline.ingestion import MarketDataIngester, NewsIngester
from data_pipeline.validation import validate_all
from data_pipeline.preprocessing import MarketDataPreprocessor

# ── Backend: feature engineering ─────────────────────────────────────────────
from feature_engineering.market_features import MarketFeatureEngineer
from feature_engineering.sentiment_features import FinBERTSentimentPipeline
from feature_engineering.feature_store import FeatureStore

# ── Backend: models ───────────────────────────────────────────────────────────
from models.regime_model import RegimeDetector
from models.volatility_model import XGBoostVolatilityModel

# ── Backend: risk engine ──────────────────────────────────────────────────────
from risk_engine.risk_calculator import RiskScoreCalculator

# ── Backend: monitoring ───────────────────────────────────────────────────────
from monitoring.logging_utils import configure_logging, PredictionLogger

# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Financial Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Config & data loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=300)
def load_parquet(rel_path: str) -> pd.DataFrame:
    p = ROOT / rel_path
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def asset_safe(asset: str) -> str:
    return asset.replace("^", "").replace("-", "_")


def load_asset_data(cfg: dict, asset: str) -> dict:
    """Load all stored artefacts for one asset from data/predictions/."""
    s    = asset_safe(asset)
    pdir = cfg["data"]["predictions_dir"]
    fdir = cfg["data"]["features_dir"]
    proc = cfg["data"]["processed_dir"]

    price_df = load_parquet(f"{proc}/{s}_processed.parquet")
    risk_df  = load_parquet(f"{pdir}/{s}_risk_scores.parquet")
    vol_df   = load_parquet(f"{pdir}/{s}_vol_predictions.parquet")

    feat_dir   = ROOT / fdir
    feat_files = sorted(feat_dir.glob(f"{s}_market_*.parquet")) if feat_dir.exists() else []
    feats_df   = pd.read_parquet(feat_files[-1]) if feat_files else pd.DataFrame()

    regime_path   = ROOT / pdir / f"{s}_regimes.parquet"
    regime_labels = (
        pd.read_parquet(regime_path).iloc[:, 0]
        if regime_path.exists()
        else pd.Series(dtype=str)
    )

    return {
        "price":   price_df,
        "risk":    risk_df,
        "vol":     vol_df,
        "feats":   feats_df,
        "regimes": regime_labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Backend pipeline — triggered from sidebar
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(cfg: dict, assets: list[str], skip_news: bool = True) -> bool:
    """
    Full pipeline: ingest -> validate -> preprocess -> features ->
    regime -> volatility -> risk scores.
    Mirrors run_pipeline.py exactly so both entry points stay in sync.
    """
    try:
        configure_logging(cfg["monitoring"]["log_dir"])
        pdir = ROOT / cfg["data"]["predictions_dir"]
        pdir.mkdir(parents=True, exist_ok=True)
        pred_logger = PredictionLogger(cfg)

        # 1 — Ingest
        with st.spinner("Fetching market data from Yahoo Finance..."):
            raw = MarketDataIngester(cfg).run(assets)
        if not raw:
            st.error("No market data returned.")
            return False

        if not skip_news:
            with st.spinner("Fetching news headlines..."):
                try:
                    NewsIngester(cfg).run()
                except Exception as e:
                    st.warning(f"News ingestion skipped: {e}")

        # 2 — Validate
        with st.spinner("Validating data quality..."):
            val_results = validate_all(raw, cfg)
            failed = {a for a, r in val_results.items() if not r.passed}
            if failed:
                st.warning(f"Excluded assets (validation failed): {failed}")
                for a in failed:
                    raw.pop(a, None)
        if not raw:
            st.error("All assets failed validation.")
            return False

        # 3 — Preprocess
        with st.spinner("Preprocessing..."):
            processed = MarketDataPreprocessor(cfg).run(raw)

        vix_df = processed.get("^VIX")
        spy_df = processed.get("SPY")

        # 4 — Feature engineering
        eng       = MarketFeatureEngineer(cfg)
        store     = FeatureStore(cfg)
        sent_pipe = FinBERTSentimentPipeline(cfg)
        all_feats: dict[str, pd.DataFrame] = {}

        for asset, df in processed.items():
            with st.spinner(f"Engineering features: {asset}..."):
                feats = eng.compute(df, asset, vix_df=vix_df, spy_df=spy_df)
                try:
                    if not skip_news:
                        news_df = NewsIngester(cfg).load(asset_tag=asset)
                        if not news_df.empty:
                            daily     = sent_pipe.compute_daily_index(news_df)
                            sentiment = sent_pipe.align_to_market(daily, df.index, lag=1)
                        else:
                            sentiment = sent_pipe.fill_zero_sentiment(df.index)
                    else:
                        sentiment = sent_pipe.fill_zero_sentiment(df.index)
                except Exception:
                    sentiment = sent_pipe.fill_zero_sentiment(df.index)
                feats["sentiment_index"] = sentiment.reindex(feats.index)
                store.save(feats, asset, "market")
                all_feats[asset] = feats

        # 5 — Regime detection
        primary = next(
            (a for a in ["SPY", "QQQ"] if a in all_feats),
            list(all_feats.keys())[0],
        )
        with st.spinner("Detecting market regimes..."):
            detector = RegimeDetector(cfg)
            detector.fit(all_feats[primary])
            detector.save(str(pdir / "regime_model.pkl"))

        all_regime_labels: dict[str, pd.Series] = {}
        for asset, feats in all_feats.items():
            labels = detector.predict_labels(feats)
            all_regime_labels[asset] = labels
            labels.to_frame("regime").to_parquet(
                pdir / f"{asset_safe(asset)}_regimes.parquet"
            )

        # 6 — Volatility forecasting
        vol_model = XGBoostVolatilityModel(cfg)
        for asset, feats in all_feats.items():
            if asset == "^VIX":
                continue
            df       = processed[asset]
            target   = eng.compute_forward_vol(df, cfg["volatility"]["forecast_horizon"])
            combined = store.build_combined(
                asset, feats,
                sentiment     = feats.get("sentiment_index"),
                regime_labels = all_regime_labels.get(asset),
            )
            with st.spinner(f"Training volatility model: {asset}..."):
                try:
                    wf = vol_model.walk_forward_fit_predict(combined, target)
                    wf.to_parquet(pdir / f"{asset_safe(asset)}_vol_predictions.parquet")
                except ValueError as e:
                    st.warning(f"{asset}: walk-forward skipped — {e}")
        vol_model.save(str(pdir / "vol_model.pkl"))

        # 7 — Risk scoring
        risk_calc = RiskScoreCalculator(cfg)
        for asset, feats in all_feats.items():
            if asset == "^VIX":
                continue
            df       = processed[asset]
            vol_path = pdir / f"{asset_safe(asset)}_vol_predictions.parquet"
            if vol_path.exists():
                forecasted_vol = (
                    pd.read_parquet(vol_path)["predicted"]
                    .reindex(feats.index).ffill()
                )
            else:
                log_ret        = np.log(df["Close"] / df["Close"].shift(1))
                forecasted_vol = log_ret.rolling(30).std() * math.sqrt(252)
                forecasted_vol = forecasted_vol.reindex(feats.index)

            drawdown = (
                (df["Close"] - df["Close"].rolling(252, min_periods=1).max())
                / df["Close"].rolling(252, min_periods=1).max()
            ).reindex(feats.index)

            with st.spinner(f"Computing risk scores: {asset}..."):
                risk_df = risk_calc.compute(
                    forecasted_vol  = forecasted_vol,
                    regime_labels   = all_regime_labels.get(
                        asset, pd.Series("sideways", index=feats.index)
                    ),
                    sentiment_index = feats.get(
                        "sentiment_index", pd.Series(0.0, index=feats.index)
                    ),
                    drawdown = drawdown,
                    vix      = vix_df["Close"].reindex(feats.index) if vix_df is not None else None,
                )
                risk_df.to_parquet(pdir / f"{asset_safe(asset)}_risk_scores.parquet")
                pred_logger.log(
                    asset        = asset,
                    model_name   = "composite_risk",
                    model_version= cfg["system"]["version"],
                    input_summary= {"n_features": len(feats.columns), "n_rows": len(feats)},
                    prediction   = risk_calc.latest_risk_payload(risk_df),
                )

        st.cache_data.clear()
        return True

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.code(traceback.format_exc())
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

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
    st.sidebar.subheader("Run Pipeline")

    selected_assets = st.sidebar.multiselect(
        "Assets to run",
        cfg["data"]["assets"],
        default=[a for a in ["SPY", "QQQ"] if a in cfg["data"]["assets"]],
    )
    skip_news = st.sidebar.checkbox("Skip news ingestion", value=True)

    if st.sidebar.button("Run Pipeline Now", type="primary", use_container_width=True):
        if not selected_assets:
            st.sidebar.error("Select at least one asset.")
        else:
            ok = run_pipeline(cfg, selected_assets, skip_news=skip_news)
            if ok:
                st.sidebar.success("Pipeline complete — dashboard updated.")
            else:
                st.sidebar.error("Pipeline failed. See errors above.")

    # Last run timestamp
    st.sidebar.markdown("---")
    s         = asset_safe(asset)
    risk_path = ROOT / cfg["data"]["predictions_dir"] / f"{s}_risk_scores.parquet"
    if risk_path.exists():
        mtime = datetime.datetime.fromtimestamp(risk_path.stat().st_mtime)
        st.sidebar.caption(f"Last run: {mtime.strftime('%Y-%m-%d %H:%M')}")
    else:
        st.sidebar.caption("No data yet — click 'Run Pipeline Now'.")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"{cfg['system']['name']}  v{cfg['system']['version']}")

    return asset, view


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg         = load_config()
    asset, view = sidebar(cfg)
    data        = load_asset_data(cfg, asset)

    st.title(f"{view}  —  {asset}")
    st.markdown("---")

    if   view == "Market Overview":     market_overview.render(data["price"], asset)
    elif view == "Regime Timeline":     regime_timeline.render(data["price"], data["regimes"], asset)
    elif view == "Volatility Forecast": volatility_forecast.render(data["vol"], asset)
    elif view == "Sentiment Dashboard": sentiment_dashboard.render(data["feats"], asset)
    elif view == "Risk Monitor":        risk_monitor.render(data["risk"], asset)
    elif view == "Report Interface":    report_interface.render(asset, cfg)


if __name__ == "__main__":
    main()
