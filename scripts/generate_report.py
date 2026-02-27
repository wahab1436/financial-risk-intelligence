"""
scripts/generate_report.py

Generate an LLM risk report for a specific asset.
Reads stored predictions — does not rerun models.

Usage:
    python -m scripts.generate_report --asset SPY
    python -m scripts.generate_report --asset QQQ --all-assets
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from llm_reports.report_generator import ReportGenerator
from llm_reports.report_storage import ReportStorage
from models.regime_model import RegimeDetector
from risk_engine.risk_calculator import RiskScoreCalculator


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_for_asset(asset: str, cfg: dict) -> None:
    safe = asset.replace("^", "").replace("-", "_")
    preds_dir = Path("data/predictions")

    # Load stored predictions
    risk_path = preds_dir / f"{safe}_risk_scores.parquet"
    regime_path = preds_dir / f"{safe}_regimes.parquet"
    vol_path = preds_dir / f"{safe}_vol_predictions.parquet"
    processed_path = Path(cfg["data"]["processed_dir"]) / f"{safe}_processed.parquet"

    if not risk_path.exists():
        logger.error(f"No risk scores found for {asset}. Run run_pipeline.py first.")
        return

    risk_df = pd.read_parquet(risk_path)
    vol_df = pd.read_parquet(vol_path) if vol_path.exists() else pd.DataFrame()
    price_df = pd.read_parquet(processed_path) if processed_path.exists() else pd.DataFrame()

    # Current values
    risk_calculator = RiskScoreCalculator(cfg)
    latest_payload = risk_calculator.latest_risk_payload(risk_df)

    # Regime
    current_regime = "unknown"
    regime_probability = 0.5
    if regime_path.exists():
        regime_labels = pd.read_parquet(regime_path).iloc[:, 0]
        current_regime = str(regime_labels.iloc[-1])

    # Volatility
    forecasted_vol = 0.0
    if not vol_df.empty and "predicted" in vol_df.columns:
        forecasted_vol = float(vol_df["predicted"].iloc[-1])

    # Sentiment
    features_dir = Path(cfg["data"]["features_dir"])
    sentiment_index = 0.0
    market_files = sorted(features_dir.glob(f"{safe}_market_*.parquet"))
    if market_files:
        feats = pd.read_parquet(market_files[-1])
        if "sentiment_index" in feats.columns:
            sentiment_index = float(feats["sentiment_index"].iloc[-1])

    # Price data
    close_price = 0.0
    ytd_return = 0.0
    if not price_df.empty:
        close_price = float(price_df["Close"].iloc[-1])
        ytd_start = price_df[price_df.index.year == price_df.index[-1].year]["Close"].iloc[0]
        ytd_return = (close_price - ytd_start) / ytd_start

    # Build payload and generate
    generator = ReportGenerator(cfg)
    storage = ReportStorage(cfg)

    full_payload = generator.build_payload(
        asset=asset,
        risk_payload=latest_payload,
        current_regime=current_regime,
        regime_probability=regime_probability,
        forecasted_vol=forecasted_vol,
        sentiment_index=sentiment_index,
        close_price=close_price,
        ytd_return=ytd_return,
    )

    report = generator.generate_asset_report(full_payload)
    paths = storage.save(report)

    logger.info(f"Report generated for {asset}")
    logger.info(f"  JSON: {paths['json']}")
    logger.info(f"  PDF:  {paths['pdf']}")
    print("\n" + "=" * 60)
    print(report["report_text"])
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="SPY", help="Asset ticker to generate report for")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--all-assets", action="store_true", help="Generate for all configured assets")
    args = parser.parse_args()

    cfg = load_config(args.config)
    assets = cfg["data"]["assets"] if args.all_assets else [args.asset]

    for asset in assets:
        try:
            generate_for_asset(asset, cfg)
        except Exception as exc:
            logger.error(f"Failed to generate report for {asset}: {exc}")
