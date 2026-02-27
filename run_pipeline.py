"""
run_pipeline.py

Full end-to-end pipeline orchestration.
Runs: ingest -> validate -> preprocess -> features -> regime -> volatility -> risk -> report
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from data_pipeline.ingestion import MarketDataIngester
from data_pipeline.preprocessing import MarketDataPreprocessor
from data_pipeline.validation import validate_all
from feature_engineering.feature_store import FeatureStore
from feature_engineering.market_features import MarketFeatureEngineer
from models.evaluation import (
    compute_regime_stability,
    compute_volatility_metrics,
)
from models.regime_model import RegimeDetector
from models.volatility_model import GARCHVolatilityModel, XGBoostVolatilityModel
from monitoring.logging import MLflowTracker, PredictionLogger, configure_logging
from risk_engine.backtesting import compute_risk_score_summary, crisis_period_analysis
from risk_engine.risk_calculator import RiskScoreCalculator


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline(cfg: dict, assets: list[str] | None = None) -> None:
    configure_logging(cfg["monitoring"]["log_dir"])
    assets = assets or cfg["data"]["assets"]

    tracker = MLflowTracker(cfg)
    pred_logger = PredictionLogger(cfg)
    predictions_dir = Path("data/predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Pipeline starting for assets: {assets}")

    # 1. Ingest
    ingester = MarketDataIngester(cfg)
    raw_data = ingester.run(assets)
    if not raw_data:
        logger.error("No data ingested. Aborting pipeline.")
        return

    # 2. Validate
    validation_results = validate_all(raw_data, cfg)
    failed = [a for a, r in validation_results.items() if not r.passed]
    if failed:
        logger.warning(f"Validation failed for: {failed}. These assets will be excluded.")
        for a in failed:
            raw_data.pop(a, None)

    # 3. Preprocess
    preprocessor = MarketDataPreprocessor(cfg)
    processed_data = preprocessor.run(raw_data)

    # Separate VIX and SPY for cross-asset feature computation
    vix_df = processed_data.get("^VIX")
    spy_df = processed_data.get("SPY")

    # 4. Feature engineering
    feature_eng = MarketFeatureEngineer(cfg)
    store = FeatureStore(cfg)
    all_features = {}

    for asset, df in processed_data.items():
        feats = feature_eng.compute(df, asset, vix_df=vix_df, spy_df=spy_df)
        store.save(feats, asset, "market")
        all_features[asset] = feats

    # 5. Regime detection (fit on SPY as the primary regime asset)
    primary = "SPY" if "SPY" in all_features else list(all_features.keys())[0]
    regime_detector = RegimeDetector(cfg)
    regime_detector.fit(all_features[primary])

    regime_labels_all = {}
    for asset, feats in all_features.items():
        labels = regime_detector.predict_labels(feats)
        probas = regime_detector.predict_proba(feats)
        regime_labels_all[asset] = labels
        labels.to_frame().to_parquet(predictions_dir / f"{asset.replace('^','').replace('-','_')}_regimes.parquet")

    # Log regime run
    stability = compute_regime_stability(regime_labels_all[primary])
    tracker.log_regime_run(
        cfg, stability, regime_detector.regime_map,
        regime_detector.get_transition_matrix()
    )

    # 6. Volatility forecasting
    vol_model = XGBoostVolatilityModel(cfg)
    garch_model = GARCHVolatilityModel(cfg)

    for asset, feats in all_features.items():
        if asset == "^VIX":
            continue
        df = processed_data[asset]
        target = feature_eng.compute_forward_vol(df, horizon=cfg["volatility"]["forecast_horizon"])

        # Add regime labels to features
        combined_feats = store.build_combined_features(
            asset, feats,
            regime_labels=regime_labels_all.get(asset)
        )

        # Walk-forward validation
        wf_results = vol_model.walk_forward_fit_predict(combined_feats, target)
        wf_results.to_parquet(predictions_dir / f"{asset.replace('^','').replace('-','_')}_vol_predictions.parquet")

        metrics = compute_volatility_metrics(wf_results)
        tracker.log_volatility_run(cfg, metrics, vol_model.model)

        # GARCH baseline
        log_ret = pd.Series(
            df["Close"].apply(lambda x: x).values,
            index=df.index,
            dtype=float,
        )
        log_ret = (df["Close"] / df["Close"].shift(1)).apply(
            lambda x: __import__("math").log(x) if x > 0 else 0
        )
        garch_model.fit(log_ret)
        garch_forecast = garch_model.forecast()
        logger.info(f"{asset} GARCH annualized vol forecast: {garch_forecast:.4f}")

    # 7. Risk scoring
    risk_calculator = RiskScoreCalculator(cfg)

    for asset, feats in all_features.items():
        if asset == "^VIX":
            continue
        df = processed_data[asset]

        drawdown = (df["Close"] - df["Close"].rolling(252, min_periods=1).max()) / df["Close"].rolling(252, min_periods=1).max()
        vol_preds = pd.read_parquet(predictions_dir / f"{asset.replace('^','').replace('-','_')}_vol_predictions.parquet")

        # Reconstruct vol forecast series aligned to all available dates
        forecasted_vol = vol_preds["predicted"].reindex(feats.index).ffill()

        regime_labels = regime_labels_all.get(asset, pd.Series(dtype=str))
        regime_named = regime_labels.map(regime_detector.regime_map).reindex(feats.index)

        sentiment = feats.get("sentiment_index", pd.Series(0.0, index=feats.index))
        vix_series = vix_df["Close"] if vix_df is not None else None

        risk_df = risk_calculator.compute(
            forecasted_vol=forecasted_vol,
            regime_labels=regime_named,
            sentiment_index=sentiment,
            drawdown=drawdown.reindex(feats.index),
            vix=vix_series,
        )
        safe_asset = asset.replace("^", "").replace("-", "_")
        risk_df.to_parquet(predictions_dir / f"{safe_asset}_risk_scores.parquet")

        summary = compute_risk_score_summary(risk_df)
        tracker.log_risk_run(cfg, summary)
        logger.info(f"{asset} Risk Summary: {summary}")

        # Backtest against crisis periods
        crisis_analysis = crisis_period_analysis(risk_df["risk_score"], df["Close"])
        if not crisis_analysis.empty:
            logger.info(f"{asset} Crisis Analysis:\n{crisis_analysis}")

        # Log latest prediction
        payload = risk_calculator.latest_risk_payload(risk_df)
        pred_logger.log_prediction(
            asset=asset,
            model_name="composite_risk_score",
            model_version=cfg["system"]["version"],
            input_features={"n_features": len(feats.columns)},
            prediction=payload,
        )

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the financial risk intelligence pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--assets", nargs="*", help="Override asset list")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline(cfg, assets=args.assets)
