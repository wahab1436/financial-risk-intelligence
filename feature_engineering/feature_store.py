"""
feature_engineering/feature_store.py

Manages persistence and retrieval of feature tables.
Features are stored as versioned parquet files with metadata.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger


class FeatureStore:
    """
    Simple file-based feature store backed by parquet.

    Naming convention:
        {feature_dir}/{asset}_{feature_set}_{version}.parquet
    Metadata:
        {feature_dir}/{asset}_{feature_set}_{version}.meta.json
    """

    def __init__(self, config: dict):
        self.store_dir = Path(config["data"]["features_dir"])
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        df: pd.DataFrame,
        asset: str,
        feature_set: str,
        version: str | None = None,
    ) -> Path:
        """
        Save a feature DataFrame with metadata.

        Args:
            df:          Feature DataFrame indexed by date.
            asset:       Ticker symbol.
            feature_set: Logical group name (e.g. 'market', 'sentiment').
            version:     Optional version string. Defaults to UTC timestamp.

        Returns:
            Path to saved parquet file.
        """
        version = version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe = asset.replace("^", "").replace("-", "_")
        stem = f"{safe}_{feature_set}_{version}"
        parquet_path = self.store_dir / f"{stem}.parquet"
        meta_path = self.store_dir / f"{stem}.meta.json"

        df.to_parquet(parquet_path)

        meta = {
            "asset": asset,
            "feature_set": feature_set,
            "version": version,
            "columns": list(df.columns),
            "rows": len(df),
            "date_range": [str(df.index.min()), str(df.index.max())],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        logger.info(f"Saved {feature_set} features for {asset}: {len(df.columns)} cols, {len(df)} rows -> {parquet_path}")
        return parquet_path

    def load_latest(self, asset: str, feature_set: str) -> pd.DataFrame:
        """
        Load the most recently saved feature set for an asset.

        Args:
            asset:       Ticker symbol.
            feature_set: Logical group name.

        Returns:
            Feature DataFrame.
        """
        safe = asset.replace("^", "").replace("-", "_")
        pattern = f"{safe}_{feature_set}_*.parquet"
        matches = sorted(self.store_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No {feature_set} features found for {asset} in {self.store_dir}"
            )
        latest = matches[-1]
        logger.info(f"Loading features from {latest}")
        return pd.read_parquet(latest)

    def load_version(self, asset: str, feature_set: str, version: str) -> pd.DataFrame:
        safe = asset.replace("^", "").replace("-", "_")
        path = self.store_dir / f"{safe}_{feature_set}_{version}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        return pd.read_parquet(path)

    def list_versions(self, asset: str, feature_set: str) -> list[dict]:
        """Return metadata for all stored versions of a feature set."""
        safe = asset.replace("^", "").replace("-", "_")
        pattern = f"{safe}_{feature_set}_*.meta.json"
        metas = []
        for meta_path in sorted(self.store_dir.glob(pattern)):
            metas.append(json.loads(meta_path.read_text()))
        return metas

    def build_combined_features(
        self,
        asset: str,
        market_features: pd.DataFrame,
        sentiment_series: pd.Series | None = None,
        regime_labels: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Merge market, sentiment, and regime features into a single DataFrame.

        Args:
            asset:            Ticker symbol for logging.
            market_features:  Output of MarketFeatureEngineer.compute().
            sentiment_series: Optional aligned sentiment index.
            regime_labels:    Optional regime label series.

        Returns:
            Combined feature DataFrame.
        """
        combined = market_features.copy()

        if sentiment_series is not None:
            combined["sentiment_index"] = sentiment_series.reindex(combined.index)

        if regime_labels is not None:
            combined["regime"] = regime_labels.reindex(combined.index)

        combined.dropna(how="all", inplace=True)
        logger.info(f"{asset}: Combined feature matrix shape: {combined.shape}")
        return combined
