"""
data_pipeline/preprocessing.py

Cleans and normalizes raw market data for feature engineering.
Processed data is stored separately from raw data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


class MarketDataPreprocessor:
    """
    Applies cleaning and alignment transformations to validated market data.
    Stores processed DataFrames in the processed data directory.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.processed_dir = Path(config["data"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """
        Full preprocessing pipeline for a single asset.

        Steps:
            1. Sort by date ascending.
            2. Forward-fill up to 2 consecutive missing values.
            3. Remove any remaining nulls in Close/Adj Close.
            4. Clip extreme return outliers (beyond 6 sigma) without deletion.
            5. Convert index to date-only (timezone-naive) for consistency.

        Args:
            df:    Raw OHLCV DataFrame.
            asset: Ticker symbol for logging.

        Returns:
            Cleaned DataFrame.
        """
        df = df.sort_index()

        # Normalize index to timezone-naive date
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()

        initial_len = len(df)

        # Forward-fill short gaps (e.g., minor data provider issues)
        df = df.ffill(limit=2)

        # Drop rows where core price fields are still missing
        before_drop = len(df)
        df = df.dropna(subset=["Close", "Adj Close"])
        dropped = before_drop - len(df)
        if dropped > 0:
            logger.warning(f"{asset}: Dropped {dropped} rows with missing Close/Adj Close")

        # Clip extreme log-return outliers in-place (preserve row count)
        log_ret = np.log(df["Close"] / df["Close"].shift(1)).fillna(0)
        mu, sigma = log_ret.mean(), log_ret.std()
        if sigma > 0:
            lower = mu - 6 * sigma
            upper = mu + 6 * sigma
            extreme = (log_ret < lower) | (log_ret > upper)
            if extreme.any():
                logger.warning(f"{asset}: Clipping {extreme.sum()} extreme return(s)")
                df.loc[extreme, "Close"] = np.nan
                df["Close"] = df["Close"].ffill()

        logger.info(f"{asset}: Preprocessed {initial_len} -> {len(df)} rows")
        return df

    def save(self, df: pd.DataFrame, asset: str) -> Path:
        safe = asset.replace("^", "").replace("-", "_")
        path = self.processed_dir / f"{safe}_processed.parquet"
        df.to_parquet(path)
        logger.info(f"Saved processed data: {path}")
        return path

    def load(self, asset: str) -> pd.DataFrame:
        safe = asset.replace("^", "").replace("-", "_")
        path = self.processed_dir / f"{safe}_processed.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No processed data found for {asset} at {path}")
        return pd.read_parquet(path)

    def run(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Process and save all assets. Returns dict of processed DataFrames."""
        processed = {}
        for asset, df in data.items():
            try:
                clean = self.process(df, asset)
                self.save(clean, asset)
                processed[asset] = clean
            except Exception as exc:
                logger.error(f"Preprocessing failed for {asset}: {exc}")
        return processed


def align_assets(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Align all assets to a common date index by inner join.
    Useful before cross-asset feature computation (beta, correlations).

    Args:
        data: Dict of asset -> DataFrame.

    Returns:
        Dict of asset -> DataFrame aligned to the common date range.
    """
    if not data:
        return data
    common_index = None
    for df in data.values():
        idx = df.index
        common_index = idx if common_index is None else common_index.intersection(idx)
    aligned = {asset: df.loc[common_index] for asset, df in data.items()}
    logger.info(f"Aligned {len(data)} assets to {len(common_index)} common trading days")
    return aligned
