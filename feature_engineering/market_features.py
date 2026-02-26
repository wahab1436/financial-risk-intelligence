"""
feature_engineering/market_features.py

Computes all market-based features from processed OHLCV data.
All features are properly lagged to prevent look-ahead bias.
"""

import numpy as np
import pandas as pd
import ta
from loguru import logger


class MarketFeatureEngineer:
    """
    Generates return-based, volatility-based, momentum, and risk features.

    Critical design rule: every feature computed at time T uses only data
    from T-lag and earlier. Default lag = 1 (next-day use of today's feature).
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.windows = config["features"]["rolling_windows"]
        self.rsi_period = config["features"]["rsi_period"]
        self.macd_fast = config["features"]["macd_fast"]
        self.macd_slow = config["features"]["macd_slow"]
        self.macd_signal = config["features"]["macd_signal"]
        self.atr_period = config["features"]["atr_period"]
        self.beta_window = config["features"]["beta_window"]
        self.lag = config["features"]["lag"]

    def compute(
        self,
        df: pd.DataFrame,
        asset: str,
        vix_df: pd.DataFrame | None = None,
        spy_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Compute all features for a single asset.

        Args:
            df:      Processed OHLCV DataFrame for the target asset.
            asset:   Ticker string (for logging).
            vix_df:  VIX Close series aligned to same index.
            spy_df:  SPY Close series for beta computation.

        Returns:
            DataFrame of features (one row per trading day).
        """
        feats = pd.DataFrame(index=df.index)
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # --- Return-based features ---
        log_ret = np.log(close / close.shift(1))
        feats["log_return"] = log_ret
        for w in self.windows:
            feats[f"return_mean_{w}d"] = log_ret.rolling(w).mean()

        # --- Volatility features ---
        for w in self.windows:
            feats[f"vol_std_{w}d"] = log_ret.rolling(w).std()
            feats[f"realized_vol_{w}d"] = (log_ret ** 2).rolling(w).sum().apply(np.sqrt)

        atr_indicator = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=self.atr_period
        )
        feats["atr"] = atr_indicator.average_true_range()

        # --- Momentum indicators ---
        feats["rsi"] = ta.momentum.RSIIndicator(close=close, window=self.rsi_period).rsi()

        macd_ind = ta.trend.MACD(
            close=close,
            window_slow=self.macd_slow,
            window_fast=self.macd_fast,
            window_sign=self.macd_signal,
        )
        feats["macd"] = macd_ind.macd()
        feats["macd_signal"] = macd_ind.macd_signal()
        feats["macd_diff"] = macd_ind.macd_diff()

        ema_short = close.ewm(span=12, adjust=False).mean()
        ema_long = close.ewm(span=26, adjust=False).mean()
        feats["ema_crossover"] = (ema_short > ema_long).astype(int)

        # --- Risk indicators ---
        for w in self.windows:
            rolling_max = close.rolling(w).max()
            feats[f"drawdown_{w}d"] = (close - rolling_max) / rolling_max

        # Rolling beta vs SPY
        if spy_df is not None:
            spy_ret = np.log(spy_df["Close"] / spy_df["Close"].shift(1)).reindex(df.index)
            cov = log_ret.rolling(self.beta_window).cov(spy_ret)
            var = spy_ret.rolling(self.beta_window).var()
            feats["rolling_beta"] = cov / var.replace(0, np.nan)

        # Rolling correlation with VIX
        if vix_df is not None:
            vix_ret = np.log(vix_df["Close"] / vix_df["Close"].shift(1)).reindex(df.index)
            feats["corr_vix"] = log_ret.rolling(30).corr(vix_ret)

        # Volume z-score
        feats["volume_zscore"] = (
            (volume - volume.rolling(30).mean()) / volume.rolling(30).std()
        )

        # Return skewness
        feats["return_skew_30d"] = log_ret.rolling(30).skew()

        # --- Apply lag to prevent leakage ---
        feats = feats.shift(self.lag)

        n_features = len(feats.columns)
        logger.info(f"{asset}: Computed {n_features} features with lag={self.lag}")
        return feats

    def compute_forward_vol(self, df: pd.DataFrame, horizon: int = 7) -> pd.Series:
        """
        Compute the target variable: forward realized volatility.
        This must NOT be lagged — it is the prediction target.

        Args:
            df:      OHLCV DataFrame.
            horizon: Number of trading days to look forward.

        Returns:
            Series of forward realized volatility values.
        """
        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        forward_vol = (
            log_ret.shift(-horizon)
            .rolling(horizon)
            .apply(lambda x: np.sqrt((x ** 2).sum()), raw=True)
        )
        forward_vol.name = f"forward_vol_{horizon}d"
        return forward_vol
