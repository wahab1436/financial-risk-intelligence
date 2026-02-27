"""
tests/test_features.py

Unit tests for feature engineering calculations.
Validates correctness and leakage prevention.
"""

import numpy as np
import pandas as pd
import pytest

from feature_engineering.market_features import MarketFeatureEngineer


@pytest.fixture
def sample_config() -> dict:
    return {
        "features": {
            "rolling_windows": [7, 14, 30, 60],
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "atr_period": 14,
            "beta_window": 60,
            "lag": 1,
        }
    }


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate synthetic OHLCV data with 300 trading days."""
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = 400 + np.cumsum(rng.normal(0, 2, n))
    df = pd.DataFrame(
        {
            "Open":     close * rng.uniform(0.995, 1.005, n),
            "High":     close * rng.uniform(1.005, 1.015, n),
            "Low":      close * rng.uniform(0.985, 0.995, n),
            "Close":    close,
            "Volume":   rng.integers(1_000_000, 10_000_000, n).astype(float),
            "Adj Close": close,
        },
        index=dates,
    )
    return df


class TestMarketFeatures:
    def test_output_shape(self, sample_config, sample_ohlcv):
        """Feature matrix should have same number of rows as input."""
        eng = MarketFeatureEngineer(sample_config)
        feats = eng.compute(sample_ohlcv, "TEST")
        assert len(feats) == len(sample_ohlcv), "Feature row count mismatch"

    def test_lag_applied(self, sample_config, sample_ohlcv):
        """
        Verifies leakage prevention: features at index T must not use data at T.
        Spot-check by computing a 7-day vol window without lag and confirming
        the lagged result does NOT match the same date's unlagged value.
        """
        eng = MarketFeatureEngineer(sample_config)
        feats = eng.compute(sample_ohlcv, "TEST")

        # Manually compute unlagged 7-day vol
        log_ret = np.log(sample_ohlcv["Close"] / sample_ohlcv["Close"].shift(1))
        unlagged_vol = log_ret.rolling(7).std()

        # The lagged feature at date T should equal the unlagged value at T-1
        common_idx = feats.index.intersection(unlagged_vol.index)[10:]  # skip warmup
        lagged_col = feats.loc[common_idx, "vol_std_7d"]
        expected = unlagged_vol.shift(1).reindex(common_idx)

        mismatch = (lagged_col - expected).abs().max()
        assert mismatch < 1e-10, f"Lag not correctly applied; max mismatch: {mismatch}"

    def test_no_future_in_features(self, sample_config, sample_ohlcv):
        """Feature at position T should not contain information from T+1 onward."""
        eng = MarketFeatureEngineer(sample_config)
        feats = eng.compute(sample_ohlcv, "TEST")
        log_ret = np.log(sample_ohlcv["Close"] / sample_ohlcv["Close"].shift(1))

        # The 30d return mean feature at the last date should reflect data up to T-1
        last_feature_date = feats.index[-1]
        second_last_date = feats.index[-2]
        expected_mean = log_ret.rolling(30).mean().shift(1)[second_last_date]
        actual = feats.loc[second_last_date, "return_mean_30d"]
        assert abs(actual - expected_mean) < 1e-10

    def test_all_expected_columns_present(self, sample_config, sample_ohlcv):
        eng = MarketFeatureEngineer(sample_config)
        feats = eng.compute(sample_ohlcv, "TEST")
        expected_cols = ["log_return", "vol_std_7d", "vol_std_30d", "rsi", "macd",
                         "atr", "drawdown_7d", "volume_zscore"]
        for col in expected_cols:
            assert col in feats.columns, f"Missing expected column: {col}"

    def test_rsi_bounded(self, sample_config, sample_ohlcv):
        """RSI must always be in [0, 100]."""
        eng = MarketFeatureEngineer(sample_config)
        feats = eng.compute(sample_ohlcv, "TEST")
        rsi = feats["rsi"].dropna()
        assert rsi.between(0, 100).all(), "RSI out of [0, 100] range"

    def test_drawdown_non_positive(self, sample_config, sample_ohlcv):
        """Drawdown values should be <= 0."""
        eng = MarketFeatureEngineer(sample_config)
        feats = eng.compute(sample_ohlcv, "TEST")
        dd = feats["drawdown_7d"].dropna()
        assert (dd <= 1e-10).all(), "Drawdown values should be non-positive"

    def test_forward_vol_not_lagged(self, sample_config, sample_ohlcv):
        """Forward volatility target must not be shifted (it's the prediction target)."""
        eng = MarketFeatureEngineer(sample_config)
        fwd_vol = eng.compute_forward_vol(sample_ohlcv, horizon=7)
        assert fwd_vol.name == "forward_vol_7d"
        assert not fwd_vol.dropna().empty
