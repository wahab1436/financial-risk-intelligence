"""
tests/test_validation.py

Tests for the data validation framework.
"""

import numpy as np
import pandas as pd
import pytest

from data_pipeline.validation import MarketDataValidator


@pytest.fixture
def validator_config() -> dict:
    return {
        "validation": {
            "max_return_zscore": 5.0,
            "max_null_pct": 0.05,
            "min_trading_days_per_year": 240,
        }
    }


@pytest.fixture
def clean_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 252
    dates = pd.bdate_range("2023-01-03", periods=n)
    close = 400 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Open":     close * 0.999,
            "High":     close * 1.005,
            "Low":      close * 0.995,
            "Close":    close,
            "Volume":   rng.integers(1e6, 5e6, n).astype(float),
            "Adj Close": close,
        },
        index=dates,
    )


class TestMarketDataValidator:
    def test_clean_data_passes_all(self, validator_config, clean_df):
        v = MarketDataValidator(validator_config)
        result = v.validate(clean_df, "SPY")
        assert result.passed, f"Clean data should pass all checks. Failures: {result.details}"

    def test_missing_columns_fails_schema(self, validator_config, clean_df):
        v = MarketDataValidator(validator_config)
        bad_df = clean_df.drop(columns=["High", "Low"])
        result = v.validate(bad_df, "SPY")
        assert not result.checks["schema_conformance"]

    def test_duplicate_index_detected(self, validator_config, clean_df):
        v = MarketDataValidator(validator_config)
        dup_df = pd.concat([clean_df, clean_df.iloc[:5]])
        dup_df = dup_df.sort_index()
        result = v.validate(dup_df, "SPY")
        assert not result.checks["no_duplicate_rows"]

    def test_excessive_nulls_detected(self, validator_config, clean_df):
        v = MarketDataValidator(validator_config)
        null_df = clean_df.copy()
        null_df.loc[null_df.index[:30], "Close"] = np.nan  # ~12% nulls
        result = v.validate(null_df, "SPY")
        assert not result.checks["null_values_within_threshold"]

    def test_extreme_return_spike_detected(self, validator_config, clean_df):
        v = MarketDataValidator(validator_config)
        spike_df = clean_df.copy()
        spike_df.loc[spike_df.index[50], "Close"] = spike_df["Close"].iloc[50] * 3.0
        result = v.validate(spike_df, "SPY")
        assert not result.checks["no_abnormal_return_spikes"]

    def test_unsorted_index_fails_ordering(self, validator_config, clean_df):
        v = MarketDataValidator(validator_config)
        shuffled = clean_df.sample(frac=1, random_state=0)
        result = v.validate(shuffled, "SPY")
        assert not result.checks["temporal_ordering"]

    def test_crypto_skips_calendar_check(self, validator_config, clean_df):
        v = MarketDataValidator(validator_config)
        result = v.validate(clean_df, "BTC-USD")
        # Crypto assets skip the NYSE calendar check
        assert result.checks["non_trading_day_consistency"]
