"""
tests/test_risk_engine.py

Unit tests for the risk scoring engine.
Validates score bounds, formula correctness, and classification bands.
"""

import numpy as np
import pandas as pd
import pytest

from risk_engine.risk_calculator import RiskScoreCalculator


@pytest.fixture
def risk_config() -> dict:
    return {
        "risk_score": {
            "weights": {
                "volatility": 0.35,
                "regime": 0.30,
                "sentiment": 0.20,
                "drawdown": 0.15,
            },
            "regime_encoding": {
                "bear": 1.0,
                "high_vol": 0.8,
                "sideways": 0.4,
                "bull": 0.1,
            },
            "normalization_window": 504,
        }
    }


@pytest.fixture
def sample_inputs() -> dict:
    n = 300
    dates = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(42)
    return {
        "forecasted_vol": pd.Series(rng.uniform(0.05, 0.40, n), index=dates),
        "regime_labels": pd.Series(
            rng.choice(["bull", "bear", "sideways"], n), index=dates
        ),
        "sentiment_index": pd.Series(rng.uniform(-1, 1, n), index=dates),
        "drawdown": pd.Series(rng.uniform(-0.30, 0.0, n), index=dates),
    }


class TestRiskScoreCalculator:
    def test_score_bounded_0_1(self, risk_config, sample_inputs):
        """Risk score must always be in [0, 1]."""
        calc = RiskScoreCalculator(risk_config)
        result = calc.compute(**sample_inputs)
        scores = result["risk_score"].dropna()
        assert (scores >= 0).all(), "Score below 0"
        assert (scores <= 1).all(), "Score above 1"

    def test_weight_normalization(self, risk_config):
        """Calculator should auto-normalize weights that don't sum to 1."""
        bad_config = {**risk_config}
        bad_config["risk_score"] = {**risk_config["risk_score"]}
        bad_config["risk_score"]["weights"] = {
            "volatility": 0.5, "regime": 0.5, "sentiment": 0.5, "drawdown": 0.5
        }
        calc = RiskScoreCalculator(bad_config)
        total = calc.w_vol + calc.w_regime + calc.w_sentiment + calc.w_drawdown
        assert abs(total - 1.0) < 0.01, "Weights not normalized"

    def test_high_risk_scenario(self, risk_config):
        """Under maximum stress conditions, score should be near 1."""
        calc = RiskScoreCalculator(risk_config)
        n = 200
        dates = pd.bdate_range("2022-01-03", periods=n)
        extreme = {
            "forecasted_vol": pd.Series([0.99] * n, index=dates),
            "regime_labels": pd.Series(["bear"] * n, index=dates),
            "sentiment_index": pd.Series([-1.0] * n, index=dates),
            "drawdown": pd.Series([-0.99] * n, index=dates),
        }
        result = calc.compute(**extreme)
        latest = result["risk_score"].iloc[-1]
        assert latest > 0.7, f"Expected high risk score, got {latest:.3f}"

    def test_low_risk_scenario(self, risk_config):
        """Under benign conditions, score should be near 0."""
        calc = RiskScoreCalculator(risk_config)
        n = 200
        dates = pd.bdate_range("2022-01-03", periods=n)
        benign = {
            "forecasted_vol": pd.Series([0.01] * n, index=dates),
            "regime_labels": pd.Series(["bull"] * n, index=dates),
            "sentiment_index": pd.Series([1.0] * n, index=dates),
            "drawdown": pd.Series([0.0] * n, index=dates),
        }
        result = calc.compute(**benign)
        latest = result["risk_score"].iloc[-1]
        assert latest < 0.50, f"Expected low risk score, got {latest:.3f}"

    def test_risk_bands_exhaustive(self, risk_config):
        """Every score in [0,1] should map to a valid band."""
        calc = RiskScoreCalculator(risk_config)
        valid_bands = {"Low", "Elevated", "High", "Severe"}
        for score in [0.0, 0.15, 0.30, 0.45, 0.55, 0.70, 0.75, 0.99, 1.0]:
            band = calc._classify(score)
            assert band in valid_bands, f"Score {score} mapped to invalid band '{band}'"

    def test_latest_risk_payload_structure(self, risk_config, sample_inputs):
        """latest_risk_payload should return all required keys."""
        calc = RiskScoreCalculator(risk_config)
        result = calc.compute(**sample_inputs)
        payload = calc.latest_risk_payload(result)
        required_keys = ["date", "risk_score", "risk_band", "components", "weights"]
        for key in required_keys:
            assert key in payload, f"Missing key in payload: {key}"

    def test_components_present(self, risk_config, sample_inputs):
        """Result DataFrame must contain all four component columns."""
        calc = RiskScoreCalculator(risk_config)
        result = calc.compute(**sample_inputs)
        for col in ["V_norm", "R_regime", "S_neg", "D_drawdown"]:
            assert col in result.columns, f"Missing component column: {col}"
