"""
risk_engine/risk_calculator.py

Composite risk score computation integrating volatility, regime, sentiment,
and drawdown into a single normalized metric on [0, 1].
"""

import numpy as np
import pandas as pd
from loguru import logger


class RiskScoreCalculator:
    """
    Computes the composite risk score:

        Risk Score = w1 * V_norm + w2 * R_regime + w3 * S_neg + w4 * D_drawdown

    All components are independently normalized to [0, 1] before aggregation.
    Weights are configured in config.yaml and calibrated on historical stress periods.
    """

    RISK_BANDS = {
        "Low":         (0.00, 0.30),
        "Elevated":    (0.30, 0.55),
        "High":        (0.55, 0.75),
        "Severe":      (0.75, 1.01),
    }

    def __init__(self, config: dict):
        weights = config["risk_score"]["weights"]
        self.w_vol = weights["volatility"]
        self.w_regime = weights["regime"]
        self.w_sentiment = weights["sentiment"]
        self.w_drawdown = weights["drawdown"]
        self.regime_encoding = config["risk_score"]["regime_encoding"]
        self.norm_window = config["risk_score"]["normalization_window"]

        # Validate weights sum to 1
        total = self.w_vol + self.w_regime + self.w_sentiment + self.w_drawdown
        if not np.isclose(total, 1.0, atol=0.01):
            logger.warning(f"Risk weights sum to {total:.4f}, expected 1.0. Normalizing.")
            self.w_vol /= total
            self.w_regime /= total
            self.w_sentiment /= total
            self.w_drawdown /= total

    def _minmax_normalize(self, series: pd.Series) -> pd.Series:
        """
        Rolling min-max normalization using the past norm_window observations.
        Prevents look-ahead: normalization bounds computed only from historical data.
        """
        roll_min = series.rolling(self.norm_window, min_periods=30).min()
        roll_max = series.rolling(self.norm_window, min_periods=30).max()
        denom = (roll_max - roll_min).replace(0, 1e-8)
        normalized = (series - roll_min) / denom
        return normalized.clip(0, 1)

    def _encode_regime(self, regime_labels: pd.Series) -> pd.Series:
        """
        Convert categorical regime labels to risk scores using the encoding map.

        Args:
            regime_labels: Series of string regime names.

        Returns:
            Series of float risk encodings in [0, 1].
        """
        return regime_labels.map(self.regime_encoding).fillna(0.5)

    def _compute_negative_sentiment(self, sentiment_index: pd.Series) -> pd.Series:
        """
        Convert sentiment index [-1, 1] to a risk component in [0, 1].
        Negative sentiment -> high risk; positive -> low risk.
        """
        # Map [-1, 1] to [1, 0] (invert and rescale)
        return ((1.0 - sentiment_index) / 2.0).clip(0, 1)

    def compute(
        self,
        forecasted_vol: pd.Series,
        regime_labels: pd.Series,
        sentiment_index: pd.Series,
        drawdown: pd.Series,
        vix: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Compute composite risk scores for each date.

        Args:
            forecasted_vol:  Predicted forward volatility series.
            regime_labels:   String regime label series.
            sentiment_index: Daily sentiment index in [-1, 1].
            drawdown:        Current drawdown series (negative values).
            vix:             Optional VIX level series.

        Returns:
            DataFrame with individual components and composite score.
        """
        # Align all series to a common index
        combined = pd.DataFrame({
            "vol": forecasted_vol,
            "regime": regime_labels,
            "sentiment": sentiment_index,
            "drawdown": drawdown,
        }).dropna(subset=["vol"])

        if vix is not None:
            combined["vix"] = vix.reindex(combined.index)

        # Component 1: Normalized volatility
        combined["V_norm"] = self._minmax_normalize(combined["vol"])

        # Component 2: Regime risk encoding
        combined["R_regime"] = self._encode_regime(combined["regime"])

        # Component 3: Negative sentiment (inverted)
        combined["S_neg"] = self._compute_negative_sentiment(
            combined["sentiment"].fillna(0)
        )

        # Component 4: Drawdown magnitude (drawdown is negative; abs for risk)
        combined["D_drawdown"] = self._minmax_normalize(combined["drawdown"].abs())

        # Composite risk score
        combined["risk_score"] = (
            self.w_vol       * combined["V_norm"]
            + self.w_regime  * combined["R_regime"]
            + self.w_sentiment * combined["S_neg"]
            + self.w_drawdown * combined["D_drawdown"]
        ).clip(0, 1)

        # Risk classification band
        combined["risk_band"] = combined["risk_score"].apply(self._classify)

        output_cols = ["V_norm", "R_regime", "S_neg", "D_drawdown", "risk_score", "risk_band"]
        logger.info(
            f"Risk scores computed for {len(combined)} dates. "
            f"Latest score: {combined['risk_score'].iloc[-1]:.3f} "
            f"({combined['risk_band'].iloc[-1]})"
        )
        return combined[output_cols]

    def _classify(self, score: float) -> str:
        for band, (lo, hi) in self.RISK_BANDS.items():
            if lo <= score < hi:
                return band
        return "Severe"

    def latest_risk_payload(self, risk_df: pd.DataFrame) -> dict:
        """
        Return a structured dict of the most recent risk snapshot
        for use in LLM report generation.
        """
        row = risk_df.iloc[-1]
        return {
            "date": str(risk_df.index[-1].date()),
            "risk_score": round(float(row["risk_score"]), 4),
            "risk_band": str(row["risk_band"]),
            "components": {
                "normalized_volatility": round(float(row["V_norm"]), 4),
                "regime_risk": round(float(row["R_regime"]), 4),
                "negative_sentiment": round(float(row["S_neg"]), 4),
                "drawdown_risk": round(float(row["D_drawdown"]), 4),
            },
            "weights": {
                "volatility": self.w_vol,
                "regime": self.w_regime,
                "sentiment": self.w_sentiment,
                "drawdown": self.w_drawdown,
            },
        }
