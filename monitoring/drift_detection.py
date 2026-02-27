"""
monitoring/drift_detection.py

Monitors feature distribution drift and regime structural changes.
Triggers retraining alerts when drift thresholds are exceeded.
"""

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


class DriftDetector:
    """
    Detects distribution drift in features and model outputs.

    Uses the Kolmogorov-Smirnov two-sample test to compare the
    reference distribution (training period) against the current window.
    """

    def __init__(self, config: dict):
        self.pvalue_threshold = config["monitoring"]["ks_pvalue_threshold"]
        self.regime_freq_threshold = config["monitoring"]["regime_freq_change_threshold"]
        self.reference_data: dict[str, pd.Series] = {}

    def set_reference(self, features: pd.DataFrame) -> None:
        """
        Store the training-period feature distributions as reference.

        Args:
            features: Feature DataFrame from the training window.
        """
        for col in features.select_dtypes(include=[np.number]).columns:
            self.reference_data[col] = features[col].dropna()
        logger.info(f"Reference distributions set for {len(self.reference_data)} features")

    def detect_feature_drift(
        self, current_features: pd.DataFrame, window: int = 30
    ) -> pd.DataFrame:
        """
        Run KS test for each numeric feature comparing training vs recent window.

        Args:
            current_features: Recent feature DataFrame.
            window:           Number of recent rows to use as the current window.

        Returns:
            DataFrame with columns [feature, ks_statistic, p_value, drifted].
        """
        if not self.reference_data:
            raise RuntimeError("Call set_reference() before detect_feature_drift()")

        recent = current_features.tail(window)
        rows = []

        for col in self.reference_data:
            if col not in recent.columns:
                continue
            ref = self.reference_data[col].values
            cur = recent[col].dropna().values
            if len(cur) < 5:
                continue
            ks_stat, p_value = stats.ks_2samp(ref, cur)
            drifted = p_value < self.pvalue_threshold
            rows.append({
                "feature": col,
                "ks_statistic": round(ks_stat, 4),
                "p_value": round(p_value, 6),
                "drifted": drifted,
            })
            if drifted:
                logger.warning(
                    f"DRIFT DETECTED | Feature: {col} | KS={ks_stat:.4f} | p={p_value:.6f}"
                )

        result = pd.DataFrame(rows)
        n_drifted = result["drifted"].sum() if not result.empty else 0
        logger.info(f"Drift check complete: {n_drifted}/{len(rows)} features drifted")
        return result

    def detect_regime_drift(
        self, historical_labels: pd.Series, recent_labels: pd.Series
    ) -> dict:
        """
        Detect changes in regime transition frequency.
        A large shift in how often regimes change may indicate structural market change.

        Args:
            historical_labels: Regime labels from the training period.
            recent_labels:     Regime labels from the recent window.

        Returns:
            Dict with transition frequency comparison and drift flag.
        """
        def transition_rate(labels: pd.Series) -> float:
            return (labels != labels.shift(1)).mean()

        hist_rate = transition_rate(historical_labels)
        recent_rate = transition_rate(recent_labels)
        relative_change = abs(recent_rate - hist_rate) / max(hist_rate, 1e-8)
        drifted = relative_change > self.regime_freq_threshold

        result = {
            "historical_transition_rate": round(hist_rate, 4),
            "recent_transition_rate": round(recent_rate, 4),
            "relative_change": round(relative_change, 4),
            "drifted": drifted,
        }

        if drifted:
            logger.warning(
                f"REGIME DRIFT DETECTED | Historical rate: {hist_rate:.4f} | "
                f"Recent rate: {recent_rate:.4f} | Change: {relative_change:.1%}"
            )
        else:
            logger.info(f"Regime drift check passed | Change: {relative_change:.1%}")

        return result

    def detect_volatility_structural_shift(
        self, vol_series: pd.Series, window: int = 63
    ) -> dict:
        """
        Detect structural shifts in volatility using a Chow-test approximation.
        Compares the recent window mean/std against the historical mean/std.

        Args:
            vol_series: Historical realized volatility series.
            window:     Recent window size in trading days.

        Returns:
            Dict with shift statistics and drift flag.
        """
        if len(vol_series) < window * 2:
            return {"drifted": False, "reason": "Insufficient data"}

        historical = vol_series.iloc[:-window]
        recent = vol_series.iloc[-window:]

        hist_mean, hist_std = historical.mean(), historical.std()
        recent_mean = recent.mean()

        # Z-score of recent mean relative to historical distribution
        z_score = abs(recent_mean - hist_mean) / max(hist_std, 1e-8)
        drifted = z_score > 2.5  # ~1% tail

        result = {
            "historical_vol_mean": round(hist_mean, 6),
            "recent_vol_mean": round(recent_mean, 6),
            "z_score": round(z_score, 4),
            "drifted": drifted,
        }

        if drifted:
            logger.warning(
                f"VOLATILITY STRUCTURAL SHIFT | Z-score: {z_score:.2f} | "
                f"Historical mean: {hist_mean:.4f} | Recent mean: {recent_mean:.4f}"
            )

        return result

    def should_retrain(
        self,
        drift_report: pd.DataFrame,
        regime_drift: dict,
        vol_shift: dict,
        feature_drift_pct_threshold: float = 0.30,
    ) -> tuple[bool, str]:
        """
        Determine whether model retraining should be triggered.

        Args:
            drift_report:               Output of detect_feature_drift().
            regime_drift:               Output of detect_regime_drift().
            vol_shift:                  Output of detect_volatility_structural_shift().
            feature_drift_pct_threshold: Fraction of features that must drift.

        Returns:
            Tuple of (should_retrain: bool, reason: str).
        """
        reasons = []

        if not drift_report.empty:
            pct_drifted = drift_report["drifted"].mean()
            if pct_drifted >= feature_drift_pct_threshold:
                reasons.append(
                    f"{pct_drifted:.0%} of features drifted (threshold {feature_drift_pct_threshold:.0%})"
                )

        if regime_drift.get("drifted"):
            reasons.append(
                f"Regime transition rate changed by {regime_drift.get('relative_change', 0):.1%}"
            )

        if vol_shift.get("drifted"):
            reasons.append(
                f"Volatility structural shift detected (z={vol_shift.get('z_score', 0):.2f})"
            )

        should_retrain = len(reasons) > 0
        reason_str = "; ".join(reasons) if reasons else "No drift detected"

        if should_retrain:
            logger.warning(f"RETRAINING RECOMMENDED: {reason_str}")
        else:
            logger.info("No retraining required")

        return should_retrain, reason_str
