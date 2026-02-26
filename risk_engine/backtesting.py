"""
risk_engine/backtesting.py

Backtests the composite risk score against historical market drawdowns.
Validates that risk score spikes precede major drawdowns.
"""

import numpy as np
import pandas as pd
from loguru import logger


KNOWN_CRISIS_DATES = {
    "GFC_2008":         ("2008-09-01", "2009-03-31"),
    "COVID_2020":       ("2020-02-19", "2020-03-23"),
    "Rate_Hike_2022":   ("2022-01-01", "2022-10-13"),
}


def compute_drawdown_series(close: pd.Series, window: int = 252) -> pd.Series:
    """
    Compute rolling drawdown from rolling peak.

    Args:
        close:  Close price series.
        window: Lookback window in trading days.

    Returns:
        Series of drawdown values (<= 0).
    """
    rolling_peak = close.rolling(window, min_periods=1).max()
    drawdown = (close - rolling_peak) / rolling_peak
    return drawdown


def compute_lead_correlation(
    risk_score: pd.Series, returns: pd.Series, max_lag: int = 20
) -> pd.DataFrame:
    """
    Compute cross-correlation between risk score and negative future returns
    at various lead lags. Validates risk score as a leading indicator.

    Args:
        risk_score: Composite risk series.
        returns:    Daily log returns.
        max_lag:    Maximum lag to test (trading days).

    Returns:
        DataFrame with columns [lag, correlation].
    """
    rows = []
    neg_returns = -returns  # Higher = worse outcome (positive = risk event)
    for lag in range(0, max_lag + 1):
        corr = risk_score.corr(neg_returns.shift(-lag))
        rows.append({"lag_days": lag, "correlation": corr})
    result = pd.DataFrame(rows)
    optimal_lag = result.loc[result["correlation"].idxmax(), "lag_days"]
    logger.info(f"Optimal predictive lag: {optimal_lag} days (corr={result['correlation'].max():.4f})")
    return result


def crisis_period_analysis(
    risk_score: pd.Series, close: pd.Series
) -> pd.DataFrame:
    """
    Evaluate risk score behavior during known crisis periods.

    For each crisis, compute:
      - Mean risk score in the pre-crisis window (30 days before)
      - Mean risk score during the crisis
      - Max drawdown during the crisis

    Args:
        risk_score: Composite risk series.
        close:      Asset close price series.

    Returns:
        DataFrame summarizing crisis-period performance.
    """
    rows = []
    for name, (start, end) in KNOWN_CRISIS_DATES.items():
        try:
            crisis_mask = (risk_score.index >= start) & (risk_score.index <= end)
            pre_start = pd.Timestamp(start) - pd.Timedelta(days=30)
            pre_mask = (risk_score.index >= str(pre_start.date())) & (risk_score.index < start)

            if crisis_mask.sum() == 0:
                continue

            pre_risk = risk_score[pre_mask].mean() if pre_mask.sum() > 0 else np.nan
            crisis_risk = risk_score[crisis_mask].mean()

            crisis_close = close[crisis_mask]
            if len(crisis_close) > 1:
                peak = crisis_close.iloc[0]
                trough = crisis_close.min()
                max_dd = (trough - peak) / peak
            else:
                max_dd = np.nan

            rows.append({
                "crisis": name,
                "pre_crisis_avg_risk": round(pre_risk, 4) if not np.isnan(pre_risk) else None,
                "during_crisis_avg_risk": round(crisis_risk, 4),
                "risk_elevation": round(crisis_risk - pre_risk, 4) if not np.isnan(pre_risk) else None,
                "max_drawdown": round(max_dd, 4) if not np.isnan(max_dd) else None,
                "n_trading_days": int(crisis_mask.sum()),
            })
        except Exception as exc:
            logger.warning(f"Could not analyze crisis {name}: {exc}")

    if not rows:
        logger.warning("No crisis periods found in the provided data range")
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index("crisis")
    logger.info(f"Crisis period analysis:\n{result.to_string()}")
    return result


def rolling_risk_return_correlation(
    risk_score: pd.Series, returns: pd.Series, window: int = 63
) -> pd.Series:
    """
    Compute rolling correlation between risk score and realized negative returns.
    A consistently negative correlation confirms risk score tracks bad outcomes.

    Args:
        risk_score: Composite risk series.
        returns:    Daily log returns.
        window:     Rolling window in trading days.

    Returns:
        Rolling correlation series.
    """
    neg_returns = -returns
    corr = risk_score.rolling(window).corr(neg_returns)
    corr.name = f"risk_return_correlation_{window}d"
    return corr


def compute_risk_score_summary(risk_df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for the risk score time series.

    Args:
        risk_df: Output from RiskScoreCalculator.compute().

    Returns:
        Dict of summary statistics.
    """
    scores = risk_df["risk_score"].dropna()
    band_pcts = risk_df["risk_band"].value_counts(normalize=True).to_dict()

    return {
        "mean_risk_score": round(scores.mean(), 4),
        "std_risk_score": round(scores.std(), 4),
        "max_risk_score": round(scores.max(), 4),
        "min_risk_score": round(scores.min(), 4),
        "pct_time_in_high_or_severe": round(
            scores[scores >= 0.55].count() / len(scores), 4
        ),
        "risk_band_distribution": {k: round(v, 4) for k, v in band_pcts.items()},
        "latest_score": round(float(scores.iloc[-1]), 4),
        "latest_band": str(risk_df["risk_band"].iloc[-1]),
    }
