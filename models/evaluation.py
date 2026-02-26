"""
models/evaluation.py

Evaluation metrics and walk-forward validation reporting for all models.
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_volatility_metrics(results: pd.DataFrame) -> dict[str, float]:
    """
    Compute regression metrics for volatility forecasting walk-forward results.

    Args:
        results: DataFrame with columns [actual, predicted].

    Returns:
        Dict of metric name -> value.
    """
    y_true = results["actual"].values
    y_pred = results["predicted"].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Directional accuracy: did the forecast capture the direction of vol change?
    actual_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    directional_acc = np.mean(actual_direction == pred_direction)

    # Information coefficient (rank correlation)
    ic = pd.Series(y_true).rank().corr(pd.Series(y_pred).rank(), method="spearman")

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "directional_accuracy": float(directional_acc),
        "information_coefficient": float(ic),
        "n_predictions": len(y_true),
    }

    logger.info("Volatility Model Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return metrics


def compute_regime_stability(labels: pd.Series) -> dict[str, float]:
    """
    Evaluate regime detection stability.

    Args:
        labels: Series of integer regime labels indexed by date.

    Returns:
        Dict with stability metrics.
    """
    # Mean regime duration (consecutive days in same regime)
    changes = (labels != labels.shift(1)).cumsum()
    durations = labels.groupby(changes).count()
    mean_duration = durations.mean()

    # Transition count per year
    n_transitions = (labels != labels.shift(1)).sum()
    n_years = (labels.index[-1] - labels.index[0]).days / 365.25
    transitions_per_year = n_transitions / max(n_years, 0.01)

    # Regime distribution
    distribution = labels.value_counts(normalize=True).to_dict()

    metrics = {
        "mean_regime_duration_days": float(mean_duration),
        "transitions_per_year": float(transitions_per_year),
        "n_transitions_total": int(n_transitions),
        "regime_distribution": distribution,
    }

    logger.info("Regime Stability:")
    for k, v in metrics.items():
        if k != "regime_distribution":
            logger.info(f"  {k}: {v:.2f}")
        else:
            logger.info(f"  {k}: {v}")

    return metrics


def rolling_backtest_stability(
    results: pd.DataFrame, window: int = 126
) -> pd.DataFrame:
    """
    Compute rolling RMSE over a walk-forward results DataFrame.
    Useful for detecting structural breaks in model performance.

    Args:
        results: DataFrame with columns [actual, predicted].
        window:  Rolling window size in trading days.

    Returns:
        DataFrame with rolling RMSE values.
    """
    errors = (results["actual"] - results["predicted"]) ** 2
    rolling_rmse = errors.rolling(window).mean().apply(np.sqrt)
    return rolling_rmse.to_frame(name=f"rolling_rmse_{window}d")


def sentiment_evaluation(
    sentiment_daily: pd.Series, returns: pd.Series, lag: int = 1
) -> dict[str, float]:
    """
    Evaluate whether sentiment leads returns (predictive alignment).

    Args:
        sentiment_daily: Daily sentiment index [-1, 1].
        returns:         Daily log returns.
        lag:             Lag at which to evaluate correlation.

    Returns:
        Dict of evaluation metrics.
    """
    aligned = pd.concat([sentiment_daily, returns], axis=1).dropna()
    aligned.columns = ["sentiment", "returns"]

    # Correlation between lagged sentiment and future returns
    lead_corr = aligned["sentiment"].shift(lag).corr(aligned["returns"])

    # Hit rate: does positive sentiment precede positive returns?
    pos_sent = aligned["sentiment"].shift(lag) > 0
    pos_ret = aligned["returns"] > 0
    hit_rate = (pos_sent == pos_ret).mean()

    metrics = {
        "lag": lag,
        "lead_correlation": float(lead_corr),
        "hit_rate": float(hit_rate),
        "n_observations": len(aligned),
    }

    logger.info("Sentiment Evaluation:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics
