"""
models/volatility_model.py

7-day forward volatility forecasting using GARCH (baseline) and XGBoost (ML model).
Walk-forward validation only — no random train-test splits.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from arch import arch_model
from loguru import logger
from sklearn.preprocessing import StandardScaler


class GARCHVolatilityModel:
    """
    GARCH(p,q) baseline for volatility forecasting.
    Provides a benchmark for the XGBoost model.
    """

    def __init__(self, config: dict):
        self.p = config["volatility"]["garch_p"]
        self.q = config["volatility"]["garch_q"]
        self.horizon = config["volatility"]["forecast_horizon"]
        self.model_result = None

    def fit(self, log_returns: pd.Series) -> "GARCHVolatilityModel":
        """
        Fit GARCH model on log returns.

        Args:
            log_returns: Series of log returns (daily).
        """
        returns_pct = log_returns * 100  # arch expects percentage returns
        am = arch_model(returns_pct.dropna(), vol="GARCH", p=self.p, q=self.q, dist="normal")
        self.model_result = am.fit(disp="off", show_warning=False)
        logger.info(f"GARCH({self.p},{self.q}) fitted. AIC={self.model_result.aic:.2f}")
        return self

    def forecast(self, steps: int | None = None) -> float:
        """
        Forecast annualized volatility for the next `steps` days.

        Args:
            steps: Forecast horizon. Defaults to configured horizon.

        Returns:
            Scalar annualized volatility forecast.
        """
        steps = steps or self.horizon
        forecast = self.model_result.forecast(horizon=steps)
        variance_forecast = forecast.variance.values[-1]
        daily_vol = np.sqrt(variance_forecast.mean()) / 100  # back to decimal
        return daily_vol * np.sqrt(252)  # annualize


class XGBoostVolatilityModel:
    """
    XGBoost regressor for 7-day forward realized volatility prediction.

    Feature engineering is handled externally; this class handles:
      - Walk-forward training and prediction
      - Model persistence
      - Out-of-sample metrics computation
    """

    FEATURE_COLS = [
        "vol_std_7d", "vol_std_14d", "vol_std_30d",
        "realized_vol_7d", "realized_vol_30d",
        "log_return", "return_mean_7d", "return_mean_30d",
        "return_skew_30d", "atr", "rsi", "macd_diff",
        "volume_zscore", "regime",
        "sentiment_index",
    ]

    def __init__(self, config: dict):
        self.cfg = config
        self.horizon = config["volatility"]["forecast_horizon"]
        self.min_train_days = config["volatility"]["walk_forward_min_train_days"]
        xgb_cfg = config["volatility"]["xgboost"]
        self.xgb_params = {
            "n_estimators": xgb_cfg["n_estimators"],
            "learning_rate": xgb_cfg["learning_rate"],
            "max_depth": xgb_cfg["max_depth"],
            "subsample": xgb_cfg["subsample"],
            "colsample_bytree": xgb_cfg["colsample_bytree"],
            "random_state": xgb_cfg["random_state"],
            "n_jobs": -1,
            "objective": "reg:squarederror",
        }
        self.model: xgb.XGBRegressor | None = None
        self.scaler = StandardScaler()
        self.feature_importances_: pd.Series | None = None

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select available feature columns from the combined feature DataFrame."""
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        return df[available]

    def walk_forward_fit_predict(
        self, features: pd.DataFrame, target: pd.Series
    ) -> pd.DataFrame:
        """
        Walk-forward (expanding window) training and prediction.

        For each step t starting after min_train_days:
          - Train on all data from [0, t-1]
          - Predict for t

        Args:
            features: Feature DataFrame aligned to target index.
            target:   Forward volatility Series (the prediction target).

        Returns:
            DataFrame with columns [actual, predicted, date].
        """
        combined = self._get_features(features).join(target, how="inner").dropna()
        target_col = target.name
        feature_cols = [c for c in combined.columns if c != target_col]

        X_all = combined[feature_cols].values
        y_all = combined[target_col].values
        dates = combined.index

        predictions = []
        logger.info(
            f"Starting walk-forward validation: {len(X_all)} samples, "
            f"min_train={self.min_train_days}, step=1"
        )

        for t in range(self.min_train_days, len(X_all)):
            X_train = X_all[:t]
            y_train = y_all[:t]
            X_test = X_all[t : t + 1]

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            model = xgb.XGBRegressor(**self.xgb_params)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)[0]
            predictions.append({"date": dates[t], "actual": y_all[t], "predicted": y_pred})

        results_df = pd.DataFrame(predictions).set_index("date")
        logger.info(f"Walk-forward complete: {len(results_df)} out-of-sample predictions")

        # Fit final model on all data for production use
        self._fit_final(X_all, y_all, feature_cols)
        return results_df

    def _fit_final(self, X: np.ndarray, y: np.ndarray, feature_cols: list[str]) -> None:
        """Fit the production model on the full dataset."""
        X_scaled = self.scaler.fit_transform(X)
        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(X_scaled, y)
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
        logger.info("Final XGBoost model fitted on full dataset")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate volatility predictions using the fitted production model.

        Args:
            features: Feature DataFrame for prediction period.

        Returns:
            Array of predicted forward volatility values.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call walk_forward_fit_predict() first.")
        X = self._get_features(features).values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_importances": self.feature_importances_,
                "cfg": self.cfg,
            }, f)
        logger.info(f"Volatility model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "XGBoostVolatilityModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(state["cfg"])
        obj.model = state["model"]
        obj.scaler = state["scaler"]
        obj.feature_importances_ = state["feature_importances"]
        logger.info(f"Volatility model loaded from {path}")
        return obj
