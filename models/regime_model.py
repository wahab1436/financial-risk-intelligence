"""
models/regime_model.py

Hidden Markov Model-based market regime detection.
Supports HMM (primary), GMM, and KMeans (baseline).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from loguru import logger
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    """
    Detects latent market regimes from unlabeled market data.

    Regime labels are not predefined; they emerge from model structure.
    Regimes are post-hoc annotated based on their statistical properties.
    """

    MODEL_CHOICES = ("hmm", "gmm", "kmeans")

    def __init__(self, config: dict):
        self.cfg = config
        self.n_states = config["regime"]["n_states"]
        self.n_iter = config["regime"]["n_iter"]
        self.cov_type = config["regime"]["covariance_type"]
        self.model_type = config["regime"]["model"]
        self.scaler = StandardScaler()
        self.model = None
        self.regime_map: dict[int, str] = {}

    def _build_feature_matrix(self, features: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
        """
        Select and prepare features for regime detection.
        Uses returns, rolling volatility, and drawdown as inputs.
        """
        cols = ["log_return", "vol_std_30d", "drawdown_30d"]
        available = [c for c in cols if c in features.columns]
        if not available:
            raise ValueError(f"None of the required regime features found. Expected: {cols}")

        sub = features[available].dropna()
        X = self.scaler.fit_transform(sub.values)
        return X, sub.index

    def fit(self, features: pd.DataFrame) -> "RegimeDetector":
        """
        Train the regime detection model.

        Args:
            features: Market feature DataFrame (output of MarketFeatureEngineer).

        Returns:
            Self (fitted).
        """
        X, valid_index = self._build_feature_matrix(features)
        logger.info(f"Fitting {self.model_type.upper()} with {self.n_states} states on {len(X)} samples")

        if self.model_type == "hmm":
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.cov_type,
                n_iter=self.n_iter,
                random_state=self.cfg["system"]["random_seed"],
            )
            self.model.fit(X)

        elif self.model_type == "gmm":
            self.model = GaussianMixture(
                n_components=self.n_states,
                covariance_type=self.cov_type,
                n_init=5,
                random_state=self.cfg["system"]["random_seed"],
            )
            self.model.fit(X)

        elif self.model_type == "kmeans":
            self.model = KMeans(
                n_clusters=self.n_states,
                n_init=10,
                random_state=self.cfg["system"]["random_seed"],
            )
            self.model.fit(X)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Choose from {self.MODEL_CHOICES}")

        # Post-fit regime annotation
        labels = self.predict_labels(features)
        self._annotate_regimes(features, labels)
        logger.info(f"Regime map: {self.regime_map}")
        return self

    def predict_labels(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict integer regime labels for new data.

        Args:
            features: Feature DataFrame.

        Returns:
            Series of integer regime labels aligned to feature index.
        """
        X, valid_index = self._build_feature_matrix(features)
        if self.model_type == "hmm":
            labels = self.model.predict(X)
        elif self.model_type == "gmm":
            labels = self.model.predict(X)
        elif self.model_type == "kmeans":
            labels = self.model.predict(X)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return pd.Series(labels, index=valid_index, name="regime_label")

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Return regime membership probabilities.

        Args:
            features: Feature DataFrame.

        Returns:
            DataFrame of shape (n_samples, n_states) with probability columns.
        """
        X, valid_index = self._build_feature_matrix(features)
        if self.model_type == "hmm":
            proba = self.model.predict_proba(X)
        elif self.model_type == "gmm":
            proba = self.model.predict_proba(X)
        elif self.model_type == "kmeans":
            # KMeans has no soft probabilities; use distance-based approximation
            distances = self.model.transform(X)
            inv_distances = 1.0 / (distances + 1e-8)
            proba = inv_distances / inv_distances.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        cols = [f"regime_prob_{i}" for i in range(self.n_states)]
        return pd.DataFrame(proba, index=valid_index, columns=cols)

    def _annotate_regimes(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """
        Assign qualitative names to integer regime labels based on
        their mean volatility and mean return characteristics.
        """
        vol_col = "vol_std_30d" if "vol_std_30d" in features.columns else None
        ret_col = "log_return" if "log_return" in features.columns else None

        regime_stats = {}
        for state in range(self.n_states):
            mask = labels == state
            stats = {}
            if vol_col:
                stats["mean_vol"] = features.loc[labels.index[mask], vol_col].mean()
            if ret_col:
                stats["mean_ret"] = features.loc[labels.index[mask], ret_col].mean()
            regime_stats[state] = stats

        # Sort by volatility descending to label regimes
        if vol_col:
            sorted_by_vol = sorted(regime_stats.items(), key=lambda x: x[1].get("mean_vol", 0), reverse=True)
            labels_by_vol = ["high_vol", "bear", "sideways", "bull"]
            for i, (state, _) in enumerate(sorted_by_vol):
                self.regime_map[state] = labels_by_vol[i] if i < len(labels_by_vol) else f"regime_{state}"
        else:
            for state in range(self.n_states):
                self.regime_map[state] = f"regime_{state}"

    def get_transition_matrix(self) -> pd.DataFrame | None:
        """Return the HMM transition matrix as a DataFrame, or None for other models."""
        if self.model_type == "hmm" and self.model is not None:
            tm = self.model.transmat_
            labels = [self.regime_map.get(i, str(i)) for i in range(self.n_states)]
            return pd.DataFrame(tm, index=labels, columns=labels)
        return None

    def get_regime_risk_encoding(self) -> dict[str, float]:
        """Return the risk-score encoding for each regime name."""
        return self.cfg["risk_score"]["regime_encoding"]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "regime_map": self.regime_map, "cfg": self.cfg}, f)
        logger.info(f"Regime model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "RegimeDetector":
        with open(path, "rb") as f:
            state = pickle.load(f)
        detector = cls(state["cfg"])
        detector.model = state["model"]
        detector.scaler = state["scaler"]
        detector.regime_map = state["regime_map"]
        logger.info(f"Regime model loaded from {path}")
        return detector
