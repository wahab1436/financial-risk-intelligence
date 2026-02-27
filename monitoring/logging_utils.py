"""
monitoring/logging_utils.py

Structured logging (loguru) and MLflow experiment tracking.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger


def configure_logging(log_dir: str = "monitoring/logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.add(
        str(Path(log_dir) / "risk_system_{time:YYYY-MM-DD}.log"),
        rotation="1 day",
        retention="30 days",
        compression="zip",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}",
    )
    logger.info("Logging configured")


class PredictionLogger:
    """Appends every model prediction to a per-asset JSONL audit log."""

    def __init__(self, config: dict):
        self.log_dir = Path(config["monitoring"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        asset: str,
        model_name: str,
        model_version: str,
        input_summary: dict,
        prediction,
    ) -> None:
        record = {
            "ts":            datetime.now(timezone.utc).isoformat(),
            "asset":         asset,
            "model":         model_name,
            "version":       model_version,
            "input_summary": input_summary,
            "prediction":    prediction,
        }
        safe = asset.replace("^", "").replace("-", "_")
        path = self.log_dir / f"{safe}_predictions.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load(self, asset: str) -> pd.DataFrame:
        safe = asset.replace("^", "").replace("-", "_")
        path = self.log_dir / f"{safe}_predictions.jsonl"
        if not path.exists():
            return pd.DataFrame()
        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        return pd.DataFrame(lines)


class MLflowTracker:

    def __init__(self, config: dict):
        import mlflow
        mlflow.set_tracking_uri(config["monitoring"]["mlflow_tracking_uri"])
        mlflow.set_experiment(config["system"]["name"])
        self._mlflow = mlflow

    def log_regime(self, config: dict, metrics: dict, regime_map: dict,
                   transition_matrix: pd.DataFrame | None = None) -> str:
        with self._mlflow.start_run(run_name="regime_detection") as run:
            self._mlflow.log_params({
                "n_states":    config["regime"]["n_states"],
                "model_type":  config["regime"]["model"],
                "cov_type":    config["regime"]["covariance_type"],
                "regime_map":  str(regime_map),
            })
            flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            self._mlflow.log_metrics(flat)
            if transition_matrix is not None:
                p = "/tmp/transition_matrix.csv"
                transition_matrix.to_csv(p)
                self._mlflow.log_artifact(p)
            return run.info.run_id

    def log_volatility(self, config: dict, metrics: dict) -> str:
        with self._mlflow.start_run(run_name="volatility") as run:
            self._mlflow.log_params(config["volatility"]["xgboost"])
            flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            self._mlflow.log_metrics(flat)
            return run.info.run_id

    def log_risk(self, config: dict, summary: dict) -> str:
        with self._mlflow.start_run(run_name="risk_scoring") as run:
            self._mlflow.log_params(config["risk_score"]["weights"])
            flat = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
            self._mlflow.log_metrics(flat)
            return run.info.run_id
