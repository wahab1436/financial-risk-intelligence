"""
data_pipeline/validation.py

Data quality validation for ingested market data.
All validation results are logged and persisted.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from loguru import logger


@dataclass
class ValidationResult:
    asset: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] Validation for {self.asset}"]
        for check, ok in self.checks.items():
            icon = "OK" if ok else "FAIL"
            detail = self.details.get(check, "")
            lines.append(f"  [{icon}] {check}: {detail}")
        return "\n".join(lines)


class MarketDataValidator:
    """
    Validates a market data DataFrame against a set of quality checks.
    Each check is independent; failure of one does not halt others.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.max_zscore = config["validation"]["max_return_zscore"]
        self.max_null_pct = config["validation"]["max_null_pct"]
        self.min_trading_days = config["validation"]["min_trading_days_per_year"]

    def validate(self, df: pd.DataFrame, asset: str) -> ValidationResult:
        result = ValidationResult(asset=asset, passed=False)

        checks = {
            "no_missing_timestamps": self._check_missing_timestamps,
            "no_duplicate_rows":     self._check_duplicates,
            "null_values_within_threshold": self._check_nulls,
            "no_abnormal_return_spikes":    self._check_return_spikes,
            "non_trading_day_consistency":  self._check_trading_days,
            "schema_conformance":    self._check_schema,
            "temporal_ordering":     self._check_ordering,
        }

        all_passed = True
        for name, fn in checks.items():
            ok, detail = fn(df, asset)
            result.checks[name] = ok
            result.details[name] = detail
            if not ok:
                all_passed = False
                logger.warning(f"{asset} | Check [{name}] FAILED: {detail}")
            else:
                logger.debug(f"{asset} | Check [{name}] passed")

        result.passed = all_passed
        logger.info(f"\n{result.summary()}")
        return result

    def _check_schema(self, df: pd.DataFrame, asset: str) -> tuple[bool, str]:
        required = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        missing = required - set(df.columns)
        if missing:
            return False, f"Missing columns: {missing}"
        return True, "All required columns present"

    def _check_ordering(self, df: pd.DataFrame, asset: str) -> tuple[bool, str]:
        if not df.index.is_monotonic_increasing:
            return False, "Index is not monotonically increasing"
        return True, "Index is ordered"

    def _check_missing_timestamps(self, df: pd.DataFrame, asset: str) -> tuple[bool, str]:
        if df.index.empty:
            return False, "DataFrame is empty"
        freq_days = df.index.to_series().diff().dt.days.dropna()
        large_gaps = freq_days[freq_days > 7]
        if not large_gaps.empty:
            return False, f"{len(large_gaps)} gap(s) > 7 days found"
        return True, "No large timestamp gaps"

    def _check_duplicates(self, df: pd.DataFrame, asset: str) -> tuple[bool, str]:
        n_dupes = df.index.duplicated().sum()
        if n_dupes > 0:
            return False, f"{n_dupes} duplicate timestamp(s)"
        return True, "No duplicates"

    def _check_nulls(self, df: pd.DataFrame, asset: str) -> tuple[bool, str]:
        null_pct = df.isnull().mean().max()
        if null_pct > self.max_null_pct:
            worst_col = df.isnull().mean().idxmax()
            return False, f"Column '{worst_col}' has {null_pct:.1%} nulls (threshold {self.max_null_pct:.0%})"
        return True, f"Max null pct: {null_pct:.2%}"

    def _check_return_spikes(self, df: pd.DataFrame, asset: str) -> tuple[bool, str]:
        if "Close" not in df.columns or len(df) < 2:
            return True, "Insufficient data for return check"
        log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        mu, sigma = log_ret.mean(), log_ret.std()
        if sigma == 0:
            return True, "Zero std, skipping spike check"
        zscores = ((log_ret - mu) / sigma).abs()
        n_spikes = (zscores > self.max_zscore).sum()
        if n_spikes > 0:
            return False, f"{n_spikes} return spike(s) exceed {self.max_zscore}z"
        return True, f"No spikes > {self.max_zscore}z"

    def _check_trading_days(self, df: pd.DataFrame, asset: str) -> tuple[bool, str]:
        # BTC-USD trades 365 days/year; skip calendar check for it
        if "BTC" in asset.upper() or "USD" in asset.upper():
            return True, "Crypto asset — no trading calendar constraint"
        try:
            nyse = mcal.get_calendar("NYSE")
            start = df.index.min().strftime("%Y-%m-%d")
            end = df.index.max().strftime("%Y-%m-%d")
            schedule = nyse.schedule(start_date=start, end_date=end)
            expected_days = len(schedule)
            actual_days = len(df)
            tolerance = 0.95
            if actual_days < expected_days * tolerance:
                return False, (f"Only {actual_days} rows vs {expected_days} expected NYSE days "
                               f"({actual_days/expected_days:.1%} coverage)")
            return True, f"{actual_days}/{expected_days} NYSE days covered"
        except Exception as exc:
            return True, f"Calendar check skipped: {exc}"


def validate_all(data: dict[str, pd.DataFrame], config: dict) -> dict[str, ValidationResult]:
    """Run validation across all assets and return results dict."""
    validator = MarketDataValidator(config)
    results = {}
    for asset, df in data.items():
        results[asset] = validator.validate(df, asset)
    return results
