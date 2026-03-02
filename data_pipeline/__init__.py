"""
data_pipeline package

Provides ingestion, validation, and preprocessing utilities
for the financial risk intelligence system.
"""

# ── Ingestion ─────────────────────────────────────────────
from .ingestion import (
    load_config,
    MarketDataIngester,
    NewsIngester,
)

# ── Validation ────────────────────────────────────────────
from .validation import (
    ValidationResult,
    MarketDataValidator,
    validate_all,
)

# ── Preprocessing ─────────────────────────────────────────
from .preprocessing import (
    MarketDataPreprocessor,
    align_assets,
)

__all__ = [
    # ingestion
    "load_config",
    "MarketDataIngester",
    "NewsIngester",

    # validation
    "ValidationResult",
    "MarketDataValidator",
    "validate_all",

    # preprocessing
    "MarketDataPreprocessor",
    "align_assets",
]
