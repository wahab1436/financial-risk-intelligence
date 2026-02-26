"""
feature_engineering/sentiment_features.py

Generates sentiment features from financial news headlines using FinBERT.
Produces a daily sentiment index per asset aligned to market data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class FinBERTSentimentPipeline:
    """
    Runs FinBERT inference on financial headlines and aggregates
    results into a daily sentiment index per asset.

    Sentiment labels from ProsusAI/finbert: positive, neutral, negative.
    """

    LABEL_SCORES = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    def __init__(self, config: dict):
        self.cfg = config
        model_name = config["sentiment"]["model_name"]
        self.batch_size = config["sentiment"]["batch_size"]
        self.max_length = config["sentiment"]["max_length"]
        self.agg = config["sentiment"]["aggregation"]

        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading FinBERT model '{model_name}' on device={device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            top_k=None,
            batch_size=self.batch_size,
            truncation=True,
            max_length=self.max_length,
        )

    @staticmethod
    def _clean(text: str) -> str:
        """Basic normalization: strip whitespace, remove null bytes."""
        if not isinstance(text, str):
            return ""
        return text.replace("\x00", "").strip()

    def score_headlines(self, headlines: list[str]) -> pd.DataFrame:
        """
        Run FinBERT inference on a list of headlines.

        Args:
            headlines: List of raw headline strings.

        Returns:
            DataFrame with columns [positive, neutral, negative, polarity, confidence].
        """
        cleaned = [self._clean(h) for h in headlines]
        # Filter empty strings but keep index alignment
        valid_mask = [bool(h) for h in cleaned]
        valid_texts = [h for h, ok in zip(cleaned, valid_mask) if ok]

        if not valid_texts:
            logger.warning("No valid headlines to score")
            return pd.DataFrame(columns=["positive", "neutral", "negative", "polarity", "confidence"])

        raw_outputs = self.pipe(valid_texts)
        records = []
        for output in raw_outputs:
            row = {item["label"].lower(): item["score"] for item in output}
            dominant = max(output, key=lambda x: x["score"])
            row["polarity"] = self.LABEL_SCORES.get(dominant["label"].lower(), 0.0)
            row["confidence"] = dominant["score"]
            records.append(row)

        result = pd.DataFrame(records)
        # Re-expand to full list (insert zeros for filtered-out empties)
        full_records = []
        valid_iter = iter(records)
        for ok in valid_mask:
            if ok:
                full_records.append(next(valid_iter))
            else:
                full_records.append({"positive": 0.0, "neutral": 1.0, "negative": 0.0, "polarity": 0.0, "confidence": 0.0})

        return pd.DataFrame(full_records)

    def compute_daily_index(
        self, news_df: pd.DataFrame, date_col: str = "published", text_col: str = "headline"
    ) -> pd.Series:
        """
        Aggregate headline-level sentiment into a daily index.

        Aggregation method (from config):
          - 'weighted_mean': confidence-weighted average polarity
          - 'mean': simple mean of polarity scores

        Args:
            news_df:  DataFrame with headline text and published datetime.
            date_col: Column name for publication datetime.
            text_col: Column name for headline text.

        Returns:
            Series indexed by date with daily sentiment score in [-1, 1].
        """
        if news_df.empty:
            logger.warning("Empty news DataFrame passed to compute_daily_index")
            return pd.Series(dtype=float)

        news_df = news_df.copy()
        news_df["date"] = pd.to_datetime(news_df[date_col]).dt.normalize()

        scores = self.score_headlines(news_df[text_col].tolist())
        news_df["polarity"] = scores["polarity"].values
        news_df["confidence"] = scores["confidence"].values

        if self.agg == "weighted_mean":
            def weighted_mean(group):
                weights = group["confidence"]
                if weights.sum() == 0:
                    return 0.0
                return np.average(group["polarity"], weights=weights)
            daily = news_df.groupby("date").apply(weighted_mean)
        else:
            daily = news_df.groupby("date")["polarity"].mean()

        daily.name = "sentiment_index"
        logger.info(f"Computed daily sentiment for {len(daily)} days")
        return daily

    def align_to_market(
        self, sentiment: pd.Series, market_index: pd.Index, lag: int = 1
    ) -> pd.Series:
        """
        Align sentiment to market trading days with lag to prevent leakage.

        News from day T is available for use on day T+lag only.

        Args:
            sentiment:    Daily sentiment Series.
            market_index: DatetimeIndex of trading days.
            lag:          Days to shift forward (default 1).

        Returns:
            Sentiment Series reindexed to market_index with lag applied.
        """
        aligned = sentiment.reindex(market_index).ffill().shift(lag)
        aligned.name = "sentiment_index_lagged"
        return aligned
