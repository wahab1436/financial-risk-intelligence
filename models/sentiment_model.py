"""
models/sentiment_model.py

Wraps the FinBERT sentiment pipeline for training-time evaluation.
The inference pipeline lives in feature_engineering/sentiment_features.py.
This module provides labeled-dataset evaluation and model benchmarking.
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


class SentimentModelEvaluator:
    """
    Evaluates the FinBERT sentiment model against a labeled dataset.
    Also provides temporal stability analysis.
    """

    def __init__(self, sentiment_pipeline):
        """
        Args:
            sentiment_pipeline: Fitted FinBERTSentimentPipeline instance.
        """
        self.pipeline = sentiment_pipeline

    def evaluate(
        self, headlines: list[str], true_labels: list[str]
    ) -> dict[str, float]:
        """
        Evaluate classification performance against ground-truth labels.

        Args:
            headlines:    List of headline strings.
            true_labels:  List of labels: 'positive', 'negative', 'neutral'.

        Returns:
            Dict of precision, recall, and F1 per class + macro averages.
        """
        scores = self.pipeline.score_headlines(headlines)
        pred_labels = []
        for _, row in scores.iterrows():
            label_scores = {
                "positive": row.get("positive", 0.0),
                "negative": row.get("negative", 0.0),
                "neutral":  row.get("neutral", 0.0),
            }
            pred_labels.append(max(label_scores, key=label_scores.get))

        report = classification_report(
            true_labels, pred_labels,
            labels=["positive", "neutral", "negative"],
            output_dict=True,
        )

        metrics = {
            "precision_macro": round(precision_score(true_labels, pred_labels, average="macro", zero_division=0), 4),
            "recall_macro":    round(recall_score(true_labels, pred_labels, average="macro", zero_division=0), 4),
            "f1_macro":        round(f1_score(true_labels, pred_labels, average="macro", zero_division=0), 4),
            "classification_report": report,
        }

        logger.info(
            f"Sentiment Evaluation | Precision={metrics['precision_macro']:.4f} | "
            f"Recall={metrics['recall_macro']:.4f} | F1={metrics['f1_macro']:.4f}"
        )
        return metrics

    def temporal_stability(
        self, news_df: pd.DataFrame, date_col: str = "published"
    ) -> pd.DataFrame:
        """
        Measure sentiment score stability across time windows.
        High variance in mean daily sentiment over stable market periods
        suggests model instability rather than genuine market signal.

        Args:
            news_df:  News DataFrame with headline and date columns.
            date_col: Column containing publication timestamps.

        Returns:
            DataFrame of monthly mean and std of sentiment index.
        """
        daily = self.pipeline.compute_daily_index(news_df, date_col=date_col)
        monthly = daily.resample("ME").agg(["mean", "std"]).round(4)
        monthly.columns = ["monthly_mean_sentiment", "monthly_std_sentiment"]
        return monthly
