"""
data_pipeline/ingestion.py

Market data and news ingestion from Yahoo Finance and RSS feeds.
"""

import os
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import pandas as pd
import requests
import yfinance as yf
import yaml
from loguru import logger
from sqlalchemy import create_engine, text


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class MarketDataIngester:
    """Downloads and stores OHLCV data from Yahoo Finance."""

    def __init__(self, config: dict):
        self.cfg = config
        self.raw_dir = Path(config["data"]["raw_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, asset: str, start: str, end: str | None = None) -> pd.DataFrame:
        """
        Download OHLCV data for a single asset.

        Args:
            asset: Ticker symbol (e.g. 'SPY', '^VIX').
            start: Start date string 'YYYY-MM-DD'.
            end:   End date string 'YYYY-MM-DD'. Defaults to today.

        Returns:
            DataFrame with columns [Open, High, Low, Close, Volume, Adj Close].
        """
        logger.info(f"Fetching market data for {asset} from {start} to {end or 'today'}")
        ticker = yf.Ticker(asset)
        df = ticker.history(start=start, end=end, auto_adjust=False)

        if df.empty:
            logger.warning(f"No data returned for {asset}")
            return df

        df.index = pd.to_datetime(df.index, utc=True).normalize()
        df = df[["Open", "High", "Low", "Close", "Volume", "Adj Close"]].copy()
        df.index.name = "Date"
        return df

    def save_raw(self, df: pd.DataFrame, asset: str) -> Path:
        """
        Persist raw data to a timestamped parquet file. Never overwrites.

        Args:
            df:    DataFrame to save.
            asset: Ticker symbol used to build the filename.

        Returns:
            Path to the saved file.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_asset = asset.replace("^", "").replace("-", "_")
        path = self.raw_dir / f"{safe_asset}_{ts}.parquet"
        df.to_parquet(path)
        logger.info(f"Saved raw data to {path}")
        return path

    def run(self, assets: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """
        Full ingestion run for all configured assets.

        Returns:
            Dict mapping asset ticker to its DataFrame.
        """
        assets = assets or self.cfg["data"]["assets"]
        start = self.cfg["data"]["start_date"]
        results = {}

        for asset in assets:
            try:
                df = self.fetch(asset, start=start)
                if not df.empty:
                    self.save_raw(df, asset)
                    results[asset] = df
            except Exception as exc:
                logger.error(f"Ingestion failed for {asset}: {exc}")

        return results


class NewsIngester:
    """Fetches news headlines from RSS feeds and stores them in PostgreSQL."""

    DEFAULT_FEEDS = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=QQQ&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.bloomberg.com/markets/news.rss",
    ]

    def __init__(self, config: dict):
        self.cfg = config
        db_url = config["data"]["db_url"]
        self.engine = create_engine(db_url, pool_pre_ping=True)
        self.table = config["data"]["news_table"]
        self._ensure_table()

    def _ensure_table(self) -> None:
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.table} (
            id          SERIAL PRIMARY KEY,
            headline    TEXT        NOT NULL,
            published   TIMESTAMPTZ NOT NULL,
            source      TEXT,
            asset_tag   TEXT,
            url         TEXT,
            content_hash CHAR(64)  UNIQUE
        );
        CREATE INDEX IF NOT EXISTS idx_news_published ON {self.table} (published);
        CREATE INDEX IF NOT EXISTS idx_news_asset ON {self.table} (asset_tag);
        """
        with self.engine.begin() as conn:
            conn.execute(text(ddl))

    def _content_hash(self, headline: str, published: str) -> str:
        return hashlib.sha256(f"{headline}{published}".encode()).hexdigest()

    def _infer_asset_tag(self, text: str) -> str | None:
        mapping = {"SPY": "SPY", "QQQ": "QQQ", "Bitcoin": "BTC-USD", "BTC": "BTC-USD",
                   "S&P": "SPY", "Nasdaq": "QQQ"}
        for keyword, tag in mapping.items():
            if keyword.lower() in text.lower():
                return tag
        return None

    def fetch_feed(self, url: str) -> list[dict]:
        logger.info(f"Parsing RSS feed: {url}")
        parsed = feedparser.parse(url)
        rows = []
        for entry in parsed.entries:
            headline = entry.get("title", "")
            published_struct = entry.get("published_parsed")
            if not published_struct:
                continue
            published = datetime(*published_struct[:6], tzinfo=timezone.utc)
            rows.append({
                "headline": headline,
                "published": published,
                "source": parsed.feed.get("title", url),
                "asset_tag": self._infer_asset_tag(headline),
                "url": entry.get("link"),
                "content_hash": self._content_hash(headline, str(published)),
            })
        return rows

    def upsert(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        inserted = 0
        insert_sql = text(f"""
            INSERT INTO {self.table} (headline, published, source, asset_tag, url, content_hash)
            VALUES (:headline, :published, :source, :asset_tag, :url, :content_hash)
            ON CONFLICT (content_hash) DO NOTHING
        """)
        with self.engine.begin() as conn:
            for row in rows:
                result = conn.execute(insert_sql, row)
                inserted += result.rowcount
        logger.info(f"Inserted {inserted} new headlines")
        return inserted

    def run(self, feeds: list[str] | None = None, delay_sec: float = 1.0) -> int:
        feeds = feeds or self.DEFAULT_FEEDS
        total = 0
        for feed_url in feeds:
            try:
                rows = self.fetch_feed(feed_url)
                total += self.upsert(rows)
                time.sleep(delay_sec)
            except Exception as exc:
                logger.error(f"Failed to ingest feed {feed_url}: {exc}")
        return total

    def load_since(self, since: str, asset_tag: str | None = None) -> pd.DataFrame:
        """Load headlines from the DB since a given date."""
        query = f"SELECT * FROM {self.table} WHERE published >= :since"
        params = {"since": since}
        if asset_tag:
            query += " AND asset_tag = :asset_tag"
            params["asset_tag"] = asset_tag
        return pd.read_sql(text(query), self.engine, params=params)


if __name__ == "__main__":
    cfg = load_config()
    ingester = MarketDataIngester(cfg)
    data = ingester.run()
    logger.info(f"Ingested {len(data)} assets")
