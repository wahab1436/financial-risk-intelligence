"""
llm_reports/report_generator.py

Calls the OpenAI API to generate structured risk reports from model outputs.
The LLM interprets pre-computed numbers — it does not perform calculations.
"""

import os
from datetime import datetime, timezone

from loguru import logger
from openai import OpenAI

from llm_reports.prompt_template import SYSTEM_PROMPT, build_report_prompt, build_summary_prompt


class ReportGenerator:
    """
    Generates structured financial risk reports via OpenAI's chat completions API.
    All quantitative inputs are pre-computed externally; the LLM provides interpretation only.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.model = config["llm"]["model"]
        self.max_tokens = config["llm"]["max_tokens"]
        self.temperature = config["llm"]["temperature"]
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before running the report generator."
            )
        self.client = OpenAI(api_key=api_key)

    def generate_asset_report(self, payload: dict) -> dict:
        """
        Generate a structured risk report for a single asset.

        Args:
            payload: Dict containing all model outputs for the asset.
                     See risk_engine/risk_calculator.py:latest_risk_payload().

        Returns:
            Dict with keys: asset, report_text, generated_at, model, payload.
        """
        user_prompt = build_report_prompt(payload)
        logger.info(f"Generating LLM report for {payload.get('asset', 'Unknown')} using {self.model}")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        report_text = response.choices[0].message.content

        return {
            "asset": payload.get("asset", "Unknown"),
            "report_text": report_text,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "payload": payload,
        }

    def generate_cross_asset_summary(self, payloads: list[dict]) -> str:
        """
        Generate a cross-asset risk summary.

        Args:
            payloads: List of per-asset payload dicts.

        Returns:
            Summary text string.
        """
        user_prompt = build_summary_prompt(payloads)
        logger.info(f"Generating cross-asset summary for {len(payloads)} assets")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=600,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def build_payload(
        self,
        asset: str,
        risk_payload: dict,
        current_regime: str,
        regime_probability: float,
        forecasted_vol: float,
        sentiment_index: float,
        close_price: float,
        ytd_return: float,
    ) -> dict:
        """
        Assemble the full structured JSON payload for the LLM.

        Args:
            asset:               Ticker symbol.
            risk_payload:        Output of RiskScoreCalculator.latest_risk_payload().
            current_regime:      String regime label (e.g. 'bear', 'bull').
            regime_probability:  Probability of the current regime [0, 1].
            forecasted_vol:      7-day forward volatility forecast (annualized).
            sentiment_index:     Current daily sentiment index [-1, 1].
            close_price:         Latest closing price.
            ytd_return:          Year-to-date log return.

        Returns:
            Complete payload dict ready for prompt injection.
        """
        return {
            "asset": asset,
            "date": risk_payload.get("date"),
            "close_price": round(close_price, 2),
            "ytd_return_pct": round(ytd_return * 100, 2),
            "current_regime": current_regime,
            "regime_probability": round(regime_probability, 4),
            "forecasted_7d_vol_annualized": round(forecasted_vol, 4),
            "sentiment_index": round(sentiment_index, 4),
            "risk_score": risk_payload.get("risk_score"),
            "risk_band": risk_payload.get("risk_band"),
            "risk_components": risk_payload.get("components"),
            "risk_weights": risk_payload.get("weights"),
        }
