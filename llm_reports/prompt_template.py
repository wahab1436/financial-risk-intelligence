"""
llm_reports/prompt_template.py

Builds structured prompts for OpenAI API from model outputs.
LLM is used strictly for interpretation — all numbers are pre-computed.
"""

import json
from datetime import date


SYSTEM_PROMPT = """You are a senior quantitative risk analyst generating a structured financial risk intelligence report.

You will receive a structured JSON payload containing pre-computed metrics from machine learning models. Your role is to interpret these numbers and provide clear, professional analysis. You must:

1. Never recalculate or second-guess the provided numbers.
2. Write in a concise, analytical, and professional tone.
3. Structure your response using the exact section headers provided.
4. Quantify statements where possible using the provided data.
5. Flag specific risk mitigation actions relevant to the regime and risk level.

Do not include disclaimers about being an AI or about the limitations of the analysis."""


def build_report_prompt(payload: dict) -> str:
    """
    Build the user prompt for the LLM report generator.

    Args:
        payload: Structured dict with all model outputs.

    Returns:
        Formatted user prompt string.
    """
    return f"""Generate a financial risk intelligence report based on the following model outputs.

MARKET DATA PAYLOAD:
{json.dumps(payload, indent=2)}

Generate a report with EXACTLY these sections in order:

## Executive Summary
One paragraph (3-4 sentences) covering the overall risk assessment, current regime, and primary risk drivers.

## Regime Analysis
Analyze the current market regime and what it implies for positioning. Reference the regime probability and historical context.

## Volatility Outlook
Interpret the forecasted 7-day volatility relative to historical norms. Assess whether volatility is rising or falling and its implications.

## Sentiment Interpretation
Explain the current sentiment index reading and what it signals about market participants' risk appetite.

## Risk Score Breakdown
Explain the composite risk score and which components are driving it. Reference the individual component weights.

## Risk Mitigation Considerations
Provide 3-5 specific, actionable risk management considerations appropriate for the current risk level and regime. These should be concrete hedging or positioning recommendations, not general advice.

---
Report Date: {date.today().isoformat()}
Asset: {payload.get('asset', 'Unknown')}
"""


def build_summary_prompt(reports: list[dict]) -> str:
    """
    Build a cross-asset summary prompt from multiple asset reports.

    Args:
        reports: List of payload dicts from multiple assets.

    Returns:
        Formatted prompt for cross-asset summary.
    """
    assets_summary = [
        {
            "asset": r.get("asset"),
            "risk_score": r.get("risk_score"),
            "risk_band": r.get("risk_band"),
            "regime": r.get("current_regime"),
        }
        for r in reports
    ]

    return f"""Generate a brief cross-asset risk summary based on the following data:

{json.dumps(assets_summary, indent=2)}

Provide a 2-paragraph summary identifying:
1. Which assets are showing the highest systemic risk and why.
2. Whether risks are correlated across assets (systemic) or idiosyncratic.

Keep the response under 200 words. Date: {date.today().isoformat()}"""
