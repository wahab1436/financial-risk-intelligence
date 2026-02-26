"""
llm_reports/report_storage.py

Persists and retrieves generated LLM reports with timestamped metadata.
Reports are stored as JSON (full payload + text) and optionally as PDF.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF
from loguru import logger


class ReportStorage:
    """
    Manages persistence of generated LLM reports.
    Reports are stored in a structured directory: reports/{asset}/{date}/
    """

    def __init__(self, config: dict):
        self.reports_dir = Path(config["llm"]["reports_dir"])
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def save(self, report: dict) -> dict[str, Path]:
        """
        Save a report as both JSON and PDF.

        Args:
            report: Dict from ReportGenerator.generate_asset_report().

        Returns:
            Dict with 'json' and 'pdf' keys mapping to saved file paths.
        """
        asset = report.get("asset", "UNKNOWN").replace("^", "").replace("-", "_")
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        asset_dir = self.reports_dir / asset
        asset_dir.mkdir(exist_ok=True)

        # Save JSON
        json_path = asset_dir / f"report_{ts}.json"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(f"Report JSON saved: {json_path}")

        # Save PDF
        pdf_path = asset_dir / f"report_{ts}.pdf"
        self._write_pdf(report, pdf_path)
        logger.info(f"Report PDF saved: {pdf_path}")

        return {"json": json_path, "pdf": pdf_path}

    def _write_pdf(self, report: dict, path: Path) -> None:
        """Render report text and metadata to a formatted PDF."""
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Header
        pdf.set_font("Helvetica", "B", size=16)
        pdf.cell(0, 10, "Financial Risk Intelligence Report", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(2)

        # Metadata table
        pdf.set_font("Helvetica", size=10)
        meta_items = [
            ("Asset", report.get("asset", "")),
            ("Generated", report.get("generated_at", "")[:19].replace("T", " ") + " UTC"),
            ("Model", report.get("model", "")),
            ("Risk Score", str(report.get("payload", {}).get("risk_score", ""))),
            ("Risk Band", str(report.get("payload", {}).get("risk_band", ""))),
        ]
        for label, value in meta_items:
            pdf.set_font("Helvetica", "B", size=10)
            pdf.cell(40, 8, f"{label}:", border=0)
            pdf.set_font("Helvetica", size=10)
            pdf.cell(0, 8, str(value), border=0, new_x="LMARGIN", new_y="NEXT")

        pdf.ln(5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # Report body
        pdf.set_font("Helvetica", size=11)
        report_text = report.get("report_text", "")
        for line in report_text.split("\n"):
            if line.startswith("## "):
                pdf.ln(3)
                pdf.set_font("Helvetica", "B", size=12)
                pdf.multi_cell(0, 8, line[3:], new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", size=11)
            elif line.strip():
                pdf.multi_cell(0, 7, line, new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.ln(3)

        pdf.output(str(path))

    def load_latest(self, asset: str) -> dict | None:
        """Load the most recently saved report for an asset."""
        safe = asset.replace("^", "").replace("-", "_")
        asset_dir = self.reports_dir / safe
        if not asset_dir.exists():
            return None
        json_files = sorted(asset_dir.glob("report_*.json"))
        if not json_files:
            return None
        return json.loads(json_files[-1].read_text(encoding="utf-8"))

    def list_reports(self, asset: str) -> list[dict]:
        """List all stored reports for an asset (metadata only)."""
        safe = asset.replace("^", "").replace("-", "_")
        asset_dir = self.reports_dir / safe
        if not asset_dir.exists():
            return []
        reports = []
        for json_file in sorted(asset_dir.glob("report_*.json"), reverse=True):
            data = json.loads(json_file.read_text(encoding="utf-8"))
            reports.append({
                "file": str(json_file),
                "asset": data.get("asset"),
                "generated_at": data.get("generated_at"),
                "risk_score": data.get("payload", {}).get("risk_score"),
                "risk_band": data.get("payload", {}).get("risk_band"),
            })
        return reports

    def get_pdf_path(self, asset: str, latest: bool = True) -> Path | None:
        """Return path to the latest (or specified) PDF report."""
        safe = asset.replace("^", "").replace("-", "_")
        asset_dir = self.reports_dir / safe
        if not asset_dir.exists():
            return None
        pdf_files = sorted(asset_dir.glob("report_*.pdf"))
        if not pdf_files:
            return None
        return pdf_files[-1] if latest else pdf_files
