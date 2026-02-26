
"""
dashboard/components/report_interface.py

Report generation interface: generate LLM reports and download as PDF.
"""

from pathlib import Path

import streamlit as st


def render(asset: str, cfg: dict) -> None:
    st.subheader("AI-Generated Risk Report")
    st.caption(
        "Reports are generated using the OpenAI API. The LLM interprets "
        "pre-computed model outputs only — it does not perform calculations."
    )

    reports_dir = Path(cfg["llm"]["reports_dir"])
    safe_asset = asset.replace("^", "").replace("-", "_")
    asset_reports_dir = reports_dir / safe_asset

    # List existing reports
    existing_jsons = sorted(asset_reports_dir.glob("report_*.json"), reverse=True) if asset_reports_dir.exists() else []
    existing_pdfs = sorted(asset_reports_dir.glob("report_*.pdf"), reverse=True) if asset_reports_dir.exists() else []

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"**Asset:** {asset}")
        if existing_jsons:
            st.markdown(f"**Reports available:** {len(existing_jsons)}")
            latest_ts = existing_jsons[0].stem.replace("report_", "")
            st.markdown(f"**Latest report:** {latest_ts[:8]} {latest_ts[9:15]} UTC")
        else:
            st.markdown("No reports generated yet.")

    with col2:
        if existing_pdfs:
            latest_pdf = existing_pdfs[0]
            with open(latest_pdf, "rb") as f:
                st.download_button(
                    label="Download Latest PDF",
                    data=f.read(),
                    file_name=latest_pdf.name,
                    mime="application/pdf",
                )

    st.markdown("---")

    # Display latest report text
    if existing_jsons:
        import json
        latest_report = json.loads(existing_jsons[0].read_text(encoding="utf-8"))
        report_text = latest_report.get("report_text", "")

        if report_text:
            st.markdown(report_text)

        with st.expander("View Input Payload"):
            st.json(latest_report.get("payload", {}))

        with st.expander("Report Metadata"):
            st.json({
                "model": latest_report.get("model"),
                "generated_at": latest_report.get("generated_at"),
                "input_tokens": latest_report.get("input_tokens"),
                "output_tokens": latest_report.get("output_tokens"),
            })
    else:
        st.info(
            "No reports found for this asset. Run the report generator script to create one:\n"
            "```bash\npython -m scripts.generate_report --asset SPY\n```"
        )

    # Report history
    if len(existing_jsons) > 1:
        st.markdown("---")
        st.subheader("Report History")
        history = []
        for jf in existing_jsons[:10]:
            import json
            data = json.loads(jf.read_text(encoding="utf-8"))
            payload = data.get("payload", {})
            history.append({
                "Generated At": data.get("generated_at", "")[:19].replace("T", " "),
                "Risk Score": payload.get("risk_score", ""),
                "Risk Band": payload.get("risk_band", ""),
                "Regime": payload.get("current_regime", ""),
            })

        import pandas as pd
        st.dataframe(pd.DataFrame(history), use_container_width=True)
