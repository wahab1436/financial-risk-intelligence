"""
dashboard/components/regime_timeline.py

Regime timeline overlay and transition matrix view.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REGIME_COLORS = {
    "bull":     "#26a69a",
    "sideways": "#f59e0b",
    "high_vol": "#f97316",
    "bear":     "#ef5350",
    "regime_0": "#6366f1",
    "regime_1": "#f59e0b",
    "regime_2": "#ef5350",
    "regime_3": "#26a69a",
}


def render(price_df: pd.DataFrame, regime_labels: pd.Series, asset: str) -> None:
    if price_df.empty:
        st.warning(f"No price data for {asset}.")
        return

    if regime_labels.empty:
        st.info("Regime labels not yet computed. Run the regime detection model first.")
        return

    # Align to common index
    common = price_df.index.intersection(regime_labels.index)
    price = price_df.loc[common, "Close"]
    regimes = regime_labels.loc[common]

    # --- Metrics ---
    col1, col2, col3 = st.columns(3)
    current_regime = str(regimes.iloc[-1]) if len(regimes) > 0 else "Unknown"
    col1.metric("Current Regime", current_regime.title())

    n_transitions = (regimes != regimes.shift(1)).sum()
    n_years = (regimes.index[-1] - regimes.index[0]).days / 365.25
    col2.metric("Transitions/Year", f"{n_transitions / max(n_years, 0.01):.1f}")

    # Days in current regime
    changes = (regimes != regimes.shift(1)).cumsum()
    current_run = int((changes == changes.iloc[-1]).sum())
    col3.metric("Days in Current Regime", str(current_run))

    st.markdown("---")

    # --- Price chart with regime shading ---
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=price.index, y=price.values,
            mode="lines", name=asset,
            line=dict(color="#90caf9", width=1.5),
        )
    )

    # Add colored background bands for each regime
    prev_regime = None
    start_idx = price.index[0]

    for i, (date, regime) in enumerate(regimes.items()):
        if regime != prev_regime:
            if prev_regime is not None:
                color = REGIME_COLORS.get(str(prev_regime), "rgba(100,100,100,0.2)")
                fig.add_vrect(
                    x0=str(start_idx.date()),
                    x1=str(date.date()),
                    fillcolor=color,
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                )
            start_idx = date
            prev_regime = regime

    # Final segment
    if prev_regime is not None:
        color = REGIME_COLORS.get(str(prev_regime), "rgba(100,100,100,0.2)")
        fig.add_vrect(
            x0=str(start_idx.date()),
            x1=str(price.index[-1].date()),
            fillcolor=color,
            opacity=0.15,
            layer="below",
            line_width=0,
        )

    fig.update_layout(
        title=f"{asset} — Price with Regime Overlay",
        height=420,
        template="plotly_dark",
        showlegend=False,
        margin=dict(t=40, b=20, l=20, r=20),
        yaxis_title="Price (USD)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Transition Matrix ---
    st.subheader("Regime Transition Matrix")
    unique_regimes = sorted(regimes.unique().tolist(), key=str)
    transition_counts = pd.DataFrame(0, index=unique_regimes, columns=unique_regimes)

    for i in range(1, len(regimes)):
        from_r = regimes.iloc[i - 1]
        to_r = regimes.iloc[i]
        transition_counts.loc[from_r, to_r] += 1

    row_sums = transition_counts.sum(axis=1).replace(0, 1)
    transition_pct = transition_counts.div(row_sums, axis=0).round(4)
    st.dataframe(transition_pct.style.format("{:.2%}").background_gradient(cmap="RdYlGn_r"))

    # --- Regime Distribution ---
    st.subheader("Regime Distribution")
    dist = regimes.value_counts(normalize=True).reset_index()
    dist.columns = ["Regime", "Frequency"]
    dist["Frequency"] = (dist["Frequency"] * 100).round(1)
    st.bar_chart(dist.set_index("Regime")["Frequency"])
