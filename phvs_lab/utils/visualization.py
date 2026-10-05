"""Plotly figure builders for the PHVS Clinical Expectations Lab.

Every figure is a pure function of module output, so the app, the export
pipeline and the tests draw from the same source. No figure invents a number:
labels that carry model output are marked with the same wording the modules
return.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from phvs_lab.config import (
    AMBER, EXPORT_DIR, GREEN, GREY_300, GREY_600, INK, PLOT_LAYOUT, PHVS_BLUE,
    PHVS_BLUE_DARK, RED, SERIES,
)

__all__ = [
    "apply_layout", "export_chart", "fig_pitch_vs_models", "fig_equity_bridge",
    "fig_pv_profile", "fig_tornado", "fig_required_inputs", "fig_response_categories",
    "fig_response_thresholds", "fig_window_sensitivity", "fig_window_adjustment",
    "fig_efficacy_projection", "fig_population_funnel", "fig_prevention_ladder",
    "fig_prescribing_contour", "fig_choice_shares", "fig_short_scenarios",
    "fig_evidence_coverage", "format_assumption_ledger", "EXPORTED_CHARTS",
]

STATUS_COLOR = {"verified_primary": GREEN, "verified_secondary": GREEN,
                "corroborated_secondary": PHVS_BLUE, "provisional_unverified": AMBER,
                "unverified": RED}


def apply_layout(fig: go.Figure, title: str = "", subtitle: str = "",
                 y_title: str = "", x_title: str = "", height: int = 460
                 ) -> go.Figure:
    """Apply the lab's brand layout to a figure."""
    top = int(height * 0.17) if title else int(height * 0.10)
    layout = dict(PLOT_LAYOUT)
    layout.update(height=height, margin=dict(PLOT_LAYOUT["margin"], t=top))
    fig.update_layout(**layout)
    margin = PLOT_LAYOUT["margin"]
    bottom_frac = margin["b"] / height
    span = 1.0 - bottom_frac - top / height
    if title:
        fig.add_annotation(text=f"<b>{title}</b>", x=0, xref="paper", xanchor="left",
                           y=(0.975 - bottom_frac) / span, yref="paper", yanchor="top",
                           showarrow=False, font=dict(size=17, color=PHVS_BLUE_DARK))
    if subtitle:
        fig.add_annotation(text=subtitle, x=0, xref="paper", xanchor="left",
                           y=(0.885 - bottom_frac) / span, yref="paper", yanchor="top",
                           showarrow=False, font=dict(size=12, color=GREY_600))
    if y_title:
        fig.update_yaxes(title_text=y_title)
    if x_title:
        fig.update_xaxes(title_text=x_title)
    return fig


def export_chart(fig: go.Figure, name: str, width: Optional[int] = None,
                 scale: int = 2) -> Path:
    """Write a PNG through kaleido into the repository exports folder.

    The exported size follows the figure's own layout so that annotations
    anchored in paper coordinates land where the layout put them.
    """
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIR / f"{name}.png"
    width = width or int(fig.layout.width or 1200)
    height = int(fig.layout.height or 460)
    fig.write_image(str(path), width=width, height=height, scale=scale)
    return path


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------

def fig_pitch_vs_models(repro: Dict, scenario: pd.DataFrame, price: float) -> go.Figure:
    """Pitch model against the extended model, the scenarios and the price."""
    labels = ["Pitch model\n(38m shares, as filed)",
              "Pitch model\n(70.2m shares, corrected)",
              "Extended model\n(bear)", "Extended model\n(reference)",
              "Extended model\n(bull)"]
    values = [repro["value_per_share_pitch_shares"],
              repro["value_per_share_current_shares"],
              *scenario.set_index("scenario").loc[["bear", "reference", "bull"],
                                                  "value_per_share"].tolist()]
    colors = [GREY_600, GREY_600, SERIES[3], PHVS_BLUE, SERIES[1]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values, marker_color=colors, text=[f"${v:,.2f}" for v in values],
        textposition="outside", name="Model value per share"))
    fig.add_hline(y=price, line_color=AMBER, line_width=2, line_dash="dash",
                  annotation_text=f"Market price ${price:,.2f}",
                  annotation_font_color=AMBER, annotation_position="bottom right",
                  annotation_xanchor="right")
    fig.update_layout(bargap=0.35)
    return apply_layout(
        fig, "Two models, one price",
        "The pitch's own cash flows reproduced, then extended with verified shares, "
        "cash, timing and approval risk. Scenario outputs are uncalibrated model results.",
        y_title="USD per share")


def fig_equity_bridge(summary: Dict) -> go.Figure:
    """Present value of operations -> enterprise value -> equity value."""
    steps = [
        ("PV of operating\nforecast", summary["pv_operating"], "absolute"),
        ("PV of terminal\nvalue", summary["pv_terminal"], "absolute"),
        ("Enterprise\nvalue", summary["enterprise_value"], "total"),
        ("Net cash", summary["net_cash"], "relative"),
        ("Equity value", summary["equity_value"], "total"),
    ]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=[s[2] for s in steps],
        x=[s[0] for s in steps], y=[s[1] for s in steps],
        text=[f"${s[1]/1e6:,.0f}m" for s in steps], textposition="outside",
        connector_line=dict(color=GREY_300),
        increasing_marker=dict(color=PHVS_BLUE),
        decreasing_marker=dict(color=RED),
        totals_marker=dict(color=PHVS_BLUE_DARK)))
    fig.update_layout(showlegend=False, yaxis=dict(tickformat="$,.0f"))
    return apply_layout(
        fig, "Equity bridge at the reference inputs",
        f"Terminal value is {summary['terminal_share_of_ev']:.1%} of enterprise value. "
        "Approval risk is applied once per product, before the bridge.",
        y_title="USD", height=480)


def fig_pv_profile(cashflows: pd.DataFrame) -> go.Figure:
    """Discounted cash flow contribution by year, split by product stream."""
    df = cashflows.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["year"], y=df["pv_prophylaxis"], name="Prophylaxis",
                         marker_color=PHVS_BLUE))
    fig.add_trace(go.Bar(x=df["year"], y=df["pv_acute"], name="Acute",
                         marker_color=SERIES[1]))
    fig.add_trace(go.Bar(x=df["year"], y=df["pv_fixed"], name="Fixed (R&D, SG&A, tax)",
                         marker_color=GREY_600))
    fig.add_trace(go.Scatter(x=df["year"], y=df["pv_fcf"].cumsum(), name="Cumulative PV",
                             mode="lines", line=dict(color=PHVS_BLUE_DARK, width=2)))
    fig.update_layout(barmode="relative", bargap=0.25)
    return apply_layout(
        fig, "Where the present value comes from",
        "Fixed costs are carried at face value; product contributions are weighted by "
        "the probability that each indication is approved.",
        y_title="Present value, USD", x_title="Year", height=500)


def fig_tornado(sens: pd.DataFrame, reference: float) -> go.Figure:
    """Value per share swing from a one-at-a-time shock, ranked."""
    df = sens.sort_values("swing")
    labels = [f"{p}  ({v:,.4g})" for p, v in zip(df["parameter"], df["stated_value"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=df["value_at_minus"], orientation="h", name="Input -10%",
        marker_color=SERIES[1]))
    fig.add_trace(go.Bar(
        y=labels, x=df["value_at_plus"], orientation="h", name="Input +10%",
        marker_color=PHVS_BLUE))
    fig.add_vline(x=reference, line_color=AMBER, line_dash="dash", line_width=2,
                  annotation_text=f"Stated ${reference:,.2f}",
                  annotation_font_color=AMBER)
    return apply_layout(
        fig, "Which inputs actually move the answer",
        "One input shocked at a time, all others held at their stated values. "
        "Swing is the distance between the two bars.",
        y_title="", x_title="USD per share", height=520)


def fig_required_inputs(mi: pd.DataFrame, sens: Optional[pd.DataFrame] = None
                        ) -> go.Figure:
    """Stated input versus the value the market price requires, in gap terms.

    The four inputs are not commensurable (two proportions, a rate and a price),
    so the bar shows how far the stated input sits from the required one, signed
    so that a positive bar always means the stated inputs already cover the price.
    """
    df = mi.copy()
    rises = {}
    if sens is not None and "direction" in sens.columns:
        rises = dict(zip(sens["parameter"], sens["direction"]))
    signed = []
    for _, r in df.iterrows():
        direction = rises.get(r["input"], "value rises with input")
        multiplier = 1.0 if direction.startswith("value rises") else -1.0
        signed.append(multiplier * float(r["stated_vs_required_pct"]))
    df["signed_gap"] = signed
    df["label"] = (df["input"].str.replace("_", " ", regex=False)
                   + "<br><span style='font-size:11px'>"
                   + [f"{s:,.4g} stated → {q:,.4g} required"
                      for s, q in zip(df["stated_value"], df["required_value"])]
                   + "</span>")
    df = df.sort_values("signed_gap")
    fig = go.Figure(go.Bar(
        x=df["signed_gap"], y=df["label"], orientation="h",
        marker_color=[GREEN if v >= 0 else RED for v in df["signed_gap"]],
        text=[f"{v:+.1%}" for v in df["signed_gap"]], textposition="outside"))
    fig.add_vline(x=0, line_color=INK, line_width=1.4)
    return apply_layout(
        fig, "What the market price requires",
        "Each input is solved separately: moved until the model equals the observed price, "
        "all else equal. A bar right of zero means the stated input is already better than "
        "what the price requires; left of zero means it would have to improve. "
        "Not a forecast and not a consensus estimate.",
        x_title="Stated input relative to the input the price requires",
        height=520)


# ---------------------------------------------------------------------------
# Clinical evidence
# ---------------------------------------------------------------------------

def fig_response_categories(dist) -> go.Figure:
    """Exclusive response categories, with the identified set when partial."""
    counts = dist.counts if dist.is_point_identified else None
    if counts is None:
        bounds = dist.identified_set()
        x = bounds["category"]
        y = bounds["max_share"] - bounds["min_share"]
        base = bounds["min_share"]
        fig = go.Figure(go.Bar(x=x, y=y, base=base, marker_color=PHVS_BLUE,
                               name="Identified set",
                               text=[f"{lo:.1%} – {hi:.1%}"
                                     for lo, hi in zip(bounds["min_share"], bounds["max_share"])],
                               textposition="outside"))
        subtitle = ("Only some thresholds were disclosed, so the exclusive categories are "
                    "bounded, not known. Bar spans every value consistent with the data.")
    else:
        props = dist.proportions
        x = list(props.keys())
        y = [props[k] for k in x]
        fig = go.Figure(go.Bar(x=x, y=y, marker_color=PHVS_BLUE,
                               text=[f"{v:.1%}" for v in y], textposition="outside"))
        subtitle = (f"{dist.trial} / {dist.arm}, n={dist.total_patients}. Exclusive categories "
                    "reconstructed by subtraction from the nested thresholds.")
    fig.update_xaxes(tickangle=-20)
    return apply_layout(fig, "Exclusive response categories", subtitle,
                        y_title="Share of patients", height=460)


def fig_response_thresholds(intervals: pd.DataFrame) -> go.Figure:
    """Cumulative thresholds with exact binomial intervals."""
    df = intervals.sort_values("level")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["threshold"], y=df["proportion"], mode="markers+lines",
        marker=dict(size=11, color=PHVS_BLUE),
        line=dict(color=SERIES[1], width=1.5), name="Reported share",
        error_y=dict(type="data", array=(df["ci_high"] - df["proportion"]).tolist(),
                     arrayminus=(df["proportion"] - df["ci_low"]).tolist(),
                     color=GREY_600, thickness=1.6, width=5)))
    fig.update_xaxes(tickangle=-20)
    return apply_layout(
        fig, "Cumulative thresholds with exact binomial intervals",
        "Clopper-Pearson intervals describe sampling only. The observation window is a "
        "design choice and is reported separately, never folded into the interval.",
        y_title="Share of patients", height=460)


def fig_window_sensitivity(curves: pd.DataFrame) -> go.Figure:
    """Attack-free probability against window length for several heterogeneity levels."""
    fig = go.Figure()
    for i, (label, grp) in enumerate(curves.groupby("rate_model", sort=False)):
        grp = grp.sort_values("window_days")
        fig.add_trace(go.Scatter(
            x=grp["window_days"], y=grp["attack_free_probability"], mode="lines+markers",
            name=label, line=dict(color=SERIES[i % len(SERIES)], width=2.2),
            marker=dict(size=7)))
    fig.update_layout(legend_title_text="Rate model")
    return apply_layout(
        fig, "The same patients, different windows",
        "Attack-free is a count over a stated window: mechanically lower over longer "
        "windows with no change in the underlying rate. Illustrative model output.",
        y_title="Probability of zero attacks", x_title="Observation window, days",
        height=480)


def fig_window_adjustment(table: pd.DataFrame) -> go.Figure:
    """Observed attack-free rate beside the same rate standardised to one window."""
    df = table.dropna(subset=["Observed attack-free"]).copy()
    df = df.sort_values("Observed attack-free")
    labels = [f"{r['Trial']} / {r['Arm']}" for _, r in df.iterrows()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Observed attack-free"], y=labels, mode="markers", name="As published",
        marker=dict(size=12, color=GREY_600, symbol="circle")))
    fig.add_trace(go.Scatter(
        x=df["Standardised to 180d"], y=labels, mode="markers",
        name="Standardised to 180 days", marker=dict(size=12, color=PHVS_BLUE,
                                                     symbol="diamond")))
    for _, r in df.iterrows():
        if pd.notna(r["Standardised to 180d"]):
            lab = labels[df.index.get_loc(r.name)]
            fig.add_trace(go.Scatter(
                x=[r["Observed attack-free"], r["Standardised to 180d"]], y=[lab, lab],
                mode="lines", showlegend=False,
                line=dict(color=GREY_300, width=2)))
    fig.update_layout(showlegend=True)
    return apply_layout(
        fig, "Attack-free rate depends on the window",
        "Separate trials, different windows and populations - shown together only to expose "
        "the window effect. Not a randomised comparison and not adjusted clinical evidence.",
        x_title="Share of patients with zero attacks", height=480)


def fig_efficacy_projection(projection: pd.DataFrame) -> go.Figure:
    """Acute demand and revenue as prevention efficacy varies."""
    df = projection.sort_values("efficacy")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["efficacy"], y=df["acute_revenue"], name="Acute revenue",
                         marker_color=SERIES[1]))
    fig.add_trace(go.Scatter(x=df["efficacy"], y=df["breakthrough_patients"],
                             name="Breakthrough patients", mode="lines+markers",
                             yaxis="y2", line=dict(color=PHVS_BLUE_DARK, width=2.4)))
    fig.add_trace(go.Scatter(x=df["efficacy"], y=df["prophylaxis_revenue"],
                             name="Prophylaxis revenue", mode="lines",
                             yaxis="y2", line=dict(color=PHVS_BLUE, width=2,
                                                   dash="dot")))
    fig.update_layout(
        barmode="overlay", bargap=0.15,
        yaxis2=dict(title="Patients / prophylaxis revenue", overlaying="y",
                    side="right", showgrid=False))
    return apply_layout(
        fig, "Better prevention lowers rescue revenue",
        "Dispersion is calibrated once from the trial's published values and then held "
        "fixed, so the direction of this curve is a modelling choice, not a refit.",
        y_title="Acute revenue, USD", x_title="Scenario efficacy (before real-world haircut)",
        height=480)


# ---------------------------------------------------------------------------
# Population and prescribing
# ---------------------------------------------------------------------------

def fig_population_funnel(ledger: pd.DataFrame) -> go.Figure:
    """Patient funnel from diagnosed population to mutually exclusive segments."""
    df = ledger.copy()
    stages = df["stage"].tolist()
    values = df["patients"].tolist()
    colors = [PHVS_BLUE if not s.startswith("Segment") else SERIES[1] for s in stages]
    fig = go.Figure(go.Bar(
        x=values, y=stages, orientation="h", marker_color=colors,
        text=[f"{v:,.0f}" for v in values], textposition="outside"))
    fig.update_yaxes(categoryorder="array", categoryarray=stages)
    return apply_layout(
        fig, "Population accounting",
        "Each stage is a subset of the one above; segments are mutually exclusive and "
        "validated to sum to 1.0 of the prophylaxis-eligible pool.",
        x_title="Patients", height=520)


def fig_prevention_ladder(ladder: pd.DataFrame) -> go.Figure:
    """Patients, attacks and revenue kept as three separate quantities."""
    units = [u for u in ladder["unit"].unique() if u in ("patients", "attacks", "USD")]
    fig = make_subplots(rows=1, cols=len(units), subplot_titles=[f"in {u}" for u in units],
                        horizontal_spacing=0.12)
    palette = [PHVS_BLUE, SERIES[1], GREY_600]
    for i, unit in enumerate(units, start=1):
        sub = ladder[ladder["unit"] == unit]
        fig.add_trace(go.Bar(x=sub["stage"], y=sub["value"], name=unit, showlegend=False,
                             marker_color=palette[(i - 1) % len(palette)],
                             text=[f"{v:,.0f}" for v in sub["value"]],
                             textposition="outside"), row=1, col=i)
        fig.update_xaxes(tickangle=-35, row=1, col=i)
    fig.update_layout(height=480)
    return apply_layout(
        fig, "Prevention and rescue, counted once",
        "The cohort appears once. Acute demand comes from the breakthrough subset inside "
        "it, so better prevention mechanically reduces rescue revenue.",
        y_title="", height=520)


def fig_prescribing_contour(surface: pd.DataFrame, price: float,
                            stated_switching: float, stated_premium: float = 1.0
                            ) -> go.Figure:
    """Value per share over oral premium and switching rate, with the price contour."""
    pivot = surface.pivot(index="switching_rate", columns="premium_multiplier",
                          values="value_per_share").sort_index().sort_index(axis=1)
    premiums = pivot.columns.to_numpy(float)
    switchings = pivot.index.to_numpy(float)
    z = pivot.to_numpy(float)
    fig = go.Figure(go.Heatmap(
        x=premiums, y=switchings, z=z,
        colorscale=[[0, "#F2F6FF"], [0.5, "#98B7F5"], [1, PHVS_BLUE]],
        colorbar=dict(title="USD/share", tickformat="$,.0f")))
    xs, ys = _price_contour(premiums, switchings, z, price)
    if xs:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                 line=dict(color=AMBER, width=3),
                                 name=f"Value = ${price:,.2f}"))
    fig.add_trace(go.Scatter(x=[stated_premium], y=[stated_switching], mode="markers+text",
                             marker=dict(size=14, color=RED, symbol="x",
                                         line=dict(width=3, color=RED)),
                             text=["stated inputs"], textposition="top right",
                             textfont=dict(color=RED), name="Stated inputs"))
    return apply_layout(
        fig, "What the price requires from prescribing",
        "Contour of model value per share over the oral price premium and the annual "
        "switching rate. The amber line is where the model equals the observed price.",
        y_title="Annual switching rate", x_title="PHVS price relative to stated price",
        height=560)


def _price_contour(x: np.ndarray, y: np.ndarray, z: np.ndarray, level: float
                   ) -> tuple[List[float], List[float]]:
    """Marching-squares style crossing of a single level on a coarse grid."""
    xs: List[float] = []
    ys: List[float] = []
    for i in range(z.shape[0] - 1):
        for j in range(z.shape[1] - 1):
            quad = [z[i, j], z[i, j + 1], z[i + 1, j + 1], z[i + 1, j]]
            pts = []
            corners = [(x[j], y[i]), (x[j + 1], y[i]), (x[j + 1], y[i + 1]), (x[j], y[i + 1])]
            for k in range(4):
                a, b = quad[k], quad[(k + 1) % 4]
                if (a - level) == 0 or (a - level) * (b - level) < 0:
                    t = 0.5 if a == b else (level - a) / (b - a)
                    x0, y0 = corners[k]
                    x1, y1 = corners[(k + 1) % 4]
                    pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
            if len(pts) >= 2:
                xs.append(pts[0][0]); ys.append(pts[0][1])
    return xs, ys


def fig_choice_shares(choice: pd.DataFrame) -> go.Figure:
    """Multinomial-logit choice shares for each segment at the stated price."""
    df = choice.sort_values("share_of_population", ascending=True)
    labels = [f"{s} ({p:.0%} of pool)" for s, p in zip(df["segment"], df["share_of_population"])]
    fig = go.Figure()
    for col, name, color in (("p_current", "Stay on current prophylaxis", GREY_600),
                             ("p_existing_oral", "Switch to existing oral", SERIES[1]),
                             ("p_phvs_oral", "Switch to PHVS oral", PHVS_BLUE)):
        fig.add_trace(go.Bar(y=labels, x=df[col], orientation="h", name=name,
                             marker_color=color, text=[f"{v:.0%}" for v in df[col]],
                             textposition="inside"))
    fig.update_layout(barmode="stack")
    return apply_layout(
        fig, "Choice shares by segment",
        "Multinomial logit over stated utilities. Coefficients are analyst assumptions in "
        "the ledger - scenario output, not observed prescribing.",
        x_title="Probability of choice", height=500)


def fig_short_scenarios(short: pd.DataFrame) -> go.Figure:
    """Net short return by scenario against the borrow cost."""
    df = short.copy()
    colors = [GREEN if v > 0 else RED for v in df["net_return_pct"]]
    fig = go.Figure(go.Bar(x=df["scenario"], y=df["net_return_pct"], marker_color=colors,
                           text=[f"{v:,.1f}%" for v in df["net_return_pct"]],
                           textposition="outside"))
    fig.add_hline(y=0, line_color=INK, line_width=1.2)
    fig.add_hline(y=float(df["carry_pct"].iloc[0]), line_color=AMBER, line_dash="dash",
                  annotation_text=f"Borrow + carry {df['carry_pct'].iloc[0]:.0%}",
                  annotation_font_color=AMBER)
    return apply_layout(
        fig, "When the short works, and when it fails",
        "Scenario outputs of the model, not recommendations. A negative bar means the "
        "model value exceeds the observed price at those inputs.",
        y_title="Net return over the stated horizon, %", height=460)


def fig_evidence_coverage(provenance: pd.DataFrame) -> go.Figure:
    """Verification status of every reported observation."""
    counts = provenance["verification_status"].value_counts()
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[STATUS_COLOR.get(i, GREY_600) for i in counts.index],
        text=counts.values, textposition="outside"))
    fig.update_layout(showlegend=False)
    return apply_layout(
        fig, "How well each observation is sourced",
        "Status comes from the evidence ledger: primary registry or filing, secondary "
        "corroboration, provisional pending publication, or unverified.",
        y_title="Observations", height=420)


def format_assumption_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """Presentation view of the assumption ledger, with provenance kept visible."""
    out = df.copy()
    out["parameter"] = out["parameter"].str.replace("_", " ", regex=False)
    out["value"] = [f"{v:,.6g}" for v in out["value"]]
    out["evidence"] = [
        f"[{s}]({u})" if isinstance(u, str) and u else (s if isinstance(s, str) else "—")
        for s, u in zip(out.get("source_id", pd.Series(dtype=str)),
                        out.get("source_url", pd.Series(dtype=str)))
    ]
    names = {"parameter": "Parameter", "value": "Value", "unit": "Unit",
             "category": "Category", "assumption_type": "Type",
             "verification_status": "Status", "evidence": "Source", "notes": "Basis"}
    keep = [c for c in names if c in out.columns]
    return out[keep].rename(columns=names).reset_index(drop=True)


EXPORTED_CHARTS = {
    "phvs_required_inputs": "The inputs the market price requires",
    "phvs_response_categories": "Reconstructed response categories",
    "phvs_value_scenarios": "Model value against the observed price",
}
