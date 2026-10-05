"""Theme, paths and static copy for the PHVS Clinical Expectations Lab.

Design brief: Pharvaris blue on white, restrained typography, no decorative
colour outside the brand palette.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
REPO_ROOT = PACKAGE_ROOT.parent
EXPORT_DIR = REPO_ROOT / "exports"

# --- Brand palette -------------------------------------------------------
# Pharvaris blue is used for headings, primary series and controls. White is
# the page background; greys are used only for structure and secondary text.
PHVS_BLUE = "#0B4FD8"          # primary brand blue
PHVS_BLUE_DARK = "#082C6E"     # headings
PHVS_BLUE_LIGHT = "#DCE7FF"    # fills, highlight bands
PHVS_BLUE_PALE = "#F2F6FF"     # table stripes, panel backgrounds
WHITE = "#FFFFFF"
INK = "#10151F"                # body text
GREY_600 = "#5A6478"           # secondary text
GREY_300 = "#D5DAE3"           # rules / borders
GREY_100 = "#F6F7F9"           # panel background
AMBER = "#B57A00"              # "unverified" flag
RED = "#B3261E"                # "contradicted" flag
GREEN = "#0F7B4F"              # "verified" flag

SERIES = [PHVS_BLUE, "#5B8DEF", "#98B7F5", GREY_600, "#8E97AA", PHVS_BLUE_DARK]

PLOT_LAYOUT = dict(
    font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif", size=13, color=INK),
    paper_bgcolor=WHITE,
    plot_bgcolor=WHITE,
    margin=dict(l=60, r=30, t=64, b=56),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis=dict(gridcolor=GREY_300, zeroline=False, showline=True, linecolor=GREY_300),
    yaxis=dict(gridcolor=GREY_300, zeroline=False, showline=True, linecolor=GREY_300),
)

# --- Static copy ---------------------------------------------------------

DISCLAIMERS = {
    "not_forecast": (
        "Scenario outputs in this lab are **uncalibrated model results**, not forecasts, "
        "consensus estimates or measured prescribing probabilities."
    ),
    "not_evidence": (
        "Simulation panels are **illustrative assumptions**, not adjusted clinical evidence. "
        "They are not used to declare comparative superiority or to infer a validated "
        "long-term outcome."
    ),
    "no_h2h": (
        "Separate trials are never presented as a randomised head-to-head comparison. "
        "Where two trials are shown together the panel is flagged and the differences in "
        "population, window and endpoint are listed."
    ),
    "no_fabrication": (
        "No patient-level data, physician interviews or consensus estimates are generated. "
        "Reconstructed categories describe aggregate counts only."
    ),
}

CONCLUSION_TEMPLATE = (
    "At the stated inputs, the market price requires {X}. "
    "Our available evidence supports {Y}, with {Z} still unmeasured."
)

SECTION_INTROS = {
    "evidence": "Every observation carries a drug, arm, population, sample size, endpoint, "
                "window, numerator/denominator, background therapy, exclusions, source URL, "
                "page, publication date and an explicit reported / calculated / assumed flag.",
    "response": "Nested response thresholds are inverted into mutually exclusive categories. "
                "The reconstruction is arithmetic, not estimation.",
    "audit": "Results are shown beside population, window and endpoint so that no two trials "
             "can be read as a randomised comparison by accident.",
    "map": "Segments, utilities and switching assumptions are editable. The contour shows what "
           "must be true for the price, not what investors believe.",
    "economics": "Cohorts are mutually exclusive, so no patient is counted twice. Attacks are "
                 "not equated with paid prescriptions.",
    "valuation": "The pitch cash-flow model is reproduced first, then extended with verified "
                 "share count, cash, timing and approval risk.",
    "survey": "Stated preferences from a discrete-choice survey are not actual prescribing "
              "behaviour; the panel states what the survey can and cannot inform.",
}

# --- Units / formatting helpers -----------------------------------------

def usd(x: float, decimals: int = 0) -> str:
    return f"${x:,.{decimals}f}"


def usd_m(x: float) -> str:
    return f"${x/1e6:,.1f}m"


def pct(x: float, decimals: int = 1) -> str:
    return f"{100*x:.{decimals}f}%"
