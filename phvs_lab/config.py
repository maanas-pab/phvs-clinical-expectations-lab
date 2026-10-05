"""Theme, paths and static copy for the PHVS Clinical Expectations Lab.

Design brief: Pharvaris blue on white, restrained typography, no decorative
colour outside the brand palette.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
REPO_ROOT = PACKAGE_ROOT.parent
EXPORT_DIR = REPO_ROOT / "docs" / "exhibits"

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
# Written for a short thesis on Pharvaris (PHVS). The lab is an original
# research component of that thesis: it leads with the recommendation, then
# works through three research questions, then shows the exhibits that carry
# the argument in a deck.

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
    "not_prescribing": (
        "Choice-model panels translate **stated, unverified preferences** into implied "
        "patients. They are scenario arithmetic, not observed prescribing behaviour, and no "
        "physician has answered the survey that would replace them."
    ),
    "position_disclosure": (
        "This is research written to support a short thesis on PHVS, not personalised "
        "investment advice. Every input, status and source used here is disclosed in the "
        "ledger below; the position is invalidated at the break-even level stated in the "
        "recommendation."
    ),
}

CONCLUSION_TEMPLATE = (
    "The recommendation is <b>{REC}</b> at a {PRICE} close. On the price's own terms that "
    "requires <b>{X}</b>. The completed work supports <b>{Y}</b>, while <b>{Z}</b> remain "
    "unmeasured - and the position is invalidated at model value above <b>{W}</b>."
)

SECTION_INTROS = {
    "recommendation": (
        "This lab is an original research component of a short thesis on Pharvaris (PHVS). "
        "It leads with the recommendation and the completed findings that carry it; every "
        "number below is reproduced from the ledger and can be re-derived from source."
    ),
    "research": (
        "Three research questions run through the thesis. Each is stated the same way - "
        "methodology, finding, investment implication, valuation impact - and each carries a "
        "status of completed, proposed or assumed so that what has been measured is never "
        "confused with what has merely been assumed."
    ),
    "exhibit1": (
        "Slide-ready exhibit 1: what the published response data actually says once nested "
        "thresholds are inverted into mutually exclusive categories, with sampling "
        "uncertainty kept separate from window length."
    ),
    "exhibit2": (
        "Slide-ready exhibit 2: the adoption arithmetic. Efficacy constrains patients, not "
        "prescriptions - the distance between the two is where the price lives."
    ),
    "exhibit3": (
        "Slide-ready exhibit 3, the centrepiece: clinical evidence chained to adoption "
        "scenarios and then to downside valuation, with every connecting assumption named "
        "and the break-even level shown against the price."
    ),
    "valuation": (
        "The pitch cash-flow model is reproduced first, then extended with verified share "
        "count, cash, timing and approval risk. The gap between the two is the whole "
        "argument, so both are shown at the same inputs."
    ),
    "limitations": (
        "The strongest case against this short, stated at full strength, followed by what is "
        "genuinely missing. Where the data cannot settle a question we give the break-even "
        "threshold instead of a conclusion."
    ),
    "evidence": (
        "Every observation carries a drug, arm, population, sample size, endpoint, window, "
        "numerator/denominator, background therapy, exclusions, source URL, page, "
        "publication date and an explicit reported / calculated / assumed flag."
    ),
}

# --- Units / formatting helpers -----------------------------------------

def usd(x: float, decimals: int = 0) -> str:
    return f"${x:,.{decimals}f}"


def usd_m(x: float) -> str:
    return f"${x/1e6:,.1f}m"


def pct(x: float, decimals: int = 1) -> str:
    return f"{100*x:.{decimals}f}%"
