# PHVS Clinical Expectations Lab

An original research component of a short thesis on Pharvaris N.V. (NASDAQ:
PHVS). The lab leads with the short recommendation, works through three research
questions - methodology, finding, investment implication, valuation impact -
and reproduces the evidence in three slide-ready exhibits that chain clinical
adoption, adoption scenarios and downside valuation.

> Research for a short thesis, not personalised investment advice. Scenario
> outputs are uncalibrated model results, not forecasts, consensus estimates or
> measured prescribing probabilities.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

## Test

```bash
.venv/bin/python -m pytest phvs_lab/tests -q
```

## Published page

```bash
.venv/bin/python build_site.py     # writes docs/index.html (GitHub Pages)
```

`build_site.py` calls the same module functions the Streamlit app does, so the
published numbers cannot drift from the app. The three exhibits are exported as
PNGs into `docs/exhibits/` for the deck:

```bash
.venv/bin/python - <<'EOF'
from phvs_lab.modules import research as rs
from phvs_lab.utils import visualization as viz
for name, fig in rs.exhibit_figures().items():
    print(viz.export_chart(fig, name))
EOF
```

## Layout

```
app.py                      Streamlit UI: eight thesis sections + ledger, charts, methodology
build_site.py               static site generator for docs/index.html
phvs_lab/config.py          theme, static copy, formatters
phvs_lab/data/*.csv         auditable evidence layer
  clinical_trials.csv         observations with source, window, numerator/denominator
  commercial_assumptions.csv  parameter ledger: value, type, status, source, note
  sources.csv                 source register with URL and publication date
  pitch_claim_audit.csv       pitch claims vs the sources behind them
  valuation_model.csv         the pitch cash flows, reproduced before extension
phvs_lab/modules/           pure-Python analysis engines (no Streamlit)
  data_loader.py             typed loaders, cache, validators, provenance helpers
  evidence.py                registry verification and audit reporting
  response_engine.py         threshold inversion, identified sets, exact intervals
  trial_audit.py             comparability flags, window standardisation
  economics.py               population funnel, segment shares, prevention/rescue bridge
  valuation.py               pitch reproduction, DCF, scenarios, market-implied solves
  prescribing_map.py         segments, logit choice, contour, break-even
  survey.py                  DCE design, response contract, conditional-logit estimator
  research.py                research component: recommendation, RQ1-RQ3, chain, exhibits
phvs_lab/utils/visualization.py  Plotly builders + kaleido export
docs/index.html             published page
docs/exhibits/              the three slide-ready exhibit PNGs
METHODOLOGY.md               how every number is produced and labelled
```

## The three research questions

| | Question | What the lab does when the data cannot answer |
|---|---|---|
| RQ1 | What does the clinical response distribution reveal that headline efficacy obscures? | States exact intervals on disclosed counts and flags what stays provisional. |
| RQ2 | How could competing options and the prevention / rescue relationship constrain adoption? | Gives the switchers the price requires and the break-even switching rate, never a measured rate. |
| RQ3 | What adoption assumptions justify the price, and how does our short case differ? | Solves each input against the observed close one at a time and states the invalidation level. |

## Rules the code enforces

- No number appears in the UI unless it comes from a module or the CSV ledger.
- Every value is tagged `reported` / `calculated` / `assumed` and carries a
  verification status; research workstreams are tagged `completed` / `proposed`
  / `assumed`.
- Nested response thresholds are inverted only when fully disclosed; otherwise
  an identified set is returned, never a point estimate.
- Separate trials are never shown as a randomised comparison.
- Segments must sum to 1.0 and cohorts must fit inside the population, or the
  model raises.
- Approval risk is applied once per product stream, never twice.
- Infeasible inputs raise instead of producing a plausible-looking number.
- Where the data cannot settle a question, the output is a break-even threshold
  or a sensitivity - never an invented conclusion.
