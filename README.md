# PHVS Clinical Expectations Lab

A research-grade Streamlit application that reverse-engineers the clinical and
prescribing assumptions required to justify Pharvaris N.V.'s observed share
price, and shows where a short thesis works, fails, and what evidence is still
missing.

> Not investment advice. Scenario outputs are uncalibrated model results, not
> forecasts, consensus estimates or measured prescribing probabilities.

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

## Layout

```
app.py                      Streamlit UI: eight sections + ledger, charts, methodology
phvs_lab/config.py          Theme, static copy, formatters
phvs_lab/data/*.csv         Auditable evidence layer
  clinical_trials.csv         observations with source, window, numerator/denominator
  commercial_assumptions.csv  parameter ledger: value, type, status, source, note
  sources.csv                 source register with URL and publication date
  pitch_claim_audit.csv       pitch claims vs the sources behind them
  valuation_model.csv         the pitch cash flows, reproduced before extension
phvs_lab/modules/           Pure-Python analysis engines (no Streamlit)
  data_loader.py             typed loaders, cache, validators, provenance helpers
  evidence.py                registry verification and audit reporting
  response_engine.py         threshold inversion, identified sets, exact intervals
  trial_audit.py             comparability flags, window standardisation
  economics.py               population funnel, segment shares, prevention/rescue bridge
  valuation.py               pitch reproduction, DCF, scenarios, market-implied solves
  prescribing_map.py         segments, logit choice, contour, break-even
  survey.py                  DCE design, response contract, conditional-logit estimator
phvs_lab/utils/visualization.py  Plotly builders + kaleido export
exports/                     PNG charts written on demand
METHODOLOGY.md               how every number is produced and labelled
```

## Rules the code enforces

- No number appears in the UI unless it comes from a module or the CSV ledger.
- Every value is tagged `reported` / `calculated` / `assumed` and carries a
  verification status.
- Nested response thresholds are inverted only when fully disclosed; otherwise
  an identified set is returned, never a point estimate.
- Separate trials are never shown as a randomised comparison.
- Segments must sum to 1.0 and cohorts must fit inside the population, or the
  model raises.
- Approval risk is applied once per product stream, never twice.
- Infeasible inputs raise instead of producing a plausible-looking number.

## Charts

Rendered on demand from the app's "Exportable charts" panel, or:

```bash
.venv/bin/python - <<'EOF'
from phvs_lab.modules import data_loader as dl, valuation as val, response_engine as re_
from phvs_lab.utils import visualization as viz
base = val.params_from_assumptions()
print(viz.export_chart(viz.fig_required_inputs(val.market_implied(base),
                                               val.sensitivity_ranking(base)),
                       "phvs_required_inputs"))
EOF
```
