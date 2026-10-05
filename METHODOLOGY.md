# Methodology

PHVS Clinical Expectations Lab reverse-engineers the assumptions required to
justify the observed Pharvaris share price (NASDAQ: PHVS, close **$31.19** on
2 October 2026). It is a measurement instrument, not a forecast.

## 1. Evidence rules

Every observation carries drug, arm, population, sample size, endpoint, window,
numerator/denominator, background therapy, exclusions, source URL, page,
publication date and one of three arithmetic tags:

| tag | meaning |
| --- | --- |
| `reported` | copied from the source without transformation |
| `calculated` | derived by arithmetic from reported values (formula stated) |
| `assumed` | an analyst input with no source measurement |

Every row also carries a `verification_status`:

| status | meaning |
| --- | --- |
| `verified_primary` | matched against the primary registry record |
| `verified_secondary` | matched against a primary filing or peer-reviewed source |
| `corroborated_secondary` | matches a credible secondary source only |
| `provisional_unverified` | company-reported, not yet posted to the registry |
| `unverified` | analyst assumption, no source measurement exists |

`phvs_lab.modules.evidence.verify_against_registry` re-queries
ClinicalTrials.gov live and compares the stored counts with the registry's own
measurements, per arm. Failures degrade to `not_found_in_registry`; they never
invent a value.

## 2. Threshold inversion

Responder endpoints are published as nested cumulative thresholds
(`>=50%`, `>=70%`, `>=90%`, attack-free). Exclusive categories are recovered by
subtraction:

```
attack-free   = c4
90-99%        = c3 - c4
70-89%        = c2 - c3
50-69%        = c1 - c2
<50%          = n - c1
```

This is only valid when all four thresholds were disclosed. With partial
disclosure the module returns an **identified set** solved by linear program
(minimum and maximum of each category subject to the disclosed equalities),
never a point estimate. Sampling uncertainty is reported separately as exact
binomial (Clopper-Pearson) intervals on the disclosed cumulative proportions.

## 3. Window mechanics

Attack-free is a count over a stated window and falls mechanically as the window
lengthens at an unchanged rate. Two responses are kept apart throughout:

- **Sampling uncertainty** → confidence interval.
- **Window length** → design choice, reported on its own line, standardised
  only through an explicit gamma-Poisson model that is labelled as a
  transformation, not a re-analysis.

Cross-trial comparisons are never presented as randomised comparisons. The audit
section lists every structural difference (population, window, endpoint,
comparator) before the numbers appear.

## 4. Population accounting

```
diagnosed → eligible for prophylaxis → payer covered → segments
```

Segments are mutually exclusive, validated to sum to 1.0, and each stage is
validated to be a subset of the previous one. Prophylaxis revenue is earned by
the whole cohort; acute revenue is earned only by the breakthrough subset inside
it, so better prevention mechanically reduces rescue revenue. Attacks, patients
and revenue are three separate quantities with separate conversion assumptions.

## 5. Dispersion calibration

Patient heterogeneity in attack rates is a gamma-Poisson (negative binomial)
mix with shape `k`. `k` is fitted **once** from the trial's own published
values (trial window, reported rate reduction, observed attack-free share) by
root-finding, then held fixed while scenario efficacy varies. This guarantees
that changing efficacy moves the answer for the expected reason rather than by
re-fitting the calibration.

## 6. Valuation

1. **Reproduce the pitch.** The imported pitch cash flows are recomputed on
   their own terms, then re-run with the verified 70.2m share count and the
   verified $362.9m net cash. Findings (terminal row omitted from the cumulative
   column, stale 38m share count, embedded price, implied 10% discount rate) are
   listed, not silently corrected.
2. **Extend.** Mid-year discounting, ramp to peak over the stated time to peak,
   gross-to-net margins, R&D and SG&A at face value pre-launch, NOL-tracker tax,
   capex and working capital, loss of exclusivity with a post-LOE R&D floor,
   terminal value with growth forced to zero after the LOE year.
3. **Risk adjust once.** Approval probability is applied once per product
   contribution stream. Post-launch operating costs are weighted by the
   probability that at least one indication is approved. Pre-launch spend and
   base SG&A are never discounted for approval risk - the conservative choice.
4. **Solve backwards.** For each input separately, `market_implied` finds the
   value at which the model equals the observed price. `required_bundle` reports
   the enterprise value the price requires as a multiple of the model.

Terminal growth, discount rate and forecast end are validated against the LOE
year so the terminal value cannot quietly capitalise post-exclusivity profits.

## 7. Prescribing map

Segments feed a multinomial logit over *stay on current prophylaxis*, *switch to
existing oral* and *switch to PHVS oral*. Utilities combine ledger intercepts,
a segment oral-preference increment, a switch cost and a price term indexed to
the assumed reference LTP price. Switchers accumulate at the stated annual
switching rate over the time to peak and are reduced by the stated
discontinuation rate. The result is divided by the eligible population to give
peak penetration, which is passed to the valuation - so the contour reads in
value per share, the same unit as the market price.

## 8. Survey instrument

A discrete-choice design exists because the coefficients above are currently
assumptions. The survey **has not been conducted**; the module provides the
design, the CSV contract for responses, a conditional-logit estimator with
standard errors, and a synthetic demo mode that is labelled as synthetic in every
output. Stated preference would still not measure revealed prescribing.

## 9. What this cannot do

- It does not forecast. Scenario outputs are uncalibrated model results.
- It does not compare separate trials as if randomised against each other.
- It does not generate patient-level data, physician interviews or consensus
  estimates.
- It does not know the real net price, the real switching rate, or durability
  beyond 24 weeks - those are the inputs section 8 lists as unmeasured.
