# Synthetic dataset — Hospital Operations & Revenue Risk Intelligence

**This is a synthetically generated dataset, not real hospital data and not a
cleaned version of the original capstone data.** It was built specifically
for a portfolio project, with intentional, documented relationships between
features and targets so that trained models reach ~90%+ accuracy honestly —
i.e. the model is learning real (engineered) signal, not memorizing leaked
information or an artifact of data cleaning.

Put this note in your README/portfolio writeup. Transparency here is what
makes the high accuracy legitimate to show off rather than misleading.

## What's real vs. engineered

- **Schema and realistic value ranges** (ages, departments, insurance
  providers, billing amounts, etc.) mirror a real hospital operations
  dataset.
- **All relationships between features and the two prediction targets
  (`risk_score`, `claim_status`) are deliberately injected**, with random
  noise layered in so the classes are not perfectly separable (a "too
  perfect" 99–100% accuracy is usually a red flag; capping around 90–92%
  with realistic error patterns is the more credible signal).
- **No target leakage.** Every model below was trained and evaluated on
  features that do NOT include a value that's mathematically derived from
  the label itself (see the leakage note below — this was actually caught
  and fixed mid-build).

## Files

| File | Rows | Notes |
|---|---|---|
| `patients_synthetic.csv` | 5,000 | same schema as original |
| `visits_synthetic.csv` | ~24,960 | same schema + 1 new column, `vitals_abnormal_flag` |
| `billing_synthetic.csv` | ~24,960 | same schema as original |

## Injected relationships

### `risk_score` (Low / Medium / High) — visits.csv
Driven by a weighted combination of:
- `chronic_flag` (chronic patients skew higher risk)
- `age > 60`
- `visit_type == 'ICU'`
- `department` in `{ICU, Cardiology}`
- `vitals_abnormal_flag` — **new column**, simulating a nurse triage flag
  captured at admission (a realistic clinical signal a real EHR would
  actually record)
- Gaussian noise (σ=0.24) + 3% random label noise

`length_of_stay_hours` is generated from the **same underlying causal
factors** as `risk_score` (chronic_flag, age, visit_type, department,
vitals_abnormal_flag) plus its own independent noise — it is correlated with
risk because they share real causes, but it is *not* derived from the
`risk_score` label itself. (An earlier draft of this generator did derive LOS
directly from the risk label — that's target leakage in disguise, the same
mistake as using `approved_amount` to predict `claim_status`. It was caught
and fixed; see "Leakage note" below.)

**Validated result:** Random Forest, features = `age, chronic_flag,
visit_type, department, length_of_stay_hours, vitals_abnormal_flag` →
**~91% accuracy**, F1 0.87–0.94 across all three classes (no single class is
trivially dominant).

### `claim_status` (Paid / Pending / Rejected) — billing.csv
Generated as two sequential threshold decisions (mirroring how a real prior-
auth / claims-adjudication rule engine might work):
1. **Rejected** vs. not — driven by high `billed_amount` and insurance
   provider (`CareOne`, `MediCareX` reject more often — a "stricter payer"
   effect), thresholded at the 85th percentile (~15% Rejected, matching
   real-world claim rejection rates).
2. **Pending** vs. **Paid** (within non-rejected) — driven by
   high-administrative-overhead departments (`ICU`, `ER`) and non-chronic
   patients, thresholded at the 70th percentile of the remainder.
3% random label noise on top of both steps.

`approved_amount` and `payment_days` are then derived **from** `claim_status`
(Paid → full amount + a payment turnaround time; Rejected → 0 + no
payment_days; Pending → partial amount + no payment_days yet). **These two
columns are consequences of the claim outcome, not predictors of it — do not
use them as model features for `claim_status`, or you'll get leakage-inflated
results.**

**Validated result:** Random Forest, features = `billed_amount,
insurance_provider, department, chronic_flag` (explicitly excluding
`approved_amount` / `payment_days`) → **~92% accuracy**.

## Leakage note (worth including in your writeup — it's a good story)

An earlier version of this generator derived `length_of_stay_hours` directly
from the `risk_score` label (e.g. "High risk → ~34 hrs average"). That's
structurally identical to the leakage problem already documented for
`approved_amount` → `claim_status`: the "predictor" is just a noisy encoding
of the answer. It was caught during model validation (accuracy looked
plausible but the feature-importance breakdown showed LOS carrying ~72% of
the signal, which is a leakage tell) and reworked so LOS shares the same
underlying causes as risk_score instead of being generated from the label.
Explicitly calling this out in an interview or writeup is a good way to
demonstrate you understand leakage, not just that you can avoid it.

## Data-quality guarantees (by construction)

- `registration_date <= visit_date <= billing_date` for every row
- `billing_date` always on/after the estimated discharge date
  (`visit_date + ceil(length_of_stay_hours / 24)` days)
- No duplicate (`patient_id`, `visit_date`, `department`) visits
- `payment_days` is only populated for `Paid` claims (null for Pending/Rejected)
- No orphaned visits/billing records, no duplicate patient IDs

## How to regenerate

Run `generate_synthetic_data.py` (seeded, `np.random.default_rng(42)`, fully
reproducible). Tune the noise levels documented inline (currently σ=0.24 for
`risk_score`, σ=0.25 for the claim-status thresholds) if you want a different
accuracy ceiling — lower noise = higher achievable accuracy, at the cost of
looking less realistic.