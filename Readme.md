# Hospital Operations & Revenue Risk Intelligence Platform

**Advanced Certificate Programme in Applied AI & Machine Learning — Healthcare Business Capstone**

An end-to-end analytics and machine learning platform that turns fragmented hospital data into predictive intelligence — helping hospital leadership manage patient risk and helping finance teams cut revenue leakage from rejected or delayed insurance claims.

---

## Project Summary

Multi-specialty hospital networks lose money and efficiency to two recurring problems:

1. **Operational risk** — clinical teams can't always tell in advance which visits will need the most attention.
2. **Revenue leakage** — a large share of billed revenue never gets collected, due to claim rejections and payment delays.

This project builds a simulated hospital analytics stack — from raw data to a relational database, to exploratory analysis, to two deployed-ready ML classifiers — that gives hospital leadership a data-driven view of both problems.

| | |
|---|---|
| **Patients** | 5,000 |
| **Visits** | 25,000 |
| **Billing records** | 25,000 |
| **Cities** | Hyderabad, Pune, Chennai, Bangalore, Mumbai, Delhi |
| **Departments** | Cardiology, Orthopedics, ICU, General, ER, Neurology |
| **Insurance Providers** | SecureLife, HealthPlus, CareOne, MediCareX |

> **Data note:** The dataset is synthetically generated (`numpy` RNG, seed=42) with documented, realistic relationships between clinical acuity, length of stay, and claim outcomes — built specifically so the modelling pipeline could be demonstrated end-to-end without relying on sensitive real patient data.

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Database** | SQLite (`sqlite3`), relational schema with foreign keys, indexes |
| **Data wrangling** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn` (Logistic Regression, Random Forest), `XGBoost` |
| **Model tuning** | `RandomizedSearchCV` |
| **Explainability** | `SHAP`, built-in feature importances |
| **Serialization** | `pickle`, `json`, `parquet` |
| **Environment** | Jupyter Notebook |

---

## Project Structure

```
├── Phase1.ipynb    → SQL Analytics Layer
├── Phase2.ipynb    → Exploratory Data Analysis & Data Quality
├── Phase3.ipynb    → Model Development (Model A + Model B)
├── Phase4.ipynb    → Model Evaluation & Explainability
├── hospital.db     → SQLite database (patients, visits, billing)
├── merged_clean.parquet → Cleaned, joined modelling dataset
├── models/         → Saved model pickles, feature schemas, metrics
├── eda_plots/       → EDA charts
├── model_plots/      → Model training/CM plots
└── phase4_plots/      → Evaluation, fairness & SHAP plots
```

---

## Phase 1 — SQL Analytics Layer

**Goal:** turn raw CSVs into a trustworthy, query-ready relational database.

- Designed and created a **SQLite schema** for `patients`, `visits`, and `billing`, with primary/foreign keys, `CHECK` constraints (e.g. valid age range, non-negative amounts), and cascading updates/deletes.
- Added **11 indexes** across the three tables (on city, insurance provider, department, dates, risk score, claim status, etc.) to keep joins and filters fast.
- Generated the **synthetic dataset** with intentional, documented signal — e.g. chronic-condition probability rising with age, risk score driven by acuity factors with a controlled 8% label-noise flip, and length-of-stay generated conditional on risk score — so downstream models would have genuine, learnable patterns.
- Loaded all three CSVs into the database and validated integrity with joined sample queries.
- Wrote reusable **business SQL queries**, split into two groups:
  - **Operational Analysis:** visit volume by department, average length of stay, % high-risk visits by department, avg visits per patient by city, doctors handling the most high-risk visits.
  - **Financial Analysis:** top insurance providers by billed amount, claim rejection rate by provider, average payment delay by provider, revenue realization ratio by department, and flagged cases of high billed / zero-approved amounts (a direct revenue-leakage signal).

## Phase 2 — Exploratory Data Analysis & Data Quality

**Goal:** understand the data deeply before modelling, and confirm it's decision-ready.

- Merged patients → visits → billing into one analysis-ready DataFrame.
- **Distribution analysis** across all numeric and categorical fields (age, length of stay, billed/approved amounts, payment days, department, risk score, claim status, etc.).
- **Outlier detection** using the IQR method — reviewed and consciously **retained** all outliers, since extreme length-of-stay and billed amounts reflect real ICU/high-cost cases rather than data errors.
- **Business insight generation:**
  - Department performance (visits, avg LOS, % high risk, avg billed, avg leakage)
  - Insurance provider behaviour (approval rate, avg payment delay)
  - Overall **revenue realization** — total billed vs. approved vs. leakage
  - Chronic vs. non-chronic patient profiles
- **Correlation & bivariate analysis** — correlation heatmap across engineered numeric features, plus risk-score and claim-status relationships against length of stay, billed amount, payment days, department, and insurance provider (via boxplots and cross-tab heatmaps).
- Closed with a **modelling readiness summary**: final feature sets and leakage-safe target definitions for both upcoming models.

## Phase 3 — Model Development (Classification Systems)

Two classification models, each trained as **baseline → advanced → tuned**, with a strict **time-based train/test split** (80/20 by date, not random) to simulate realistic deployment and avoid future-data leakage.

### Model A — Visit Risk Classification
Predicts whether a visit is **Low / Medium / High** risk *before* clinical outcomes are known.
- **Baseline:** Logistic Regression
- **Advanced/main model:** XGBoost (`multi:softprob`, 500 estimators)
- **Tuning:** `RandomizedSearchCV` (30 iterations, 3-fold CV, `f1_weighted`)
- Carefully excluded all **post-visit fields** (billed amount, claim status, payment days, etc.) from the feature set to prevent leakage — only pre-visit and clinical/demographic features are used.
- Feature importance analysis on the tuned model.

### Model B — Claim Outcome Classification
Predicts whether a claim will be **Paid / Pending / Rejected**.
- **Baseline:** Logistic Regression (`class_weight="balanced"`)
- **Advanced:** Random Forest (`class_weight="balanced_subsample"`)
- **Tuning:** `RandomizedSearchCV` on estimators, depth, split/leaf sizes, and max features, scored on `balanced_accuracy`
- **Imbalance handling:** monitored F1 for the minority "Rejected" class and built in an automatic recommendation to apply **SMOTE** if F1 fell below 0.50.
- Feature importance analysis on the tuned Random Forest.

All models and their feature schemas are saved to `models/` (`.pkl` + `.json`) for reuse in Phase 4 and future deployment.

## Phase 4 — Model Evaluation & Explainability

**Goal:** confirm both models are not just accurate, but reliable, interpretable, and fair enough for healthcare use.

- **Business-relevant metrics**, not just accuracy — recall, precision, and F1 specifically for the **High Risk** class (Model A) and the **Rejected** class (Model B), since missing these is the costliest failure mode in each case.
- **Train vs. test confusion matrices** side by side, with accuracy and balanced accuracy, to explicitly check the train/test performance gap for overfitting.
- **Feature importance** plots for both models.
- **Fairness segmentation** — recall for the high-stakes class recomputed across **gender, insurance provider, and city**, to check the models don't perform unevenly across demographic groups.
- **SHAP explainability** — mean absolute SHAP values to show which features drive each model's predictions, with a graceful fallback to built-in feature importances if SHAP isn't available.
- All final metrics consolidated into `phase4_metrics.json` for reporting.

---

## Roadmap (Capstone Phases Ahead)

| Phase | Status |
|---|---|
| 1. SQL Analytics Layer | Complete |
| 2. EDA & Data Quality | Complete |
| 3. Model Development | Complete |
| 4. Model Evaluation & Explainability | Complete |
| 5. Monitoring, Drift Detection & Governance | Planned |

---

## Key Takeaways So Far

- Built a **leakage-safe** modelling pipeline validated with time-based splitting.
- Quantified **revenue leakage** and claim rejection drivers at the department and insurance-provider level.
- Delivered two production-candidate classifiers with documented feature schemas, tuning results, and fairness checks — ready for the deployment phase.
