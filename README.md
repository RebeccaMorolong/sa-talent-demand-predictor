# SA Talent Demand Predictor — ML Engineering Project

# 🎯 Project Goal

Build an end-to-end ML system that **predicts talent/skill demand across South African industries**, and investigates how **degree requirements vs. skill-based hiring** contributes to the unemployment crisis.

> **Core Thesis:** South Africa's unemployment rate is partly driven by degree gatekeeping in job postings — not a shortage of skilled people. This project quantifies that gap and builds a fairer talent-matching system.
> 

---

# 📌 Problem Framing

Two interconnected problems to solve:

1. **Predictive Problem** — Which skills will be in demand in which industries over the next 6–24 months?
2. **Analytical / Causal Problem** — Do degree requirements in job postings correlate with longer vacancy periods, hiring discrimination, or unemployment rates in specific demographics?

---

# 🗄️ Data Sources

## South African Government & Official Data

| Source | What You Get | Link |
| --- | --- | --- |
| **Stats SA (QLFS)** | Quarterly Labour Force Survey — unemployment by province, sector, education level | [stats.gov.za](http://stats.gov.za) |
| **DHET** | Graduate output by field, TVET college data | [dhet.gov.za](http://dhet.gov.za) |
| **SETA Reports** | Sector Skills Plans with demand forecasts (MERSETA, FASSET, INSETA, etc.) | Each SETA has its own portal |
| **LMIP** | Labour Market Intelligence — occupation & sector data | [lmip.org.za](http://lmip.org.za) |
| **World Bank Open Data** | SA employment, education, and economic indicators | [data.worldbank.org](http://data.worldbank.org) |

## Job Postings (Scraping / APIs)

| Source | Method |
| --- | --- |
| **LinkedIn** | API (with approval) or scrape — extract skills vs. degree requirements |
| **Pnet** | Scrape — skills, degree requirements, location, salary |
| **CareerJunction** | Scrape — industry, seniority, job descriptions |
| **Indeed ZA** | Scrape — salary bands, time posted |

## Skill Taxonomy Reference

- **ONET Online** ([onetonline.org](http://onetonline.org)) — rich skill-to-occupation mappings, adaptable to SA context
- **Kaggle** — search "South Africa unemployment" or "job postings skills" for starter datasets
- **ISCO** — international standard occupation codes, mappable to SA

---

# 🔧 Data Pipeline Architecture

```
Raw Sources (Stats SA, Job Postings, SETA PDFs)
        ↓
Data Ingestion (Python: requests, scrapy, pdfplumber)
        ↓
Data Lake (store raw — CSV, JSON, PDF extracts)
        ↓
ETL / Feature Engineering (pandas, dbt)
        ↓
Feature Store
  - skills frequency
  - degree mention rate
  - industry demand trends
  - unemployment rate by sector
        ↓
ML Models
        ↓
Dashboard / API
```

---

# ⚙️ Feature Engineering

## From Job Postings

- Skill mentions (Python, welding, project management, etc.)
- Degree requirements (yes/no, field, level — matric / diploma / degree / postgrad)
- Industry / sector
- Location (province)
- Seniority level
- Salary range (if available)
- Time to fill (track posting duration)

## From Stats SA QLFS

- Unemployment rate by education level, province, age, gender, race
- Employment by sector quarter-over-quarter

## Key Derived Features

- **Degree requirement rate per industry** — % of postings requiring a degree
- **Skill gap index** — skills demanded vs. skills in workforce
- **Degree premium vs. skills premium** by sector

---

# 🤖 ML Models

## Model 1 — Skill Demand Forecasting (Time Series)

**Goal:** Predict which skills will grow in demand by industry over the next 12 months.

- **Algorithm:** Prophet, LSTM, or XGBoost with time features
- **Target:** Skill mention frequency per quarter
- **Features:** Past demand trends, economic indicators, sector growth rates

## Model 2 — Degree vs. Skills Classifier

**Goal:** Given a job posting, predict whether it's degree-gated unnecessarily.

- **Algorithm:** Random Forest / XGBoost / BERT fine-tuned on job descriptions
- **Target:** Degree required (yes/no) — compare to actual job complexity
- **Insight:** Exposes bias in posting behavior

## Model 3 — Unemployment Risk by Skills Gap

**Goal:** Predict unemployment probability for a profile given their skills vs. local demand.

- **Algorithm:** Logistic Regression / Gradient Boosting
- **Target:** Employed / Unemployed
- **Features:** Skill set, education level, province, age, sector

## Model 4 (Bonus) — NLP Skill Extractor

**Goal:** Automatically extract skills from job postings and CVs, mapping to a standard taxonomy.

- **Tools:** spaCy or fine-tuned BERT

---

# ⚖️ Bias Analysis Layer

This is what makes the project stand out. After modeling, run:

- **Fairness audit** — Does the model disadvantage people with skills but no degree?
- **Counterfactual analysis** — If degree requirements are removed from postings, how does the predicted talent pool change by race, gender, province?
- **Disparity analysis** — Compare hiring outcomes across demographic groups

**Tools:** IBM AI Fairness 360, Microsoft Fairlearn, or manual disparity analysis with pandas

---

# 🛠️ Tech Stack

| Layer | Tools |
| --- | --- |
| Data Ingestion | Python, Scrapy, pdfplumber, requests |
| Storage | PostgreSQL or DuckDB (lightweight) |
| Processing | pandas, polars, dbt |
| ML | scikit-learn, XGBoost, HuggingFace, Prophet |
| NLP | spaCy, sentence-transformers |
| Experiment Tracking | MLflow or Weights & Biases |
| API | FastAPI |
| Dashboard | Streamlit or Metabase |
| Orchestration | Airflow or Prefect |
| Version Control | Git + DVC for data versioning |

---

# 📦 Project Deliverables

- [ ]  **Skill Demand Dashboard** — Interactive view of top growing skills per industry and province
- [ ]  **Degree vs. Skills Gap Report** — Quantified analysis of how degree gatekeeping correlates with unemployment in SA
- [ ]  **Talent Matching API** — Input a person's skills → output best-fit industries and roles, no degree filter
- [ ]  **Policy Insight Report** — Recommendations for employers and government (DHET, SETA, NYDA)

---
---

# 🎯 Potential Impact

This project is publishable and could attract attention from:

- **NYDA** (National Youth Development Agency)
- **SETA bodies** looking for evidence-based skills planning
- **SA HR tech startups** building fairer hiring platforms
- **Academic journals** focused on labour economics and AI fairness



[Step 7 — API & Dashboard](https://www.notion.so/Step-7-API-Dashboard-3250579a39a48184a7f5cce7c52a2746?pvs=21)

[Project Roadmap](https://www.notion.so/Project-Roadmap-3250579a39a481819d84c5b1f8f284d3?pvs=21)
