# SA Talent Demand Predictor

An end-to-end ML project that predicts skill demand across South African industries
and investigates how degree-based hiring barriers contribute to unemployment.

**Core question:** Are South African employers filtering out skilled candidates
because of degree requirements that don't actually reflect job complexity?

---

## What this project does

- Scrapes job postings from SA job boards and extracts skill vs. degree signals
- Downloads and processes Stats SA QLFS labour data
- Trains ML models to forecast which skills are growing in demand by industry
- Classifies whether degree requirements in postings are justified
- Scores unemployment risk based on skills profile, not qualifications
- Runs a fairness audit to quantify the bias introduced by degree gatekeeping
- Serves predictions via a FastAPI endpoint and a Streamlit dashboard

---

## Project layout

```
sa-talent-demand-predictor/
├── data/
│   ├── raw/            # Untouched source data (QLFS, scraped posts, SETA PDFs)
│   ├── processed/      # Cleaned features and labels
│   └── external/       # ONET, ISCO skill taxonomy references
├── notebooks/          # EDA and model experimentation
├── src/
│   ├── ingestion/      # Scrapers and downloaders
│   ├── processing/     # Cleaning, feature engineering, NLP skill extraction
│   ├── models/         # Training scripts for all three models
│   ├── evaluation/     # Metrics and bias auditing
│   └── utils/          # Shared helpers (DB, logging)
├── api/                # FastAPI talent matching service
├── dashboard/          # Streamlit visualisation app
└── tests/              # Unit tests
```

---

## Quickstart

```bash
git clone <repo>
cd sa-talent-demand-predictor

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

cp .env.example .env      # Fill in any credentials
```

Run the pipeline step by step:

```bash
# 1. Scrape job postings
python src/ingestion/scraper_careerjunction.py

# 2. Build features
python src/processing/feature_engineer.py

# 3. Train models
python src/models/skill_demand_forecast.py
python src/models/degree_classifier.py
python src/models/unemployment_risk.py

# 4. Bias audit
python src/evaluation/bias_audit.py

# 5. Start API
uvicorn api.main:app --reload

# 6. Launch dashboard
streamlit run dashboard/app.py
```

---

## Data sources

| Source | What it gives you |
|--------|------------------|
| Stats SA QLFS | Unemployment by province, sector, education level |
| CareerJunction / Pnet | SA job postings with skill and degree signals |
| SETA Sector Skills Plans | Industry-level skill demand forecasts (PDFs) |
| LMIP | Labour market intelligence by occupation |
| ONET | Skill-to-occupation taxonomy (adapted for SA) |
| World Bank | Macro economic and employment indicators |

---

## Models

1. **Skill Demand Forecasting** — Prophet time series predicting skill mention
   frequency per industry over 12 months
2. **Degree vs Skills Classifier** — Random Forest on TF-IDF job descriptions,
   flags postings where degree requirements look unjustified
3. **Unemployment Risk Score** — XGBoost pipeline scoring how at-risk a skills
   profile is given local demand signals

---

## Fairness & bias

After modelling, we run a fairness audit using Microsoft Fairlearn to measure
whether the models disadvantage people who have skills but no formal degree.
We also run a counterfactual analysis showing how the eligible talent pool
changes when degree filters are removed.

---

## Stack

Python 3.10+ · pandas · scikit-learn · XGBoost · Prophet · spaCy ·
HuggingFace Transformers · Fairlearn · FastAPI · Streamlit · MLflow · DuckDB · DVC
