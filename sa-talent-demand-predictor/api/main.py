"""
SA Talent Demand API

Endpoints:
  POST /match   — score a candidate profile and return best-fit industries
  GET  /demand  — return top in-demand skills (from forecast CSV)
  GET  /health  — liveness check
"""

import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import CandidateProfile, DemandResponse, MatchResponse
from src.utils.logger import logger

MODEL_PATH = Path("data/processed/models/unemployment_risk.pkl")
FORECAST_PATH = Path("data/processed/forecasts/skill_demand_forecast.csv")

# Loaded at startup — avoid reloading on every request
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            _state["risk_model"] = pickle.load(f)
        logger.info("Unemployment risk model loaded")
    else:
        logger.warning(f"Model not found at {MODEL_PATH} — run `make train` first")
        _state["risk_model"] = None

    # Load forecasts
    if FORECAST_PATH.exists():
        _state["forecasts"] = pd.read_csv(FORECAST_PATH, parse_dates=["ds"])
        logger.info(f"Forecasts loaded — {len(_state['forecasts'])} rows")
    else:
        _state["forecasts"] = None

    yield
    _state.clear()


app = FastAPI(
    title="SA Talent Demand API",
    description=(
        "Predicts skill demand across South African industries and scores "
        "candidate profiles without using degree as a primary filter."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _state.get("risk_model") is not None,
        "forecasts_loaded": _state.get("forecasts") is not None,
    }


@app.post("/match", response_model=MatchResponse)
def match_talent(profile: CandidateProfile):
    model = _state.get("risk_model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run `make train`")

    input_df = pd.DataFrame([{
        "skill_count": len(profile.skills),
        "age": profile.age,
        "province": profile.province.lower(),
        "industry": "unknown",
        "education_level": profile.education_level.lower(),
    }])

    try:
        unemployed_prob = float(model.predict_proba(input_df)[0][0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    # Rank industries by how well the candidate's skills map to demand
    # (simplified — replace with actual skill-demand join once forecasts are live)
    industry_scores = _rank_industries(profile.skills)

    return MatchResponse(
        unemployment_risk_score=round(unemployed_prob, 3),
        recommended_industries=industry_scores[:3],
        skills_matched=profile.skills,
        top_missing_skills=_suggest_missing_skills(profile.skills),
        degree_not_required_roles=_roles_without_degree_filter(profile.skills),
    )


@app.get("/demand", response_model=DemandResponse)
def skill_demand(top_n: int = 10):
    forecasts = _state.get("forecasts")
    if forecasts is None:
        raise HTTPException(status_code=503, detail="Forecast data not loaded — run `make train`")

    # Take the most recent forecast period per skill
    latest = (
        forecasts.sort_values("ds")
        .groupby("skill")
        .last()
        .reset_index()
        .sort_values("yhat", ascending=False)
        .head(top_n)
    )

    return DemandResponse(
        skills=[
            {"skill": row["skill"], "predicted_demand": round(row["yhat"], 2)}
            for _, row in latest.iterrows()
        ]
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_INDUSTRY_SKILL_MAP = {
    "tech": ["python", "sql", "javascript", "cloud", "docker", "machine learning", "data analysis"],
    "finance": ["accounting", "excel", "financial reporting", "auditing", "payroll", "budgeting"],
    "healthcare": ["nursing", "patient care"],
    "education": ["teaching", "curriculum"],
    "engineering": ["autocad", "mechanical", "electrical", "instrumentation"],
    "construction": ["construction", "welding", "plumbing"],
    "logistics": ["logistics", "supply chain", "procurement"],
    "trades": ["welding", "plumbing", "electrical", "artisan"],
}

_ALL_SKILLS = [
    "python", "sql", "excel", "machine learning", "data analysis", "cloud",
    "project management", "communication", "leadership", "docker", "javascript",
    "power bi", "accounting", "nursing", "teaching",
]


def _rank_industries(skills: list[str]) -> list[str]:
    skill_set = set(s.lower() for s in skills)
    scores = {
        industry: len(skill_set & set(kws))
        for industry, kws in _INDUSTRY_SKILL_MAP.items()
    }
    return sorted(scores, key=scores.get, reverse=True)


def _suggest_missing_skills(skills: list[str]) -> list[str]:
    skill_set = set(s.lower() for s in skills)
    return [s for s in _ALL_SKILLS if s not in skill_set][:5]


def _roles_without_degree_filter(skills: list[str]) -> list[str]:
    skill_set = set(s.lower() for s in skills)
    candidates = []
    if "python" in skill_set or "sql" in skill_set:
        candidates.append("Junior Data Analyst")
    if "machine learning" in skill_set:
        candidates.append("ML Engineer (Associate)")
    if "accounting" in skill_set or "excel" in skill_set:
        candidates.append("Bookkeeper / Junior Accountant")
    if "welding" in skill_set or "electrical" in skill_set:
        candidates.append("Artisan / Technician")
    if "nursing" in skill_set:
        candidates.append("Community Health Worker")
    if not candidates:
        candidates = ["Customer Service Representative", "Data Capturer", "Admin Coordinator"]
    return candidates
