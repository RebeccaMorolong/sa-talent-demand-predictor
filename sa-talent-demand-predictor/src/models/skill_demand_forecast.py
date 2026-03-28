"""
Skill Demand Forecasting — Prophet time series model.

For each skill in the top-N most mentioned, fits a weekly Prophet model
and forecasts 52 weeks ahead. Results are saved as CSV and logged to MLflow.

Usage:
    python src/models/skill_demand_forecast.py
"""

import ast
from collections import Counter
from pathlib import Path

import mlflow
import pandas as pd
from prophet import Prophet
from src.utils.logger import logger

FEATURES_PATH = Path("data/processed/features/job_features.csv")
OUTPUT_DIR = Path("data/processed/forecasts")
TOP_N_SKILLS = 10
FORECAST_WEEKS = 52


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date_scraped"])
    df["date_scraped"] = pd.to_datetime(df["date_scraped"], utc=True).dt.tz_localize(None)
    return df


def top_skills(df: pd.DataFrame, n: int = TOP_N_SKILLS) -> list[str]:
    all_skills: list[str] = []
    for entry in df["skills_str"].dropna():
        all_skills.extend([s.strip() for s in entry.split(",") if s.strip()])
    counts = Counter(all_skills)
    return [skill for skill, _ in counts.most_common(n)]


def build_timeseries(df: pd.DataFrame, skill: str) -> pd.DataFrame:
    df = df.copy()
    df["week"] = df["date_scraped"].dt.to_period("W").dt.start_time
    df["has_skill"] = df["skills_str"].fillna("").str.contains(skill, case=False).astype(int)
    ts = df.groupby("week")["has_skill"].sum().reset_index()
    ts.columns = ["ds", "y"]
    return ts


def forecast_skill(skill: str, ts: pd.DataFrame, periods: int = FORECAST_WEEKS) -> pd.DataFrame:
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(ts)
    future = model.make_future_dataframe(periods=periods, freq="W")
    forecast = model.predict(future)
    forecast["skill"] = skill
    return forecast[["ds", "skill", "yhat", "yhat_lower", "yhat_upper"]]


def run() -> None:
    df = load_features()
    skills = top_skills(df)
    logger.info(f"Forecasting demand for top {len(skills)} skills: {skills}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("skill_demand_forecast")

    all_forecasts: list[pd.DataFrame] = []
    for skill in skills:
        ts = build_timeseries(df, skill)
        if len(ts) < 4:
            logger.warning(f"Not enough data to forecast '{skill}' — skipping")
            continue

        with mlflow.start_run(run_name=f"forecast_{skill.replace(' ', '_')}"):
            forecast = forecast_skill(skill, ts)
            all_forecasts.append(forecast)
            mlflow.log_param("skill", skill)
            mlflow.log_param("forecast_weeks", FORECAST_WEEKS)
            mlflow.log_metric("mean_predicted_demand", forecast["yhat"].mean())
            logger.info(f"  {skill}: avg predicted demand = {forecast['yhat'].mean():.1f}")

    if all_forecasts:
        combined = pd.concat(all_forecasts, ignore_index=True)
        out = OUTPUT_DIR / "skill_demand_forecast.csv"
        combined.to_csv(out, index=False)
        logger.info(f"Forecasts saved → {out}")


if __name__ == "__main__":
    run()
