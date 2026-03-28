"""
Unemployment Risk Model.

Trains an XGBoost pipeline that scores how at-risk a candidate is
of remaining unemployed given their skills profile, province, and
education level — relative to local demand signals.

Note: This model needs a merged dataset of job-seeker profiles +
QLFS employment status. Until you have that, the script generates
a synthetic training set so the pipeline can be validated end-to-end.
Replace `_synthetic_data()` with your real QLFS-joined dataset.

Usage:
    python src/models/unemployment_risk.py
"""

import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from src.utils.logger import logger

MODEL_OUT = Path("data/processed/models/unemployment_risk.pkl")

PROVINCES = [
    "gauteng", "western cape", "kwazulu-natal", "eastern cape",
    "limpopo", "mpumalanga", "north west", "free state", "northern cape", "unknown",
]
INDUSTRIES = [
    "tech", "finance", "healthcare", "education", "engineering",
    "construction", "logistics", "sales", "hr", "trades", "other",
]
EDUCATION_LEVELS = ["matric", "diploma", "degree", "honours", "masters", "phd", "not_specified"]


def _synthetic_data(n: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic training data as a stand-in until real QLFS profiles
    are available. The relationships here are intentionally realistic:
    - Higher skill count → lower unemployment risk
    - Degree required in local market → lower risk for graduates
    - Rural provinces → higher risk
    """
    rng = np.random.default_rng(42)
    province = rng.choice(PROVINCES, n)
    industry = rng.choice(INDUSTRIES, n)
    education = rng.choice(EDUCATION_LEVELS, n)
    age = rng.integers(18, 60, n)
    skill_count = rng.integers(0, 12, n)

    # Build a rough unemployment probability
    risk = 0.45
    risk -= skill_count * 0.03
    risk += np.where(np.isin(province, ["limpopo", "eastern cape", "north west"]), 0.10, 0)
    risk -= np.where(np.isin(education, ["degree", "honours", "masters", "phd"]), 0.08, 0)
    risk += np.where(age < 25, 0.12, 0)
    risk = np.clip(risk, 0.05, 0.95)
    employed = rng.binomial(1, 1 - risk, n)

    return pd.DataFrame({
        "skill_count": skill_count,
        "age": age,
        "province": province,
        "industry": industry,
        "education_level": education,
        "employed": employed,
    })


def build_pipeline() -> Pipeline:
    numeric = ["skill_count", "age"]
    categorical = ["province", "industry", "education_level"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])

    return Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def run() -> None:
    df = _synthetic_data()
    logger.info(f"Training on {len(df)} samples (synthetic — replace with real QLFS data)")

    X = df.drop(columns=["employed"])
    y = df["employed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("unemployment_risk")
    with mlflow.start_run(run_name="xgb_risk_model"):
        pipeline = build_pipeline()

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_aucs = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        logger.info(f"CV AUC scores: {cv_aucs.round(3)}  mean={cv_aucs.mean():.3f}")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_params({
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.05,
            "data": "synthetic",
        })
        mlflow.log_metrics({
            "cv_auc_mean": cv_aucs.mean(),
            "cv_auc_std": cv_aucs.std(),
            "test_auc": test_auc,
        })
        mlflow.sklearn.log_model(pipeline, "unemployment_risk")

        logger.info(f"Test AUC: {test_auc:.3f}")
        logger.info("\n" + classification_report(y_test, y_pred))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Model saved → {MODEL_OUT}")


if __name__ == "__main__":
    run()
