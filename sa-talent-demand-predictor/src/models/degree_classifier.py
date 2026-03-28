"""
Degree vs Skills Classifier.

Trains a Random Forest on TF-IDF job description vectors to predict
whether a posting requires a formal degree.

The classifier is NOT the end goal — it's a diagnostic tool.
Once trained, we inspect which postings the model flags as "degree required"
and compare that to the actual job complexity (measured by skill count)
to surface potential over-credentialing.

Usage:
    python src/models/degree_classifier.py
"""

import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.utils.logger import logger

FEATURES_PATH = Path("data/processed/features/job_features.csv")
MODEL_OUT = Path("data/processed/models/degree_classifier.pkl")


def load() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH)
    df = df.dropna(subset=["description", "requires_degree"])
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
        )),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def run() -> None:
    df = load()
    X = df["description"]
    y = df["requires_degree"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("degree_classifier")
    with mlflow.start_run(run_name="rf_tfidf_baseline"):
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_params({
            "max_features": 8000,
            "ngram_range": "(1,2)",
            "n_estimators": 300,
        })
        mlflow.log_metrics({
            "accuracy": report["accuracy"],
            "auc": auc,
            "f1_degree": report.get("1", {}).get("f1-score", 0),
            "f1_no_degree": report.get("0", {}).get("f1-score", 0),
        })
        mlflow.sklearn.log_model(pipeline, "degree_classifier")

        logger.info(f"AUC: {auc:.3f}")
        logger.info("\n" + classification_report(y_test, y_pred))

        # Flag postings where model predicts "no degree needed" but posting requires one
        df_test = df.loc[X_test.index].copy()
        df_test["predicted_degree"] = y_pred
        overcredentialed = df_test[
            (df_test["requires_degree"] == 1) & (df_test["predicted_degree"] == 0)
        ]
        logger.info(
            f"Postings flagged as potentially over-credentialed: "
            f"{len(overcredentialed)} / {len(df_test)} ({len(overcredentialed)/len(df_test)*100:.1f}%)"
        )

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Model saved → {MODEL_OUT}")


if __name__ == "__main__":
    run()
