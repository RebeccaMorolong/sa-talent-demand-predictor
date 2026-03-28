"""
Loads raw job postings and applies basic cleaning:
- Drop rows with no title or description
- Normalise text case and whitespace
- Parse dates
- Remove duplicates
"""

import json
from pathlib import Path

import pandas as pd
from src.utils.logger import logger

RAW_PATH = Path("data/raw/job_postings/careerjunction.json")
CLEAN_PATH = Path("data/processed/clean_postings.csv")


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    original_len = len(df)

    df = df.dropna(subset=["title", "description"])
    df["title"] = df["title"].str.strip().str.lower()
    df["company"] = df["company"].str.strip()
    df["location"] = df["location"].fillna("").str.strip().str.lower()
    df["description"] = df["description"].str.strip()
    df["date_scraped"] = pd.to_datetime(df["date_scraped"], utc=True, errors="coerce")

    # Drop near-duplicates — same title, company, and first 100 chars of description
    df["_dedup_key"] = (
        df["title"] + df["company"].fillna("") + df["description"].str[:100]
    )
    df = df.drop_duplicates(subset=["_dedup_key"]).drop(columns=["_dedup_key"])
    df = df.reset_index(drop=True)

    logger.info(f"Cleaning: {original_len} → {len(df)} rows after dedup and null drop")
    return df


def run() -> pd.DataFrame:
    df = load_raw()
    df = clean(df)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    logger.info(f"Clean postings saved → {CLEAN_PATH}")
    return df


if __name__ == "__main__":
    run()
