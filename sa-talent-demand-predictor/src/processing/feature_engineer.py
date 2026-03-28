"""
Builds the main feature set from cleaned job postings.

Reads:  data/processed/clean_postings.csv
Writes: data/processed/features/job_features.csv
"""

import ast
from pathlib import Path

import pandas as pd
from src.processing.skill_extractor import extract_skills, requires_degree, extract_education_level
from src.utils.logger import logger

INPUT = Path("data/processed/clean_postings.csv")
OUTPUT = Path("data/processed/features/job_features.csv")

# Simple rule-based industry tagger — extend this with a classifier later
INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "tech": ["developer", "engineer", "data", "software", "it ", "cloud", "devops", "analyst"],
    "finance": ["accountant", "auditor", "finance", "banking", "investment", "payroll", "tax"],
    "healthcare": ["nurse", "doctor", "pharmacist", "health", "clinical", "hospital", "care"],
    "education": ["teacher", "lecturer", "tutor", "principal", "curriculum", "school"],
    "engineering": ["civil engineer", "mechanical", "electrical engineer", "structural", "hvac"],
    "construction": ["construction", "site manager", "quantity surveyor", "foreman"],
    "logistics": ["logistics", "supply chain", "warehouse", "driver", "freight", "procurement"],
    "sales": ["sales", "business development", "account manager", "retail", "commercial"],
    "hr": ["hr ", "human resources", "recruiter", "talent", "people"],
    "trades": ["welding", "plumbing", "artisan", "fitter", "electrician", "boilermaker"],
}

PROVINCES = [
    "gauteng", "western cape", "kwazulu-natal", "eastern cape",
    "limpopo", "mpumalanga", "north west", "free state", "northern cape",
]


def tag_industry(title: str, description: str) -> str:
    combined = f"{title} {description}".lower()
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return industry
    return "other"


def tag_province(location: str) -> str:
    loc_lower = str(location).lower()
    # Alias common city names to provinces
    city_map = {
        "johannesburg": "gauteng", "joburg": "gauteng", "sandton": "gauteng",
        "pretoria": "gauteng", "centurion": "gauteng", "midrand": "gauteng",
        "cape town": "western cape", "bellville": "western cape", "stellenbosch": "western cape",
        "durban": "kwazulu-natal", "pinetown": "kwazulu-natal",
        "port elizabeth": "eastern cape", "gqeberha": "eastern cape",
        "bloemfontein": "free state", "polokwane": "limpopo",
        "nelspruit": "mpumalanga", "mbombela": "mpumalanga",
        "kimberley": "northern cape", "rustenburg": "north west",
    }
    for city, province in city_map.items():
        if city in loc_lower:
            return province
    for province in PROVINCES:
        if province in loc_lower:
            return province
    return "unknown"


def build_features(input_path: Path = INPUT, output_path: Path = OUTPUT) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    logger.info(f"Building features for {len(df)} postings...")

    df["skills"] = df["description"].apply(extract_skills)
    df["skill_count"] = df["skills"].apply(len)
    df["requires_degree"] = df["description"].apply(requires_degree).astype(int)
    df["education_level_required"] = df["description"].apply(extract_education_level)
    df["industry"] = df.apply(
        lambda r: tag_industry(str(r.get("title", "")), str(r.get("description", ""))), axis=1
    )
    df["province"] = df["location"].apply(tag_province)

    # Stringify the list so it survives a CSV round-trip
    df["skills_str"] = df["skills"].apply(lambda s: ", ".join(s))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Feature set saved → {output_path}  ({len(df)} rows, {df.columns.tolist()})")
    return df


if __name__ == "__main__":
    build_features()
