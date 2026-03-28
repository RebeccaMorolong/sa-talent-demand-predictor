"""
Extracts skills and degree signals from job descriptions.

Two approaches:
  1. Keyword matching against a curated SA skill list (fast, explainable)
  2. spaCy NER for entity-level extraction (richer, requires en_core_web_sm)

Run: python -m spacy download en_core_web_sm
"""

import re
from functools import lru_cache

import spacy

# ---------------------------------------------------------------------------
# Skill taxonomy — expand this list as you collect more postings
# ---------------------------------------------------------------------------
SKILLS: list[str] = [
    # Tech
    "python", "sql", "java", "javascript", "react", "node.js", "django",
    "machine learning", "deep learning", "data analysis", "data engineering",
    "cloud", "aws", "azure", "gcp", "docker", "kubernetes", "linux",
    "excel", "power bi", "tableau", "autocad",
    # Business / professional
    "project management", "agile", "scrum", "leadership", "communication",
    "stakeholder management", "budgeting", "financial reporting", "accounting",
    "auditing", "payroll", "procurement", "supply chain", "logistics",
    # Trades / technical
    "welding", "plumbing", "electrical", "construction", "hvac",
    "mechanical", "instrumentation", "artisan",
    # People services
    "nursing", "patient care", "teaching", "curriculum", "counselling",
    "social work", "community development",
    # Languages
    "zulu", "xhosa", "afrikaans", "sotho",
]

# ---------------------------------------------------------------------------
# Degree / qualification requirement patterns
# ---------------------------------------------------------------------------
DEGREE_PATTERNS: list[str] = [
    r"bachelor'?s?\s+degree",
    r"b\.?\s*sc",
    r"b\.?\s*com",
    r"b\.?\s*tech",
    r"honours\s+degree",
    r"master'?s?\s+degree",
    r"m\.?\s*sc",
    r"mba",
    r"phd",
    r"doctorate",
    r"degree\s+in",
    r"university\s+qualification",
    r"tertiary\s+qualification",
    r"national\s+diploma",
    r"postgraduate",
]

_DEGREE_RE = re.compile("|".join(DEGREE_PATTERNS), flags=re.IGNORECASE)


@lru_cache(maxsize=1)
def _nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None


def extract_skills(text: str) -> list[str]:
    """Return a sorted list of skills found in the text."""
    lower = text.lower()
    return sorted({s for s in SKILLS if s in lower})


def requires_degree(text: str) -> bool:
    """Return True if the posting explicitly requires a formal degree."""
    return bool(_DEGREE_RE.search(text))


def extract_education_level(text: str) -> str:
    """Classify the highest education level required."""
    text_lower = text.lower()
    if re.search(r"phd|doctorate", text_lower):
        return "phd"
    if re.search(r"master|mba|m\.sc", text_lower):
        return "masters"
    if re.search(r"honours", text_lower):
        return "honours"
    if re.search(r"bachelor|b\.sc|b\.com|b\.tech|degree in", text_lower):
        return "degree"
    if re.search(r"national diploma|nd\b", text_lower):
        return "diploma"
    if re.search(r"matric|grade 12", text_lower):
        return "matric"
    return "not_specified"


def enrich_row(row: dict) -> dict:
    """Add skill and degree fields to a single job posting dict."""
    description = row.get("description", "") or ""
    skills = extract_skills(description)
    row["skills"] = skills
    row["skill_count"] = len(skills)
    row["requires_degree"] = int(requires_degree(description))
    row["education_level_required"] = extract_education_level(description)
    return row
