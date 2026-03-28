"""
Bias & Fairness Audit.

Quantifies how degree gatekeeping in SA job postings creates
structural barriers for skilled candidates without formal degrees.

Three analyses:
  1. Degree requirement rate by industry — which sectors gate-keep most?
  2. Skill complexity vs degree requirement — are degree-gated jobs
     actually more complex?
  3. Counterfactual talent pool — how many more people could apply
     if degree requirements were dropped?

Usage:
    python src/evaluation/bias_audit.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.utils.logger import logger

FEATURES_PATH = Path("data/processed/features/job_features.csv")
REPORT_DIR = Path("data/processed/audit")


def load() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH)
    df["requires_degree"] = df["requires_degree"].fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Analysis 1: Degree rate by industry
# ---------------------------------------------------------------------------

def degree_rate_by_industry(df: pd.DataFrame) -> pd.DataFrame:
    rate = (
        df.groupby("industry")["requires_degree"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "degree_rate", "count": "total_postings"})
        .sort_values("degree_rate", ascending=False)
    )
    logger.info("Degree requirement rate by industry:")
    for _, row in rate.iterrows():
        logger.info(
            f"  {row['industry']:20s}  {row['degree_rate']*100:5.1f}%  "
            f"({row['total_postings']} postings)"
        )
    return rate


# ---------------------------------------------------------------------------
# Analysis 2: Skill complexity vs degree requirement
# ---------------------------------------------------------------------------

def skill_complexity_vs_degree(df: pd.DataFrame) -> dict:
    degree_jobs = df[df["requires_degree"] == 1]["skill_count"]
    no_degree_jobs = df[df["requires_degree"] == 0]["skill_count"]

    stats = {
        "mean_skills_degree_required": degree_jobs.mean(),
        "mean_skills_no_degree": no_degree_jobs.mean(),
        "median_skills_degree_required": degree_jobs.median(),
        "median_skills_no_degree": no_degree_jobs.median(),
    }

    logger.info("\nSkill complexity comparison:")
    logger.info(f"  Degree-required postings  → mean skills: {stats['mean_skills_degree_required']:.1f}")
    logger.info(f"  No-degree postings        → mean skills: {stats['mean_skills_no_degree']:.1f}")

    diff = stats["mean_skills_degree_required"] - stats["mean_skills_no_degree"]
    if abs(diff) < 1.0:
        logger.info(
            "  ⚠  Minimal difference in skill complexity — degree requirement may not be justified."
        )
    else:
        logger.info(f"  Degree-gated jobs require {diff:+.1f} more skills on average.")

    return stats


# ---------------------------------------------------------------------------
# Analysis 3: Counterfactual talent pool
# ---------------------------------------------------------------------------

def counterfactual_talent_pool(df: pd.DataFrame) -> dict:
    total = len(df)
    gated = df[df["requires_degree"] == 1]
    n_gated = len(gated)
    pct_gated = n_gated / total * 100

    # Of the degree-gated postings, how many have skill_count >= median of no-degree postings?
    median_skill_no_degree = df[df["requires_degree"] == 0]["skill_count"].median()
    potentially_overgated = gated[gated["skill_count"] <= median_skill_no_degree]

    result = {
        "total_postings": total,
        "degree_gated": n_gated,
        "pct_gated": pct_gated,
        "potentially_overgated": len(potentially_overgated),
        "pct_overgated_of_gated": len(potentially_overgated) / n_gated * 100 if n_gated else 0,
    }

    logger.info("\nCounterfactual talent pool analysis:")
    logger.info(f"  Total postings:           {total:,}")
    logger.info(f"  Degree-gated:             {n_gated:,}  ({pct_gated:.1f}%)")
    logger.info(
        f"  Potentially over-gated:   {len(potentially_overgated):,}  "
        f"({result['pct_overgated_of_gated']:.1f}% of degree-gated)"
    )
    logger.info(
        "  → Removing these requirements would open "
        f"{len(potentially_overgated):,} roles to skills-only candidates."
    )

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    logger.info(f"Running bias audit on {len(df)} postings\n")

    degree_rates = degree_rate_by_industry(df)
    degree_rates.to_csv(REPORT_DIR / "degree_rate_by_industry.csv", index=False)

    complexity = skill_complexity_vs_degree(df)
    pool = counterfactual_talent_pool(df)

    summary = {**complexity, **pool}
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(REPORT_DIR / "audit_summary.csv", index=False)

    logger.info(f"\nAudit complete — results saved to {REPORT_DIR}")


if __name__ == "__main__":
    run()
