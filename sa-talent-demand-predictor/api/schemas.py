from pydantic import BaseModel, Field
from typing import List


class CandidateProfile(BaseModel):
    skills: List[str] = Field(..., example=["python", "sql", "data analysis"])
    province: str = Field(..., example="gauteng")
    education_level: str = Field(
        ...,
        example="diploma",
        description="matric | diploma | degree | honours | masters | phd | not_specified",
    )
    age: int = Field(..., ge=15, le=80, example=27)


class MatchResponse(BaseModel):
    unemployment_risk_score: float = Field(
        ..., description="Probability of unemployment (0 = low risk, 1 = high risk)"
    )
    recommended_industries: List[str]
    skills_matched: List[str]
    top_missing_skills: List[str]
    degree_not_required_roles: List[str]


class SkillDemandItem(BaseModel):
    skill: str
    predicted_demand: float


class DemandResponse(BaseModel):
    skills: List[SkillDemandItem]
