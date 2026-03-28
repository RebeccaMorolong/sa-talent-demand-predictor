"""
Tests for processing — skill extraction and feature engineering.
"""

import pytest
from src.processing.skill_extractor import extract_skills, requires_degree, extract_education_level


def test_extract_skills_finds_known_skills():
    text = "We need someone with Python, SQL, and experience in machine learning."
    skills = extract_skills(text)
    assert "python" in skills
    assert "sql" in skills
    assert "machine learning" in skills


def test_extract_skills_returns_empty_for_no_match():
    skills = extract_skills("The candidate must be enthusiastic and reliable.")
    assert skills == []


def test_requires_degree_true():
    assert requires_degree("Applicants must hold a Bachelor's degree in Commerce.") is True
    assert requires_degree("A B.Sc or B.Com is required.") is True
    assert requires_degree("Postgraduate qualification preferred.") is True


def test_requires_degree_false():
    assert requires_degree("Experience in welding and 3 years on the job.") is False
    assert requires_degree("Must have a valid driver's license and good communication.") is False


def test_education_level_detection():
    assert extract_education_level("PhD in Computer Science required.") == "phd"
    assert extract_education_level("Master's degree in Finance preferred.") == "masters"
    assert extract_education_level("Bachelor of Science in Engineering.") == "degree"
    assert extract_education_level("National Diploma in Electrical Engineering.") == "diploma"
    assert extract_education_level("Grade 12 / Matric certificate.") == "matric"
    assert extract_education_level("Good communication skills required.") == "not_specified"
