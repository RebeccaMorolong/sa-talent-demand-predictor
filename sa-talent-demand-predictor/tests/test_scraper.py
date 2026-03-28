"""
Tests for ingestion layer.
These use mocking so they run without hitting live websites.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.scraper_careerjunction import _parse_listing, save
import json
import tempfile
from pathlib import Path


def _make_tag(title="Data Analyst", company="Acme Ltd", location="Gauteng", description="Python SQL"):
    tag = MagicMock()
    tag.select_one = lambda sel: MagicMock(
        **{"get_text.return_value": {
            ".job-result-title": title,
            ".job-result-company-name": company,
            ".job-result-location": location,
            ".job-result-description": description,
        }.get(sel, "")}
    )
    link = MagicMock()
    link.__getitem__ = lambda self, key: "/jobs/123"
    tag.select_one = lambda sel: (
        MagicMock(**{"get_text.return_value": title}) if "title" in sel
        else MagicMock(**{"get_text.return_value": company}) if "company" in sel
        else MagicMock(**{"get_text.return_value": location}) if "location" in sel
        else MagicMock(**{"get_text.return_value": description}) if "description" in sel
        else link if "href" in sel
        else None
    )
    return tag


def test_parse_listing_returns_dict():
    tag = _make_tag()
    result = _parse_listing(tag)
    assert result is not None
    assert "title" in result
    assert "source" in result
    assert result["source"] == "careerjunction"


def test_save_creates_file():
    jobs = [{"title": "Developer", "company": "Test", "location": "Cape Town",
              "description": "Python", "source": "careerjunction", "date_scraped": "2024-01-01"}]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "jobs.json"
        save(jobs, path)
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert len(loaded) == 1
        assert loaded[0]["title"] == "Developer"
