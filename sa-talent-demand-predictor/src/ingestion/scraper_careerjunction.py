"""
Scrapes job postings from CareerJunction.

Saves raw results to data/raw/job_postings/careerjunction.json.

Scraping etiquette:
- Respects SCRAPER_DELAY from .env (default 2 s between requests)
- Sends a descriptive User-Agent so the site knows what's hitting it
- Does NOT bypass any login walls or rate-limiting mechanisms
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from src.utils.logger import logger

SAVE_PATH = Path("data/raw/job_postings/careerjunction.json")
DELAY = float(os.getenv("SCRAPER_DELAY", "2.0"))
HEADERS = {
    "User-Agent": os.getenv(
        "SCRAPER_USER_AGENT",
        "Mozilla/5.0 (compatible; ResearchBot/1.0)",
    )
}

# Broad keywords covering a cross-section of SA industries
SEARCH_KEYWORDS = [
    "software developer",
    "data analyst",
    "data engineer",
    "electrician",
    "project manager",
    "registered nurse",
    "accountant",
    "civil engineer",
    "teacher",
    "logistics coordinator",
    "sales representative",
    "HR manager",
    "welding",
    "artisan",
]


def _parse_listing(tag) -> dict | None:
    """Extract fields from a single job card element."""
    try:
        title = tag.select_one(".job-result-title")
        company = tag.select_one(".job-result-company-name")
        location = tag.select_one(".job-result-location")
        snippet = tag.select_one(".job-result-description")
        link_tag = tag.select_one("a[href]")

        return {
            "title": title.get_text(strip=True) if title else None,
            "company": company.get_text(strip=True) if company else None,
            "location": location.get_text(strip=True) if location else None,
            "description": snippet.get_text(strip=True) if snippet else None,
            "url": "https://www.careerjunction.co.za" + link_tag["href"]
            if link_tag
            else None,
            "source": "careerjunction",
            "date_scraped": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.debug(f"Could not parse listing: {exc}")
        return None


def scrape_keyword(keyword: str, pages: int = 5) -> list[dict]:
    base = "https://www.careerjunction.co.za/jobs/results"
    results = []

    for page in range(1, pages + 1):
        params = {"keywords": keyword, "page": page}
        try:
            resp = requests.get(base, headers=HEADERS, params=params, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            listings = soup.select(".job-result-item")

            if not listings:
                logger.debug(f"No listings on page {page} for '{keyword}' — stopping early")
                break

            for tag in listings:
                job = _parse_listing(tag)
                if job and job["title"]:
                    results.append(job)

            logger.info(f"  '{keyword}' page {page}: {len(listings)} listings")
            time.sleep(DELAY)

        except requests.RequestException as exc:
            logger.warning(f"Request failed for '{keyword}' page {page}: {exc}")
            break

    return results


def scrape_all(pages_per_keyword: int = 5) -> list[dict]:
    all_jobs: list[dict] = []
    seen_urls: set[str] = set()

    for kw in SEARCH_KEYWORDS:
        logger.info(f"Scraping: {kw}")
        jobs = scrape_keyword(kw, pages=pages_per_keyword)
        for job in jobs:
            url = job.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                all_jobs.append(job)

    logger.info(f"Total unique postings collected: {len(all_jobs)}")
    return all_jobs


def save(jobs: list[dict], path: Path = SAVE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(jobs)} postings → {path}")


if __name__ == "__main__":
    jobs = scrape_all(pages_per_keyword=5)
    save(jobs)
