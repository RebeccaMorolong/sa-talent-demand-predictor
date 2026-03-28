"""
Download QLFS data from Stats SA.

Stats SA publishes Quarterly Labour Force Survey results at:
https://www.statssa.gov.za/?page_id=1854&PPN=P0211

Because they don't have a stable direct-download API, this script
gives you two paths:
  1. Auto-download a known URL (update the URLS dict as new quarters land)
  2. Point it at a local file you've already downloaded manually
"""

import os
import requests
from pathlib import Path
from src.utils.logger import logger

SAVE_DIR = Path("data/raw/qlfs")

# Update these URLs each quarter — check stats.gov.za for the latest
KNOWN_URLS: dict[str, str] = {
    "Q3_2024": "https://www.statssa.gov.za/publications/P0211/P02113rdQuarter2024.pdf",
}

HEADERS = {
    "User-Agent": os.getenv(
        "SCRAPER_USER_AGENT",
        "Mozilla/5.0 (compatible; ResearchBot/1.0)",
    )
}


def download_file(url: str, save_path: Path) -> bool:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Saved {save_path.name} ({save_path.stat().st_size / 1024:.0f} KB)")
        return True
    except requests.RequestException as exc:
        logger.error(f"Failed to download {url}: {exc}")
        return False


def download_all() -> None:
    for label, url in KNOWN_URLS.items():
        fname = url.split("/")[-1]
        dest = SAVE_DIR / fname
        if dest.exists():
            logger.info(f"Already have {label} — skipping")
            continue
        logger.info(f"Downloading {label}...")
        download_file(url, dest)


if __name__ == "__main__":
    download_all()
