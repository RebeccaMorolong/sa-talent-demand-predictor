"""
Extract text from SETA Sector Skills Plan PDFs.

Drop any SETA PDF into data/raw/seta_reports/ and run this script.
Outputs one JSON file per PDF in data/processed/seta_text/.
"""

import json
from pathlib import Path

import pdfplumber
from src.utils.logger import logger

INPUT_DIR = Path("data/raw/seta_reports")
OUTPUT_DIR = Path("data/processed/seta_text")


def extract_pdf(pdf_path: Path) -> dict:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i + 1, "text": text.strip()})

    return {
        "source_file": pdf_path.name,
        "total_pages": len(pages),
        "pages": pages,
    }


def extract_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDFs found in {INPUT_DIR}. Download SETA reports first.")
        return

    for pdf_path in pdf_files:
        out_path = OUTPUT_DIR / (pdf_path.stem + ".json")
        if out_path.exists():
            logger.info(f"Already extracted: {pdf_path.name}")
            continue

        logger.info(f"Extracting: {pdf_path.name}")
        result = extract_pdf(pdf_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"  → {result['total_pages']} pages saved to {out_path.name}")


if __name__ == "__main__":
    extract_all()
