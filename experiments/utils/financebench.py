"""Shared utilities for downloading and converting FinanceBench 10-K PDFs."""

from pathlib import Path

import fitz
import requests

FINANCEBENCH_PDF_BASE = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/"
)


def download_pdf(filename: str, output_dir: Path) -> Path:
    """Download a PDF from the FinanceBench GitHub repo."""
    output_path = output_dir / filename
    if output_path.exists():
        print(f"Already downloaded: {output_path}")
        return output_path

    url = FINANCEBENCH_PDF_BASE + filename
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(resp.content)
    print(f"Saved to {output_path} ({len(resp.content)} bytes)")
    return output_path


def pdf_to_text(pdf_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    with fitz.open(str(pdf_path)) as doc:
        return "\n".join(page.get_text() for page in doc)


def ensure_text_file(company: str, year: int, pdf_dir: Path, text_dir: Path) -> Path:
    """Ensure the text file exists, downloading and converting the PDF if needed."""
    doc_name = f"{company.upper()}_{year}_10K"
    text_path = text_dir / f"{doc_name}.txt"

    if text_path.exists():
        print(f"Text file already exists: {text_path}")
        return text_path

    pdf_filename = f"{doc_name}.pdf"
    pdf_path = download_pdf(pdf_filename, pdf_dir)

    text = pdf_to_text(pdf_path)
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    print(f"Converted {pdf_filename} -> {text_path.name} ({len(text)} chars)")
    return text_path
