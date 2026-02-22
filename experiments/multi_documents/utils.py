from pathlib import Path

import requests
import fitz

SCRIPT_DIR = Path(__file__).resolve().parent
PDF_DIR = SCRIPT_DIR / "data" / "pdfs"
TEXT_DIR = SCRIPT_DIR / "data" / "texts"

FINANCEBENCH_PDF_BASE = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/"
)

DOCUMENTS = {
    "AMD_2022_10K": "AMD_2022_10K.pdf",
    "PEPSICO_2022_10K": "PEPSICO_2022_10K.pdf",
}


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


def download_and_convert():
    """Download all PDFs and convert them to text files."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    for doc_name, pdf_filename in DOCUMENTS.items():
        pdf_path = download_pdf(pdf_filename, PDF_DIR)
        text = pdf_to_text(pdf_path)

        text_path = TEXT_DIR / f"{doc_name}.txt"
        text_path.write_text(text, encoding="utf-8")
        print(f"Converted {pdf_filename} -> {text_path.name} ({len(text)} chars)")


if __name__ == "__main__":
    download_and_convert()
