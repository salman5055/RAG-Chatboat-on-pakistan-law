import re
from pathlib import Path
from pypdf import PdfReader


PDF_PATH = Path("data/peca_2016.pdf")


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from the PDF."""
    reader = PdfReader(pdf_path)

    pages_text = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages_text.append(text)

    return "\n".join(pages_text)


def find_legal_sections(text: str) -> list[str]:
    """Find numbered legal sections in the extracted text."""
    pattern = r"(?m)^\s*(\d+[A-Z]?)\.\s+.+$"

    sections = re.findall(pattern, text)

    return sections


def main() -> None:
    """Inspect the legal section structure of the PDF."""
    text = extract_pdf_text(PDF_PATH)

    sections = find_legal_sections(text)

    print("Total detected section headings:", len(sections))
    print("-" * 50)

    print("First 30 detected section numbers:")

    for section in sections[:30]:
        print(section)


if __name__ == "__main__":
    main()