from pathlib import Path
from pypdf import PdfReader


PDF_PATH = Path("data/peca_2016.pdf")


def find_section_on_pages(pdf_path: Path, section_number: str) -> None:
    """Find pages containing a specific section heading."""
    reader = PdfReader(pdf_path)

    search_text = f"{section_number}."

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if search_text in text:
            print("=" * 60)
            print(f"PAGE {page_number}")
            print("=" * 60)

            position = text.find(search_text)

            print(text[position:position + 1500])


def main() -> None:
    """Find pages containing Section 1."""
    find_section_on_pages(PDF_PATH, "1")


if __name__ == "__main__":
    main()