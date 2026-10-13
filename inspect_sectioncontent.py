from pathlib import Path
from pypdf import PdfReader


PDF_PATH = Path("data/peca_2016.pdf")


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from all pages of the PDF."""
    reader = PdfReader(pdf_path)

    pages_text = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages_text.append(text)

    return "\n".join(pages_text)


def show_section_occurrences(text: str, section_number: str) -> None:
    """Show all occurrences of a section number in the extracted text."""
    marker = f"{section_number}."

    start = 0
    occurrence = 0

    while True:
        position = text.find(marker, start)

        if position == -1:
            break

        occurrence += 1

        print("=" * 60)
        print(f"OCCURRENCE {occurrence}")
        print("=" * 60)

        print(text[position:position + 1000])

        start = position + len(marker)

        if occurrence >= 5:
            break


def main() -> None:
    """Extract the PDF and inspect occurrences of Section 1."""
    text = extract_pdf_text(PDF_PATH)

    show_section_occurrences(text, "1")


if __name__ == "__main__":
    main()