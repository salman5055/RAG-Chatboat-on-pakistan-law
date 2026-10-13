from pathlib import Path
from pypdf import PdfReader


PDF_PATH = Path("data/peca_2016.pdf")


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from all pages of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Combined text extracted from all PDF pages.
    """
    reader = PdfReader(pdf_path)

    pages_text = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages_text.append(text)

    return "\n".join(pages_text)


def main() -> None:
    """Extract the law PDF and display basic extraction statistics."""
    text = extract_pdf_text(PDF_PATH)

    print("PDF extraction successful")
    print("-" * 50)
    print("Total characters:", len(text))
    print("Total words:", len(text.split()))
    print("-" * 50)
    print("First 500 characters:")
    print(text[:500])


if __name__ == "__main__":
    main()