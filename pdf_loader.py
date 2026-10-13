from pathlib import Path
from pypdf import PdfReader


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from all pages of a PDF.

    Args:
        pdf_path: Path of the PDF file.

    Returns:
        Combined text from the PDF.
    """

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_file}"
        )

    reader = PdfReader(pdf_file)

    pages_text = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages_text.append(text)

    return "\n".join(pages_text)


def main() -> None:
    pdf_path = "data/peca_2016.pdf"

    text = extract_pdf_text(pdf_path)

    print("=" * 50)
    print("PDF EXTRACTION TEST")
    print("=" * 50)

    print("PDF:", pdf_path)
    print("Total pages:", len(PdfReader(pdf_path).pages))
    print("Total characters:", len(text))
    print("Total words:", len(text.split()))

    print("-" * 50)
    print("First 500 characters:")
    print("-" * 50)

    print(text[:500])


if __name__ == "__main__":
    main()