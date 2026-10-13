from pathlib import Path

from pypdf import PdfReader


def extract_uploaded_pdf(
    pdf_path: str,
) -> list[dict]:
    """
    Extract text from an uploaded PDF page by page.

    Args:
        pdf_path: Path of the uploaded PDF.

    Returns:
        A list containing page number and extracted text.
    """

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_file}"
        )

    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        text = page.extract_text() or ""

        if text.strip():

            pages.append(
                {
                    "page_number": page_number,
                    "text": text.strip(),
                }
            )

    return pages


def main() -> None:

    pdf_path = "data/peca_2016.pdf"

    pages = extract_uploaded_pdf(pdf_path)

    print("=" * 50)
    print("UPLOADED PDF EXTRACTION TEST")
    print("=" * 50)

    print("PDF:", pdf_path)
    print("Pages with text:", len(pages))

    if pages:

        print("\nFirst page number:")
        print(pages[0]["page_number"])

        print("\nFirst 500 characters:")
        print("-" * 50)
        print(pages[0]["text"][:500])


if __name__ == "__main__":
    main()