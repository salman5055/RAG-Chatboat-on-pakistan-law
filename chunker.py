from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_pages(
    pdf_path: str,
    start_page: int = 1,
) -> list[dict]:
    """
    Read a PDF page by page and keep the page number with its text.

    Args:
        pdf_path: Path to the PDF.
        start_page: First PDF page to include.

    Returns:
        List containing page number and extracted text.
    """

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_file}"
        )

    reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        # Skip pages before start_page
        if page_number < start_page:
            continue

        text = page.extract_text() or ""

        # Skip completely empty pages
        if text.strip():
            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                }
            )

    return pages


def create_chunks(
    pages: list[dict],
    source_name: str = "unknown.pdf",
) -> list[dict]:
    """
    Split PDF pages into smaller chunks
    while preserving useful metadata.

    Args:
        pages: List of PDF pages.
        source_name: Name of the source PDF.

    Returns:
        List of chunks with metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    all_chunks = []

    chunk_id = 0

    for page in pages:

        chunks = splitter.split_text(
            page["text"]
        )

        for chunk in chunks:

            all_chunks.append(
                {
                    "source": source_name,
                    "page_number": page["page_number"],
                    "chunk_id": chunk_id,
                    "text": chunk,
                }
            )

            chunk_id += 1

    return all_chunks


def main() -> None:
    """
    Test PDF loading and chunking.
    """

    pdf_path = "data/peca_2016.pdf"

    # PECA actual law starts from PDF page 6
    pages = load_pdf_pages(
        pdf_path,
        start_page=6,
    )

    chunks = create_chunks(
        pages,
        source_name="peca_2016.pdf",
    )

    print("=" * 50)
    print("PAGE-AWARE PDF CHUNKING TEST")
    print("=" * 50)

    print("Total PDF pages:", len(pages))
    print("Total chunks:", len(chunks))

    print("-" * 50)

    if chunks:

        first_chunk = chunks[0]

        print("First chunk metadata:")
        print("-" * 50)

        print("Source:", first_chunk["source"])
        print("Page:", first_chunk["page_number"])
        print("Chunk ID:", first_chunk["chunk_id"])

        print("-" * 50)
        print("First chunk text:")
        print("-" * 50)

        print(first_chunk["text"])

    else:
        print("No chunks were created.")


if __name__ == "__main__":
    main()