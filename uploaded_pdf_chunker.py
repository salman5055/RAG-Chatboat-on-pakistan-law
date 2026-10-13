from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_uploaded_pdf(
    pages: list[dict],
    source_name: str,
) -> list[dict]:
    """
    Split uploaded PDF pages into smaller, searchable chunks.

    Smaller chunks improve retrieval for specific definitions,
    clauses, paragraphs, and legal provisions.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )

    chunks = []

    chunk_id = 0

    for page in pages:

        page_text = page["text"].strip()

        if not page_text:
            continue

        page_chunks = splitter.split_text(
            page_text
        )

        for text in page_chunks:

            text = text.strip()

            if not text:
                continue

            chunks.append(
                {
                    "source": source_name,
                    "page_number": page["page_number"],
                    "chunk_id": chunk_id,
                    "text": text,
                }
            )

            chunk_id += 1

    return chunks


def main() -> None:

    from uploaded_pdf_loader import extract_uploaded_pdf

    pdf_path = "data/peca_2016.pdf"

    pages = extract_uploaded_pdf(
        pdf_path
    )

    chunks = chunk_uploaded_pdf(
        pages,
        source_name="peca_2016.pdf",
    )

    print("=" * 50)
    print("UPLOADED PDF CHUNKING TEST")
    print("=" * 50)

    print("Pages:", len(pages))
    print("Total chunks:", len(chunks))

    if chunks:

        print("\nFirst chunk metadata:")
        print("-" * 50)

        print("Source:", chunks[0]["source"])
        print("Page:", chunks[0]["page_number"])
        print("Chunk ID:", chunks[0]["chunk_id"])

        print("\nFirst chunk text:")
        print("-" * 50)

        print(chunks[0]["text"][:500])


if __name__ == "__main__":
    main()