import re
import json
from pathlib import Path

from pypdf import PdfReader


PDF_PATH = Path("data/peca_2016.pdf")
OUTPUT_PATH = Path("data/law_chunks.json")


def extract_pages(pdf_path: Path) -> list[str]:
    """Extract text from every PDF page."""
    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return pages


def extract_law_pages(pages: list[str]) -> str:
    """Combine pages containing the actual law text.

    The first five pages are the table of contents.
    The actual Act starts from page 6.
    """
    law_pages = pages[5:]

    return "\n".join(law_pages)


def split_into_sections(text: str) -> list[dict]:
    """Split the law text into numbered legal sections."""
    pattern = r"(?m)^\s*(\d+[A-Z]?)\.\s+(.+?)(?=\n)"

    matches = list(re.finditer(pattern, text))

    sections = []

    for index, match in enumerate(matches):
        section_number = match.group(1)
        heading = match.group(2).strip()

        start = match.start()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )

        section_text = text[start:end].strip()

        sections.append(
            {
                "section_number": section_number,
                "heading": heading,
                "text": section_text,
            }
        )

    return sections


def create_chunks(
    sections: list[dict],
    max_characters: int = 3000,
) -> list[dict]:
    """Create RAG chunks from legal sections.

    Large legal sections are divided into smaller overlapping chunks.
    """
    chunks = []

    for section in sections:
        text = section["text"]

        if len(text) <= max_characters:
            chunks.append(
                {
                    "section_number": section["section_number"],
                    "heading": section["heading"],
                    "text": text,
                }
            )

            continue

        start = 0
        chunk_number = 1

        while start < len(text):
            end = start + max_characters

            chunk_text = text[start:end].strip()

            chunks.append(
                {
                    "section_number": section["section_number"],
                    "heading": section["heading"],
                    "chunk_number": chunk_number,
                    "text": chunk_text,
                }
            )

            start = end - 300
            chunk_number += 1

    return chunks


def save_chunks(chunks: list[dict], output_path: Path) -> None:
    """Save chunks to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)


def main() -> None:
    """Run the complete legal chunking pipeline."""
    pages = extract_pages(PDF_PATH)

    print("Total PDF pages:", len(pages))

    law_text = extract_law_pages(pages)

    print("Law text characters:", len(law_text))

    sections = split_into_sections(law_text)

    print("Detected legal sections:", len(sections))

    chunks = create_chunks(sections)

    print("Total RAG chunks:", len(chunks))

    save_chunks(chunks, OUTPUT_PATH)

    print("Chunks saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()