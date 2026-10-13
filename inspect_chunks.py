import json
from pathlib import Path


CHUNKS_PATH = Path("data/law_chunks.json")


def load_chunks(file_path: Path) -> list[dict]:
    """Load legal chunks from a JSON file."""
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """Display sample legal chunks for verification."""
    chunks = load_chunks(CHUNKS_PATH)

    print("Total chunks:", len(chunks))
    print("=" * 70)

    for index, chunk in enumerate(chunks[:5], start=1):
        print(f"\nCHUNK {index}")
        print("-" * 70)
        print("Section:", chunk["section_number"])
        print("Heading:", chunk["heading"])
        print("Text:")
        print(chunk["text"][:800])
        print()


if __name__ == "__main__":
    main()