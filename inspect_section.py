import json
from pathlib import Path


CHUNKS_PATH = Path("data/law_chunks.json")


def main():
    """Inspect PECA sections related to spamming."""

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    print("=" * 70)
    print("PECA SECTION INSPECTION")
    print("=" * 70)

    for chunk in chunks:
        section = str(chunk.get("section_number", ""))

        if section in {"22", "22A", "22B", "22C"}:
            print()
            print("-" * 70)
            print("Section:", section)
            print("Heading:", chunk.get("heading"))
            print("Text:")
            print(chunk.get("text", "")[:1500])


if __name__ == "__main__":
    main()