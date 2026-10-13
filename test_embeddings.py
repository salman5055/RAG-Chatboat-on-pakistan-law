from langchain_huggingface import HuggingFaceEmbeddings


def main() -> None:
    """Test the local Hugging Face embedding model."""

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    text = "Section 24 deals with cyber stalking."

    vector = embeddings.embed_query(text)

    print("Embedding model loaded successfully!")
    print("-" * 50)
    print("Input text:", text)
    print("Vector length:", len(vector))
    print("First 5 values:", vector[:5])


if __name__ == "__main__":
    main()