from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create and return the Hugging Face embedding model.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
    )

    return embeddings


def main() -> None:
    """
    Test the embedding model.
    """

    embeddings = get_embedding_model()

    text = "What is cyber stalking under PECA?"

    vector = embeddings.embed_query(text)

    print("=" * 50)
    print("EMBEDDING MODEL TEST")
    print("=" * 50)

    print("Model:", EMBEDDING_MODEL)
    print("Input:", text)
    print("Vector length:", len(vector))

    print("-" * 50)
    print("First 5 values:")
    print(vector[:5])


if __name__ == "__main__":
    main()