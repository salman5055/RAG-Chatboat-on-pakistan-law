import re
from typing import List

from embeddings import get_embedding_model


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    a: list[float],
    b: list[float]
) -> float:

    if not a or not b:
        return 0.0

    dot = sum(
        x * y
        for x, y in zip(a, b)
    )

    mag_a = sum(
        x * x
        for x in a
    ) ** 0.5

    mag_b = sum(
        y * y
        for y in b
    ) ** 0.5

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (
        mag_a * mag_b
    )


# ============================================================
# SENTENCE SPLITTER
# ============================================================

def split_sentences(text: str) -> list[str]:

    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# GET DOCUMENT TEXT
# ============================================================

def get_document_text(document) -> str:

    return getattr(
        document,
        "page_content",
        ""
    ).strip()


# ============================================================
# GET PDF CONTEXT
# ============================================================

def get_context(documents: List) -> str:

    context_parts = []

    for document in documents:

        text = get_document_text(
            document
        )

        if text:
            context_parts.append(text)

    return "\n".join(
        context_parts
    )


# ============================================================
# KEYWORD SUPPORT
# ============================================================

def keyword_support(
    answer_sentence: str,
    context: str
) -> float:

    answer_words = set(
        normalize_text(
            answer_sentence
        ).split()
    )

    context_words = set(
        normalize_text(
            context
        ).split()
    )

    if not answer_words:
        return 0.0

    matched_words = (
        answer_words
        .intersection(
            context_words
        )
    )

    return (
        len(matched_words)
        /
        len(answer_words)
    )


# ============================================================
# SEMANTIC SUPPORT
# ============================================================

def semantic_support(
    answer_sentence: str,
    documents: List,
    embeddings
) -> float:

    if not answer_sentence.strip():
        return 0.0

    context_chunks = []

    for document in documents:

        text = get_document_text(
            document
        )

        if text:
            context_chunks.append(
                text
            )

    if not context_chunks:
        return 0.0

    # Embed answer only once
    answer_vector = embeddings.embed_query(
        answer_sentence
    )

    # Compare with PDF chunks
    best_score = 0.0

    for chunk in context_chunks:

        chunk_vector = embeddings.embed_query(
            chunk
        )

        similarity = cosine_similarity(
            answer_vector,
            chunk_vector
        )

        if similarity > best_score:
            best_score = similarity

    return max(
        0.0,
        min(
            1.0,
            best_score
        )
    )


# ============================================================
# MAIN ACCURACY CALCULATION
# ============================================================

def calculate_grounded_accuracy(
    answer: str,
    documents: List
) -> float:

    if not answer or not answer.strip():
        return 0.0

    if not documents:
        return 0.0

    context = get_context(
        documents
    )

    if not context:
        return 0.0

    sentences = split_sentences(
        answer
    )

    if not sentences:
        return 0.0

    # Load embedding model ONCE
    embeddings = get_embedding_model()

    sentence_scores = []

    for sentence in sentences:

        semantic_score = semantic_support(
            sentence,
            documents,
            embeddings
        )

        lexical_score = keyword_support(
            sentence,
            context
        )

        # Semantic meaning has higher weight
        final_sentence_score = (
            semantic_score * 0.80
            +
            lexical_score * 0.20
        )

        sentence_scores.append(
            final_sentence_score
        )

    if not sentence_scores:
        return 0.0

    final_score = (
        sum(sentence_scores)
        /
        len(sentence_scores)
    )

    accuracy = final_score * 100

    return round(
        max(
            0.0,
            min(
                100.0,
                accuracy
            )
        ),
        2
    )


# ============================================================
# EVALUATE ANSWER
# ============================================================

def evaluate_answer(
    question: str,
    answer: str,
    documents: List
) -> dict:

    accuracy = calculate_grounded_accuracy(
        answer,
        documents
    )

    if accuracy >= 95.0:

        status = "🟢 PASS"

    else:

        status = "🟠 BELOW 95%"


    return {

        "question": question,

        "accuracy": accuracy,

        "required_accuracy": 95.0,

        "status": status,

        "retrieved_documents": len(
            documents
        )
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    class TestDocument:

        def __init__(
            self,
            text
        ):
            self.page_content = text


    test_context = """
    Laravel is a PHP framework
    used for backend web development.
    """

    test_answer = """
    Laravel is a PHP framework
    used for backend web development.
    """

    documents = [
        TestDocument(
            test_context
        )
    ]

    result = evaluate_answer(
        question="What is Laravel?",
        answer=test_answer,
        documents=documents
    )

    print("=" * 60)
    print(
        "DYNAMIC PDF ACCURACY EVALUATION"
    )
    print("=" * 60)

    print(
        "Question:",
        result["question"]
    )

    print(
        "Accuracy:",
        f'{result["accuracy"]:.2f}%'
    )

    print(
        "Required Accuracy:",
        f'{result["required_accuracy"]:.2f}%'
    )

    print(
        "Status:",
        result["status"]
    )

    print(
        "Retrieved Documents:",
        result["retrieved_documents"]
    )

    print("=" * 60)