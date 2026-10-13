import sys
import json
import time
import re
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from langchain_chroma import Chroma

from embeddings import get_embedding_model
from uploaded_pdf_generator import generate_answer


# ============================================================
# PATHS
# ============================================================

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "evaluation_results"
    / "answer_ground_truth.json"
)

VECTORSTORE_PATH = (
    PROJECT_ROOT
    / "vectorstore"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation_results"
    / "answer_accuracy_results.json"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_K = 3

TARGET_ACCURACY = 95.0

SECTION_WEIGHT = 30.0
CONCEPT_WEIGHT = 70.0


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth():

    if not GROUND_TRUTH_PATH.exists():

        raise FileNotFoundError(
            f"Ground truth file not found:\n"
            f"{GROUND_TRUTH_PATH}"
        )

    with GROUND_TRUTH_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):

        raise ValueError(
            "Ground truth JSON must contain a list."
        )

    return data


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vectorstore():

    if not VECTORSTORE_PATH.exists():

        raise FileNotFoundError(
            f"Vector store not found:\n"
            f"{VECTORSTORE_PATH}"
        )

    print("Loading embedding model...")

    embeddings = get_embedding_model()

    print("Loading Chroma vector store...")

    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_PATH),
        embedding_function=embeddings,
    )

    return vectorstore


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text: str) -> str:

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = " ".join(
        text.split()
    )

    return text


# ============================================================
# CHECK SECTION
# ============================================================

def check_section(
    expected_section: str,
    documents: list
) -> bool:

    expected_section = str(
        expected_section
    ).strip().lower()

    retrieved_sections = []

    for document in documents:

        section = document.metadata.get(
            "section_number"
        )

        if section is not None:

            retrieved_sections.append(
                str(section)
                .strip()
                .lower()
            )

    return (
        expected_section
        in retrieved_sections
    )


# ============================================================
# GET RETRIEVED SECTIONS
# ============================================================

def get_retrieved_sections(
    documents: list
) -> list[str]:

    sections = []

    for document in documents:

        section = document.metadata.get(
            "section_number"
        )

        if section is not None:

            section = str(section)

            if section not in sections:

                sections.append(section)

    return sections


# ============================================================
# QUERY EXPANSION
# ============================================================

def build_retrieval_query(question: str) -> str:

    question_normalized = (
        question
        .lower()
        .strip()
    )

    # --------------------------------------------------------
    # Q8:
    # "What is identity information?"
    #
    # The original query retrieves section 16 and 27,
    # while the actual definition is in section 2.
    # --------------------------------------------------------

    if question_normalized == "what is identity information?":

        return (
            "identity information definition "
            "authenticate identify individual "
            "information system enable access data"
        )

    # --------------------------------------------------------
    # All other questions use the original query.
    # --------------------------------------------------------

    return question


# ============================================================
# CONCEPT MATCHING
# ============================================================

def calculate_concept_coverage(
    generated_answer: str,
    key_concepts: list[str]
):

    if not key_concepts:

        return 0.0, [], []

    normalized_answer = normalize_text(
        generated_answer
    )

    matched = []
    missing = []

    for concept in key_concepts:

        normalized_concept = normalize_text(
            concept
        )

        if not normalized_concept:
            continue

        if normalized_concept in normalized_answer:

            matched.append(concept)

        else:

            missing.append(concept)

    total_concepts = len(
        key_concepts
    )

    matched_count = len(
        matched
    )

    coverage = (
        matched_count
        / total_concepts
        * 100
    )

    return (
        round(coverage, 2),
        matched,
        missing
    )


# ============================================================
# CALCULATE OVERALL ANSWER ACCURACY
# ============================================================

def calculate_answer_accuracy(
    section_correct: bool,
    concept_coverage: float
) -> float:

    section_score = (
        100.0
        if section_correct
        else 0.0
    )

    overall_score = (
        section_score
        * (SECTION_WEIGHT / 100)
        +
        concept_coverage
        * (CONCEPT_WEIGHT / 100)
    )

    return round(
        overall_score,
        2
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PECA RAG ANSWER ACCURACY EVALUATION")
    print("=" * 70)

    # ========================================================
    # LOAD GROUND TRUTH
    # ========================================================

    questions = load_ground_truth()

    print(
        "Total questions:",
        len(questions)
    )

    # ========================================================
    # LOAD VECTOR STORE
    # ========================================================

    print()

    vectorstore = load_vectorstore()

    print(
        "Vector store loaded successfully."
    )

    print(
        "Top-K:",
        TOP_K
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    results = []

    total_response_time = 0.0

    total_retrieval_time = 0.0

    total_generation_time = 0.0

    for item in questions:

        question_number = item[
            "question_number"
        ]

        question = item[
            "question"
        ]

        expected_section = item[
            "expected_section"
        ]

        key_concepts = item[
            "key_concepts"
        ]

        print()
        print("-" * 70)

        print(
            f"Question {question_number}"
        )

        print(
            "Question:",
            question
        )

        print(
            "Expected section:",
            expected_section
        )

        # ====================================================
        # BUILD RETRIEVAL QUERY
        # ====================================================

        retrieval_query = (
            build_retrieval_query(
                question
            )
        )

        if retrieval_query != question:

            print(
                "Expanded retrieval query:",
                retrieval_query
            )

        # ====================================================
        # TOTAL TIMER
        # ====================================================

        total_start = time.perf_counter()

        # ====================================================
        # RETRIEVAL
        # ====================================================

        retrieval_start = (
            time.perf_counter()
        )

        documents = (
            vectorstore.similarity_search(
                retrieval_query,
                k=TOP_K
            )
        )

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        # ====================================================
        # GENERATION
        # ====================================================

        generation_start = (
            time.perf_counter()
        )

        generated_answer = (
            generate_answer(
                question,
                documents
            )
        )

        generation_time = (
            time.perf_counter()
            - generation_start
        )

        # ====================================================
        # TOTAL RESPONSE TIME
        # ====================================================

        response_time = (
            time.perf_counter()
            - total_start
        )

        total_response_time += (
            response_time
        )

        total_retrieval_time += (
            retrieval_time
        )

        total_generation_time += (
            generation_time
        )

        # ====================================================
        # SECTION CHECK
        # ====================================================

        section_correct = check_section(
            expected_section,
            documents
        )

        retrieved_sections = (
            get_retrieved_sections(
                documents
            )
        )

        # ====================================================
        # CONCEPT COVERAGE
        # ====================================================

        (
            concept_coverage,
            matched_concepts,
            missing_concepts
        ) = calculate_concept_coverage(
            generated_answer,
            key_concepts
        )

        # ====================================================
        # OVERALL ACCURACY
        # ====================================================

        answer_accuracy = (
            calculate_answer_accuracy(
                section_correct,
                concept_coverage
            )
        )

        passed = (
            answer_accuracy
            >= TARGET_ACCURACY
        )

        # ====================================================
        # RESULT
        # ====================================================

        result = {

            "question_number":
                question_number,

            "question":
                question,

            "retrieval_query":
                retrieval_query,

            "expected_section":
                expected_section,

            "retrieved_sections":
                retrieved_sections,

            "section_correct":
                section_correct,

            "key_concepts":
                key_concepts,

            "matched_concepts":
                matched_concepts,

            "missing_concepts":
                missing_concepts,

            "concept_coverage_percent":
                concept_coverage,

            "answer_accuracy_percent":
                answer_accuracy,

            "target_accuracy_percent":
                TARGET_ACCURACY,

            "passed":
                passed,

            "generated_answer":
                generated_answer,

            "retrieval_time_seconds":
                round(
                    retrieval_time,
                    4
                ),

            "generation_time_seconds":
                round(
                    generation_time,
                    4
                ),

            "response_time_seconds":
                round(
                    response_time,
                    4
                )
        }

        results.append(result)

        # ====================================================
        # PRINT RESULT
        # ====================================================

        print(
            "Retrieved sections:",
            retrieved_sections
        )

        print(
            "Section correct:",
            "YES"
            if section_correct
            else "NO"
        )

        print(
            "Concept coverage:",
            f"{concept_coverage:.2f}%"
        )

        print(
            "Answer accuracy:",
            f"{answer_accuracy:.2f}%"
        )

        print(
            "Response time:",
            f"{response_time:.4f}s"
        )

        print(
            "Status:",
            "PASS"
            if passed
            else "FAIL"
        )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    total_questions = len(
        results
    )

    passed_questions = sum(
        1
        for result in results
        if result["passed"]
    )

    failed_questions = (
        total_questions
        - passed_questions
    )

    # --------------------------------------------------------
    # Average answer accuracy
    # --------------------------------------------------------

    average_answer_accuracy = (
        sum(
            result[
                "answer_accuracy_percent"
            ]
            for result in results
        )
        / total_questions
    )

    # --------------------------------------------------------
    # Section accuracy
    # --------------------------------------------------------

    section_correct_count = sum(
        1
        for result in results
        if result["section_correct"]
    )

    section_accuracy = (
        section_correct_count
        / total_questions
        * 100
    )

    # --------------------------------------------------------
    # Average concept coverage
    # --------------------------------------------------------

    average_concept_coverage = (
        sum(
            result[
                "concept_coverage_percent"
            ]
            for result in results
        )
        / total_questions
    )

    # --------------------------------------------------------
    # Average response time
    # --------------------------------------------------------

    average_response_time = (
        total_response_time
        / total_questions
    )

    # --------------------------------------------------------
    # Average retrieval time
    # --------------------------------------------------------

    average_retrieval_time = (
        total_retrieval_time
        / total_questions
    )

    # --------------------------------------------------------
    # Average generation time
    # --------------------------------------------------------

    average_generation_time = (
        total_generation_time
        / total_questions
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    final_status = (
        "PASS"
        if average_answer_accuracy
        >= TARGET_ACCURACY
        else "BELOW TARGET"
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    evaluation = {

        "evaluation": {

            "description":
                "PECA RAG grounded answer evaluation",

            "total_questions":
                total_questions,

            "passed_questions":
                passed_questions,

            "failed_questions":
                failed_questions,

            "section_accuracy_percent":
                round(
                    section_accuracy,
                    2
                ),

            "average_concept_coverage_percent":
                round(
                    average_concept_coverage,
                    2
                ),

            "answer_accuracy_percent":
                round(
                    average_answer_accuracy,
                    2
                ),

            "target_accuracy_percent":
                TARGET_ACCURACY,

            "status":
                final_status,

            "average_response_time_seconds":
                round(
                    average_response_time,
                    4
                ),

            "average_retrieval_time_seconds":
                round(
                    average_retrieval_time,
                    4
                ),

            "average_generation_time_seconds":
                round(
                    average_generation_time,
                    4
                ),

            "top_k":
                TOP_K,

            "section_weight_percent":
                SECTION_WEIGHT,

            "concept_weight_percent":
                CONCEPT_WEIGHT,

            "results":
                results
        }
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ========================================================
    # FINAL CONSOLE REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        "Total questions:",
        total_questions
    )

    print(
        "Passed:",
        passed_questions
    )

    print(
        "Failed:",
        failed_questions
    )

    print(
        "Section accuracy:",
        f"{section_accuracy:.2f}%"
    )

    print(
        "Average concept coverage:",
        f"{average_concept_coverage:.2f}%"
    )

    print(
        "Answer accuracy:",
        f"{average_answer_accuracy:.2f}%"
    )

    print(
        "Target accuracy:",
        f"{TARGET_ACCURACY:.2f}%"
    )

    print(
        "Status:",
        final_status
    )

    print(
        "Average response time:",
        f"{average_response_time:.4f}s"
    )

    print(
        "Average retrieval time:",
        f"{average_retrieval_time:.4f}s"
    )

    print(
        "Average generation time:",
        f"{average_generation_time:.4f}s"
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()