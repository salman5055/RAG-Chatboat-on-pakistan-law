import os
import time
from typing import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from schemas import ChatResponse, Source


# ============================================================
# CONFIGURATION
# ============================================================

VECTORSTORE_PATH = "vectorstore"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

GEMINI_MODEL = "gemini-3.5-flash-lite"


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env"
    )


# ============================================================
# LANGGRAPH STATE
# ============================================================

class RAGState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    question: str

    is_in_scope: bool
    scope_message: str

    documents: list
    sources: list[Source]
    context: str

    answer: str

    retrieval_time: float
    generation_time: float
    total_time: float

    response: ChatResponse


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vectorstore() -> Chroma:
    """Load the existing PECA Chroma vector store."""

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings,
    )

    return vectorstore


# ============================================================
# NODE 1 — SCOPE CHECK
# ============================================================

def scope_check(state: RAGState) -> RAGState:
    """
    Check whether the question is related to PECA.

    This is a deterministic rule-based guardrail.
    """

    question = state["question"].lower()

    peca_keywords = [
        "peca",
        "prevention of electronic crimes",
        "cyber crime",
        "cybercrime",
        "cyber stalking",
        "cyber terrorism",
        "electronic crime",
        "electronic crimes",
        "electronic fraud",
        "electronic forgery",
        "unauthorized access",
        "unauthorized copying",
        "information system",
        "information systems",
        "online harassment",
        "identity information",
        "child pornography",
        "minor",
        "spoofing",
        "spamming",
        "cyberbullying",
    ]

    is_in_scope = any(
        keyword in question
        for keyword in peca_keywords
    )

    if is_in_scope:

        return {
            "is_in_scope": True,
            "scope_message": (
                "Question is within PECA scope."
            ),
        }

    return {
        "is_in_scope": False,
        "scope_message": (
            "This chatbot only answers questions "
            "related to the Prevention of Electronic "
            "Crimes Act, 2016."
        ),
        "answer": (
            "This question is outside the scope of "
            "the PECA chatbot."
        ),
        "sources": [],
        "retrieval_time": 0.0,
        "generation_time": 0.0,
        "total_time": 0.0,
    }


# ============================================================
# ROUTING AFTER SCOPE CHECK
# ============================================================

def route_after_scope(state: RAGState) -> str:
    """Route question based on scope."""

    if state["is_in_scope"]:
        return "retrieve"

    return "response"


# ============================================================
# NODE 2 — RETRIEVAL
# ============================================================

def retrieve_documents(state: RAGState) -> RAGState:
    """Retrieve relevant PECA chunks from Chroma."""

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    start_retrieval = time.perf_counter()

    documents = retriever.invoke(
        state["question"]
    )

    retrieval_time = (
        time.perf_counter()
        - start_retrieval
    )

    sources = []

    for document in documents:

        source = Source(
            section_number=document.metadata.get(
                "section_number",
                "Unknown",
            ),
            heading=document.metadata.get(
                "heading",
                "Unknown",
            ),
        )

        sources.append(source)

    context = format_context(
        documents
    )

    return {
        "documents": documents,
        "sources": sources,
        "context": context,
        "retrieval_time": retrieval_time,
    }


# ============================================================
# FORMAT RETRIEVED DOCUMENTS
# ============================================================

def format_context(documents) -> str:
    """
    Convert retrieved documents into context
    for the Gemini model.
    """

    context_parts = []

    for document in documents:

        section = document.metadata.get(
            "section_number",
            "Unknown",
        )

        heading = document.metadata.get(
            "heading",
            "Unknown",
        )

        content = document.page_content

        context_parts.append(
            f"Section {section}: {heading}\n"
            f"{content}"
        )

    return "\n\n---\n\n".join(
        context_parts
    )


# ============================================================
# NODE 3 — GENERATION
# ============================================================

def generate_answer(state: RAGState) -> RAGState:
    """Generate a grounded answer using Gemini."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a legal information assistant for the
Prevention of Electronic Crimes Act, 2016 (PECA), Pakistan.

Answer the user's question ONLY using the provided context.

Rules:

1. Use only the provided PECA context.
2. Do not invent legal provisions.
3. Do not use outside knowledge.
4. If the context does not contain enough information,
   clearly state that the provided PECA document does
   not contain enough information.
5. Mention the relevant PECA section when appropriate.
6. Keep the answer factual and concise.
7. Do not provide legal advice.
8. Do not make assumptions beyond the provided text.

PECA CONTEXT:

{context}
""",
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=API_KEY,
        temperature=0,
        max_retries=1,
    )

    messages = prompt.invoke(
        {
            "context": state["context"],
            "question": state["question"],
        }
    )

    start_generation = time.perf_counter()

    response = llm.invoke(
        messages
    )

    generation_time = (
        time.perf_counter()
        - start_generation
    )

    # Handle normal string content
    if isinstance(
        response.content,
        str,
    ):

        answer = response.content

    # Handle list-based Gemini content
    elif isinstance(
        response.content,
        list,
    ):

        text_parts = []

        for item in response.content:

            if isinstance(
                item,
                dict,
            ):

                text = item.get(
                    "text"
                )

                if text:
                    text_parts.append(
                        text
                    )

        answer = "\n".join(
            text_parts
        )

    else:

        answer = str(
            response.content
        )

    total_time = (
        state["retrieval_time"]
        + generation_time
    )

    return {
        "answer": answer,
        "generation_time": generation_time,
        "total_time": total_time,
    }


# ============================================================
# NODE 4 — CREATE PYDANTIC RESPONSE
# ============================================================

def create_response(
    state: RAGState,
) -> RAGState:
    """Create the final structured response."""

    response = ChatResponse(
        answer=state["answer"],
        sources=state.get(
            "sources",
            [],
        ),
        is_in_scope=state["is_in_scope"],
        retrieval_time=state.get(
            "retrieval_time",
            0.0,
        ),
        generation_time=state.get(
            "generation_time",
            0.0,
        ),
        total_time=state.get(
            "total_time",
            0.0,
        ),
    )

    return {
        "response": response
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph():
    """Build the complete PECA RAG LangGraph."""

    graph = StateGraph(
        RAGState
    )

    # Add nodes
    graph.add_node(
        "scope_check",
        scope_check,
    )

    graph.add_node(
        "retrieve",
        retrieve_documents,
    )

    graph.add_node(
        "generate",
        generate_answer,
    )

    graph.add_node(
        "create_response",
        create_response,
    )

    # Entry point
    graph.set_entry_point(
        "scope_check"
    )

    # Scope routing
    graph.add_conditional_edges(
        "scope_check",
        route_after_scope,
        {
            "retrieve": "retrieve",
            "response": "create_response",
        },
    )

    # Retrieval → Generation
    graph.add_edge(
        "retrieve",
        "generate",
    )

    # Generation → Response
    graph.add_edge(
        "generate",
        "create_response",
    )

    # Response → End
    graph.add_edge(
        "create_response",
        END,
    )

    return graph.compile()


# ============================================================
# TEST FUNCTION
# ============================================================

def run_question(
    question: str,
) -> ChatResponse:
    """Run one question through the RAG graph."""

    app = build_graph()

    result = app.invoke(
        {
            "question": question
        }
    )

    return result["response"]


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Test the complete RAG pipeline."""

    question = (
        "What is cyber stalking under PECA?"
    )

    print("=" * 60)
    print("PECA RAG CHATBOT")
    print("=" * 60)

    print("Question:")
    print(question)

    print("=" * 60)

    try:

        response = run_question(
            question
        )

        print(
            "In scope:",
            response.is_in_scope,
        )

        print("-" * 60)

        print("Answer:")
        print(
            response.answer
        )

        print("-" * 60)

        print("Sources:")

        for source in response.sources:

            print(
                f"Section "
                f"{source.section_number}: "
                f"{source.heading}"
            )

        print("-" * 60)

        print(
            "Retrieval time:",
            f"{response.retrieval_time:.4f}",
            "seconds",
        )

        print(
            "Generation time:",
            f"{response.generation_time:.4f}",
            "seconds",
        )

        print(
            "Total time:",
            f"{response.total_time:.4f}",
            "seconds",
        )

        print("=" * 60)

    except Exception as error:

        print("-" * 60)
        print("ERROR")
        print("-" * 60)

        print(
            type(error).__name__
        )

        print(error)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()