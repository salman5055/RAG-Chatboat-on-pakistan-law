import os
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from schemas import ChatResponse, Source


VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


def load_vectorstore() -> Chroma:
    """Load the existing PECA Chroma vector store."""

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings,
    )


def create_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini language model."""

    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0,
    )


def create_prompt() -> ChatPromptTemplate:
    """Create a grounded legal RAG prompt."""

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a legal information assistant for the
Prevention of Electronic Crimes Act, 2016 (PECA), Pakistan.

Answer the user's question ONLY using the provided context.

Rules:
1. Do not invent legal provisions.
2. Do not use information outside the provided context.
3. If the context does not contain enough information,
   clearly say that the available PECA document does not
   provide enough information to answer.
4. Keep the answer factual and concise.
5. Mention the relevant PECA section when appropriate.

Context:
{context}
""",
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )


def format_context(documents) -> str:
    """Convert retrieved documents into prompt context."""

    context_parts = []

    for document in documents:
        section = document.metadata.get("section_number", "Unknown")
        heading = document.metadata.get("heading", "Unknown")

        context_parts.append(
            f"Section {section}: {heading}\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


def answer_question(question: str) -> ChatResponse:
    """Retrieve PECA context and generate a grounded answer."""

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    start_retrieval = time.perf_counter()

    documents = retriever.invoke(question)

    retrieval_time = time.perf_counter() - start_retrieval

    context = format_context(documents)

    prompt = create_prompt()
    llm = create_llm()

    messages = prompt.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    start_generation = time.perf_counter()

    response = llm.invoke(messages)

    generation_time = time.perf_counter() - start_generation

    total_time = retrieval_time + generation_time

    sources = [
        Source(
            section_number=document.metadata.get(
                "section_number", "Unknown"
            ),
            heading=document.metadata.get(
                "heading", "Unknown"
            ),
        )
        for document in documents
    ]

    return ChatResponse(
        answer=response.content,
        sources=sources,
        is_in_scope=True,
        retrieval_time=retrieval_time,
        generation_time=generation_time,
        total_time=total_time,
    )


def main() -> None:
    """Test the complete retrieval + generation pipeline."""

    question = "What is cyber stalking under PECA?"

    print("Question:")
    print(question)
    print("=" * 60)

    result = answer_question(question)

    print("Answer:")
    print(result.answer)
    print("=" * 60)

    print("Sources:")

    for source in result.sources:
        print(
            f"Section {source.section_number}: "
            f"{source.heading}"
        )

    print("=" * 60)
    print(f"Retrieval time: {result.retrieval_time:.4f} seconds")
    print(f"Generation time: {result.generation_time:.4f} seconds")
    print(f"Total time: {result.total_time:.4f} seconds")


if __name__ == "__main__":
    main()