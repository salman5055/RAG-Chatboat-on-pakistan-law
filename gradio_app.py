import time
from pathlib import Path

import gradio as gr
from langchain_chroma import Chroma

from uploaded_vectorstore import create_uploaded_vectorstore
from uploaded_pdf_generator import generate_answer
from relevance_scorer import calculate_grounded_accuracy
from embeddings import get_embedding_model


# ==========================================================
# CONFIGURATION
# ==========================================================

VECTORSTORE_ROOT = Path("vectorstores/uploaded")

ACCURACY_THRESHOLD = 95.0

TOP_K = 5


# ==========================================================
# PDF UPLOAD
# ==========================================================

def upload_pdf(pdf_file):

    if pdf_file is None:
        return (
            "❌ No PDF selected.",
            None,
        )

    try:

        pdf_path = Path(pdf_file)

        print("=" * 60)
        print("PROCESSING UPLOADED PDF")
        print("=" * 60)
        print("PDF:", pdf_path.name)

        # Create vector store
        create_uploaded_vectorstore(
            str(pdf_path)
        )

        # Vector store path
        store_path = (
            VECTORSTORE_ROOT
            / pdf_path.stem
        )

        print(
            "Vector store:",
            store_path,
        )

        return (
            f"✅ PDF processed successfully\n\n"
            f"**Document:** {pdf_path.name}\n\n"
            f"**Status:** Ready for questions",
            str(store_path),
        )

    except Exception as error:

        print("=" * 60)
        print("PDF PROCESSING ERROR")
        print("=" * 60)
        print(error)

        return (
            f"❌ Error processing PDF\n\n"
            f"{error}",
            None,
        )


# ==========================================================
# ASK QUESTION
# ==========================================================

def ask_question(
    question,
    store_path,
):

    # ------------------------------------------------------
    # VALIDATE QUESTION
    # ------------------------------------------------------

    if not question or not question.strip():

        return (
            "❌ Please enter a question.",
            "",
            "No evaluation available.",
        )

    # ------------------------------------------------------
    # VALIDATE PDF
    # ------------------------------------------------------

    if not store_path:

        return (
            "❌ Please upload and process a PDF first.",
            "",
            "No evaluation available.",
        )

    try:

        # ==================================================
        # TOTAL TIMER
        # ==================================================

        total_start = time.perf_counter()

        print("\n")
        print("=" * 60)
        print("QUESTION")
        print("=" * 60)
        print(question)

        # ==================================================
        # LOAD EMBEDDINGS
        # ==================================================

        embeddings = get_embedding_model()

        # ==================================================
        # LOAD VECTOR STORE
        # ==================================================

        vectorstore = Chroma(
            persist_directory=store_path,
            embedding_function=embeddings,
        )

        # ==================================================
        # RETRIEVAL
        # ==================================================

        retrieval_start = time.perf_counter()

        documents = vectorstore.similarity_search(
            question,
            k=TOP_K,
        )

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        print(
            "Retrieved documents:",
            len(documents),
        )

        # ==================================================
        # GENERATE ANSWER
        # ==================================================

        generation_start = time.perf_counter()

        answer = generate_answer(
            question,
            documents,
        )

        generation_time = (
            time.perf_counter()
            - generation_start
        )

        # ==================================================
        # GROUNDED ACCURACY
        # ==================================================

        accuracy_start = time.perf_counter()

        grounded_accuracy = (
            calculate_grounded_accuracy(
                answer,
                documents,
            )
        )

        accuracy_time = (
            time.perf_counter()
            - accuracy_start
        )

        # ==================================================
        # TOTAL RESPONSE TIME
        # ==================================================

        total_time = (
            time.perf_counter()
            - total_start
        )

        # ==================================================
        # SOURCES
        # ==================================================

        source_lines = []

        for index, document in enumerate(
            documents,
            start=1,
        ):

            source = document.metadata.get(
                "source",
                "Unknown",
            )

            page = document.metadata.get(
                "page_number",
                "Unknown",
            )

            source_lines.append(
                f"{index}. **{source}** — Page {page}"
            )

        if source_lines:

            sources = "\n".join(
                source_lines
            )

        else:

            sources = (
                "No source information available."
            )

        # ==================================================
        # ACCURACY STATUS
        # ==================================================

        if grounded_accuracy >= ACCURACY_THRESHOLD:

            score_status = "🟢 PASS"

        else:

            score_status = (
                "🟠 BELOW 95%"
            )

        # ==================================================
        # EVALUATION METRICS
        # ==================================================

        metrics = f"""
### 📊 Answer Evaluation

**Grounded Answer Accuracy:** `{grounded_accuracy:.2f}%`

**Required Accuracy:** `{ACCURACY_THRESHOLD:.2f}%`

**Status:** {score_status}

> The grounded accuracy score measures how strongly the generated answer is supported by the retrieved content from the uploaded PDF.

---

### ⏱️ Response Time

| Metric | Time |
|---|---:|
| Retrieval | `{retrieval_time:.4f}s` |
| Generation | `{generation_time:.4f}s` |
| Accuracy Evaluation | `{accuracy_time:.4f}s` |
| **Total Response Time** | **`{total_time:.4f}s`** |
"""

        # ==================================================
        # TERMINAL OUTPUT
        # ==================================================

        print("=" * 60)
        print("RESULT")
        print("=" * 60)

        print(
            "Grounded Accuracy:",
            f"{grounded_accuracy:.2f}%",
        )

        print(
            "Required Accuracy:",
            f"{ACCURACY_THRESHOLD:.2f}%",
        )

        print(
            "Status:",
            score_status,
        )

        print(
            "Retrieval time:",
            f"{retrieval_time:.4f}s",
        )

        print(
            "Generation time:",
            f"{generation_time:.4f}s",
        )

        print(
            "Accuracy evaluation:",
            f"{accuracy_time:.4f}s",
        )

        print(
            "Total response time:",
            f"{total_time:.4f}s",
        )

        print("=" * 60)

        # ==================================================
        # RETURN TO GRADIO
        # ==================================================

        return (
            answer,
            sources,
            metrics,
        )

    except Exception as error:

        print("=" * 60)
        print("QUESTION ANSWERING ERROR")
        print("=" * 60)
        print(error)

        return (
            f"❌ Error:\n\n{error}",
            "",
            "Evaluation unavailable.",
        )


# ==========================================================
# GRADIO UI
# ==========================================================

with gr.Blocks(
    title="LawDoc AI — PDF RAG Assistant"
) as demo:

    # ======================================================
    # HEADER
    # ======================================================

    gr.Markdown(
        """
# ⚖️ LawDoc AI

### Intelligent PDF Question Answering System

Upload **any PDF**, process it, and ask questions based on
the document using **Retrieval-Augmented Generation (RAG)**.

The system retrieves relevant content from the uploaded PDF,
generates an answer, shows the source pages, and calculates
a grounded answer-support score and response time.
"""
    )

    # ======================================================
    # STATE
    # ======================================================

    store_path = gr.State(None)

    # ======================================================
    # MAIN LAYOUT
    # ======================================================

    with gr.Row():

        # --------------------------------------------------
        # LEFT COLUMN
        # --------------------------------------------------

        with gr.Column(
            scale=1
        ):

            gr.Markdown(
                "### 📄 Upload Document"
            )

            pdf_file = gr.File(
                label="Select PDF",
                file_types=[".pdf"],
                type="filepath",
            )

            upload_button = gr.Button(
                "Process PDF",
                variant="primary",
            )

            upload_status = gr.Markdown(
                label="Document Status",
                value="No PDF processed yet.",
            )

        # --------------------------------------------------
        # RIGHT COLUMN
        # --------------------------------------------------

        with gr.Column(
            scale=2
        ):

            gr.Markdown(
                "### 💬 Ask Questions"
            )

            question = gr.Textbox(
                label="Your Question",
                placeholder=(
                    "Ask something about the uploaded PDF..."
                ),
                lines=3,
            )

            ask_button = gr.Button(
                "Ask Question",
                variant="primary",
            )

            gr.Markdown(
                "### 🤖 Generated Answer"
            )

            answer = gr.Markdown(
                value="Answer will appear here.",
            )

            gr.Markdown(
                "### 📚 Sources"
            )

            sources = gr.Markdown(
                value="Sources will appear here.",
            )

            gr.Markdown(
                "### 📊 Evaluation & Performance"
            )

            metrics = gr.Markdown(
                value="Evaluation will appear here.",
            )

    # ======================================================
    # PDF PROCESSING EVENT
    # ======================================================

    upload_button.click(
        fn=upload_pdf,
        inputs=pdf_file,
        outputs=[
            upload_status,
            store_path,
        ],
    )

    # ======================================================
    # QUESTION ANSWERING EVENT
    # ======================================================

    ask_button.click(
        fn=ask_question,
        inputs=[
            question,
            store_path,
        ],
        outputs=[
            answer,
            sources,
            metrics,
        ],
    )


# ==========================================================
# START APPLICATION
# ==========================================================

if __name__ == "__main__":

    demo.launch()