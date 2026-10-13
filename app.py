from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from graph import run_question
from schemas import ChatRequest, ChatResponse


app = FastAPI(
    title="PECA RAG Chatbot",
    description=(
        "RAG chatbot for the Prevention of Electronic Crimes Act, 2016"
    ),
    version="1.0.0",
)


# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "message": "PECA RAG Chatbot API is running"
    }


@app.get("/health")
def health():
    """API health status."""
    return {
        "status": "healthy"
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
    """Process a PECA question through the RAG pipeline."""

    response = run_question(
        request.question
    )

    return response