from pydantic import BaseModel, Field


class Source(BaseModel):
    """Source information from the PECA document."""

    section_number: str
    heading: str


class ChatRequest(BaseModel):
    """User question sent to the chatbot."""

    question: str = Field(
        min_length=1,
        description="User's question about PECA",
    )


class ChatResponse(BaseModel):
    """Final response returned by the RAG chatbot."""

    answer: str

    sources: list[Source] = Field(
        default_factory=list
    )

    is_in_scope: bool

    retrieval_time: float = Field(
        description="Retrieval time in seconds"
    )

    generation_time: float = Field(
        description="Generation time in seconds"
    )

    total_time: float = Field(
        description="Total processing time in seconds"
    )