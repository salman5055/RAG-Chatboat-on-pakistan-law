import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

GEMINI_MODEL = "gemini-3.5-flash-lite"


def generate_answer(question: str, documents: list) -> str:

    context_parts = []

    for index, document in enumerate(documents, start=1):

        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page_number", "Unknown")
        section = document.metadata.get("section_number", "Unknown")

        context_parts.append(
            f"""
CONTEXT {index}

Source: {source}
Page: {page}
Section: {section}

Content:
{document.page_content}
"""
        )

    context = "\n\n-----------------------------\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a grounded document question-answering assistant.

Your ONLY source of information is the provided document context.

IMPORTANT RULES:

1. Read ALL provided context carefully before answering.
2. Answer the user's question using information found in the context.
3. Do NOT use outside knowledge.
4. Do NOT invent facts.
5. If the context contains even partial information relevant to the question,
   use that information to answer.
6. Do NOT say that information is unavailable if the provided context
   contains the answer.
7. Include important details from the context, including:
   - definitions
   - conditions
   - actions
   - exceptions
   - punishments
   - penalties
   - amounts
   - time periods
8. Preserve the meaning of the original document.
9. If the question asks for a definition, provide the definition clearly.
10. If the question asks for punishment, provide the punishment clearly.
11. Mention the relevant page when possible.
12. Keep the answer concise but complete.

Only if the answer is genuinely absent from ALL provided context, say:

"The uploaded document does not contain enough information to answer this question."

DOCUMENT CONTEXT:

{context}
"""
            ),
            ("human", "{question}"),
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
            "context": context,
            "question": question,
        }
    )

    response = llm.invoke(messages)

    if isinstance(response.content, str):
        return response.content

    if isinstance(response.content, list):

        text_parts = []

        for item in response.content:

            if isinstance(item, dict):

                text = item.get("text")

                if text:
                    text_parts.append(text)

        return "\n".join(text_parts)

    return str(response.content)