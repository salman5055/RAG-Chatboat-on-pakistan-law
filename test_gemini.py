import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


def main() -> None:
    """Test Gemini through LangChain."""

    print("Creating LangChain Gemini model...")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0,
        max_retries=3,
    )

    print("Sending request...")

    response = llm.invoke(
        "Say only: LangChain Gemini is working."
    )

    print("Response received!")
    print("-" * 50)
    print(response.content)


if __name__ == "__main__":
    main()