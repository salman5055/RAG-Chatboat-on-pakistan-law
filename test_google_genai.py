import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


def main() -> None:
    """Test Gemini directly using Google's GenAI SDK."""

    print("Creating Gemini client...")

    client = genai.Client(
        api_key=api_key
    )

    print("Sending test request...")

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Say only: Gemini connection is working."
    )

    print("Response received!")
    print("-" * 50)
    print(response.text)


if __name__ == "__main__":
    main()