import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print("Gemini API key loaded successfully!")
    print("API key is NOT displayed.")
else:
    print("ERROR: GEMINI_API_KEY was not found.")