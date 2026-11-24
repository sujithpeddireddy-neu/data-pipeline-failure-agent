"""List model IDs available to your API key."""
import os
from google import genai

if __name__ == "__main__":
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for model in client.models.list():
        if "generateContent" in (model.supported_actions or []):
            print(model.name)
