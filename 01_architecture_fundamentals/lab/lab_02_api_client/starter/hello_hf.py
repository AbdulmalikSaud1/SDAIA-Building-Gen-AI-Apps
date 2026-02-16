"""
Lab 2 — Steps 1 & 2: Environment Setup + First API Call

Step 1: Read through the get_api_token() function — understand why we
        never hardcode tokens.
Step 2: Complete the TODO at the bottom to make your first API call.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_api_token():
    """Retrieve API token with validation."""
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise EnvironmentError(
            "HUGGINGFACE_API_TOKEN not found. "
            "Create a .env file with your token or set the environment variable."
        )
    # Accept both HuggingFace (hf_) and OpenRouter (sk-or-v1-) tokens
    if not (token.startswith("hf_") or token.startswith("sk-or-v1-")):
        raise ValueError(
            "Invalid token format. Expected HuggingFace (hf_) or OpenRouter (sk-or-v1-) token."
        )
    return token


# --- Configuration ---
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "openrouter/auto"  # Auto-select best available free model
TOKEN = get_api_token()
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}


# =====================================================================
# TODO (Step 2): Make your first API call
#
# Modified to work with OpenRouter API (chat completions format)
# =====================================================================

prompt = "Explain what a vector database is in one paragraph:"

# OpenRouter uses chat completion format
payload = {
    "model": MODEL_ID,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 150,
    "temperature": 0.7
}

print(f"Sending request to {MODEL_ID}...")
response = requests.post(API_URL, headers=HEADERS, json=payload)
response.raise_for_status()
result = response.json()

print("\nGenerated Text:")
print(result["choices"][0]["message"]["content"])
