"""Module for handling chat completions via Azure OpenAI API."""

from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

def basic_chat(
    prompt: str,
    model: str = "gpt-4o",
    system_message: str = "You're a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """Basic chat completion function using Azure OpenAI.

    Args:
        prompt: The user's input prompt
        model: The model deployment name in Azure OpenAI (default: 'gpt-4o')
        system_message: The system message to set context
        temperature: Controls randomness (0-1)
        max_tokens: Maximum number of tokens to generate

    Returns:
        The model's response as a string

    Raises:
        ValueError: If Azure OpenAI credentials are not properly set
        Exception: If the API call fails
    """
    # Initialize Azure OpenAI client
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZ_OPENAI_ENDPOINT")

    if not api_key or not endpoint:
        raise ValueError("Azure OpenAI credentials must be set in environment variables")

    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-02-15-preview",
        azure_endpoint=endpoint
    )

    # Prepare messages
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    # Create completion request parameters
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    # Create chat completion
    completion = client.chat.completions.create(**params)

    # Return response
    return completion.choices[0].message.content or ""


def main() -> None:
    """Example usage of basic chat completion."""
    try:
        # Example: Generate a poem about empathy
        response = basic_chat(
            prompt="Write a poem about empathy. Make it nice and simple, a little homorous.",
            model="gpt-4o",  # Make sure this matches your deployment name
            system_message="You're a creative poet.",
            temperature=0.8
        )
        print("Response:")
        print(response)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
