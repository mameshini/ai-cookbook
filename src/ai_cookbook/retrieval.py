"""Module for implementing knowledge base retrieval using Azure OpenAI API."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------------------------
# Step 1: Define the knowledge base retrieval tool
# --------------------------------------------------------------

class KBRecord(BaseModel):
    """Knowledge base record structure.
    
    Attributes:
        id: Unique identifier for the record
        question: The question this record answers
        answer: The answer to the question
    """
    id: int
    question: str
    answer: str


class KBResponse(BaseModel):
    """Structured format for knowledge base response.
    
    Attributes:
        answer: The answer to the user's question
        source: The record id where the answer was found
    """
    answer: str = Field(description="The answer to the user's question.")
    source: int = Field(description="The record id of the answer.")


def search_kb(question: str) -> Dict:
    """Get the answer to the user's question from the knowledge base.
    
    Args:
        question: The user's question to search for
        
    Returns:
        The knowledge base records as a dictionary
        
    Raises:
        FileNotFoundError: If the knowledge base file doesn't exist
        json.JSONDecodeError: If the knowledge base file is invalid JSON
    """
    kb_path = Path(__file__).parent / "kb.json"
    with open(kb_path, "r") as f:
        return json.load(f)


def get_kb_answer(text: str) -> KBResponse:
    """Get an answer from the knowledge base using Azure OpenAI function calling.
    
    Args:
        text: The user's question
        
    Returns:
        A KBResponse object containing the answer and source
        
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
        api_version="2024-10-21",
        azure_endpoint=endpoint
    )

    # Define the tools (functions) available to the model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_kb",
                "description": "Get the answer to the user's question from the knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                    },
                    "required": ["question"],
                },
            },
        }
    ]

    # Step 1: Call model with the user's question
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store. "
                      "If the question cannot be answered from the knowledge base, politely explain that you can only "
                      "answer questions about store policies and services."
        },
        {"role": "user", "content": text},
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    # Step 2: Check if the model wants to use the knowledge base
    tool_calls = completion.choices[0].message.tool_calls
    
    if tool_calls:
        # Model wants to search the knowledge base
        for tool_call in tool_calls:
            if tool_call.function.name == "search_kb":
                args = json.loads(tool_call.function.arguments)
                result = search_kb(**args)
                
                # Step 3: Add function result to messages
                messages.append(completion.choices[0].message)
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
                )
    else:
        # Question is not about the knowledge base
        return KBResponse(
            answer="I can only answer questions about our store's policies and services. "
                   "For questions about weather, current events, or other topics, please consult appropriate sources.",
            source=0  # 0 indicates answer not from knowledge base
        )

    # Step 4: Get final response from model
    completion_2 = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        response_format=KBResponse
    )
    
    return completion_2.choices[0].message.parsed


def main() -> None:
    """Example usage of knowledge base retrieval."""
    try:
        # Example 1: Question about return policy (in KB)
        print("\nExample 1: Question about return policy")
        response1 = get_kb_answer(
            text="What is your return policy?"
        )
        print("Knowledge Base Response:")
        print(f"Answer: {response1.answer}")
        print(f"Source: Record #{response1.source}")

        # Example 2: Question about shipping (in KB)
        print("\nExample 2: Question about shipping")
        response2 = get_kb_answer(
            text="Do you ship to other countries?"
        )
        print("Knowledge Base Response:")
        print(f"Answer: {response2.answer}")
        print(f"Source: Record #{response2.source}")

        # Example 3: Question not in KB
        print("\nExample 3: Question not in knowledge base")
        response3 = get_kb_answer(
            text="What is the weather like today?"
        )
        print("Knowledge Base Response:")
        print(f"Answer: {response3.answer}")
        print(f"Source: Record #{response3.source}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
