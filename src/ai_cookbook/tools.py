"""Module for implementing function calling using Azure OpenAI API."""

import json
import os
from typing import Dict

import requests
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------------------------
# Step 1: Define the tool (function) that we want to call
# --------------------------------------------------------------

def get_weather(latitude: float, longitude: float) -> Dict:
    """Get current temperature for provided coordinates.
    
    Args:
        latitude: The latitude coordinate
        longitude: The longitude coordinate
        
    Returns:
        A dictionary containing current weather data including temperature in Fahrenheit
    """
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}"
        f"&current=temperature_2m&temperature_unit=fahrenheit"
    )
    data = response.json()
    return data["current"]


# --------------------------------------------------------------
# Step 2: Define the response format using Pydantic
# --------------------------------------------------------------

class WeatherResponse(BaseModel):
    """Structured format for weather response.
    
    Attributes:
        temperature: The current temperature in Fahrenheit
        response: A natural language response to the user's question
    """
    temperature: float = Field(
        description="The current temperature in Fahrenheit for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )


def get_location_weather(text: str) -> WeatherResponse:
    """Get weather information for a location using Azure OpenAI function calling.
    
    Args:
        text: The user's question about weather in a location
        
    Returns:
        A WeatherResponse object containing temperature and natural language response
        
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
                "name": "get_weather",
                "description": "Get current temperature for provided coordinates in Fahrenheit.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"},
                    },
                    "required": ["latitude", "longitude"],
                },
            },
        }
    ]

    # Step 1: Call model with the user's question
    messages = [
        {"role": "system", "content": "You are a helpful weather assistant. "},
        {"role": "user", "content": text},
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    # Step 2: Execute the weather function with the model's chosen coordinates
    for tool_call in completion.choices[0].message.tool_calls:
        if tool_call.function.name == "get_weather":
            args = json.loads(tool_call.function.arguments)
            result = get_weather(**args)
            
            # Step 3: Add function result to messages
            messages.append(completion.choices[0].message)
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
            )

    # Step 4: Get final response from model
    completion_2 = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        response_format=WeatherResponse
    )
    
    return completion_2.choices[0].message.parsed


def main() -> None:
    """Example usage of weather function calling."""
    try:
        # Example: Get weather for San Diego
        weather = get_location_weather(
            text="What's the weather like in San Diego today?"
        )
        print("\nWeather Information:")
        print(f"Temperature: {weather.temperature}°F")
        print(f"Response: {weather.response}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
