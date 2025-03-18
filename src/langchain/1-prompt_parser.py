"""Module for handling prompt parsing and text transformations using Azure OpenAI."""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime

from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

class CustomerServiceResponse(BaseModel):
    """Schema for customer service response."""
    tone: str = Field(description="The overall tone of the response")
    subject: str = Field(description="The main subject or issue being addressed")
    response: str = Field(description="The actual response text")
    next_steps: List[str] = Field(description="List of recommended next steps")

def get_azure_client() -> AzureOpenAI:
    """Initialize and return Azure OpenAI client.
    
    Returns:
        AzureOpenAI: Configured Azure OpenAI client
        
    Raises:
        ValueError: If Azure OpenAI credentials are not properly set
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key or not endpoint:
        raise ValueError("Azure OpenAI credentials must be set in environment variables")

    return AzureOpenAI(
        api_key=api_key,
        api_version="2024-10-21",
        azure_endpoint=endpoint
    )

def get_completion(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0,
    response_format: Optional[Dict[str, Any]] = None
) -> str:
    """Get completion from Azure OpenAI.
    
    Args:
        prompt: The input prompt
        model: The model deployment name in Azure OpenAI
        temperature: Controls randomness (0-1)
        response_format: Optional format specification for structured output
        
    Returns:
        str: The model's response
    """
    client = get_azure_client()
    
    messages = [{"role": "user", "content": prompt}]
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    
    if response_format:
        params["response_format"] = response_format
    
    completion = client.chat.completions.create(**params)
    return completion.choices[0].message.content or ""

def translate_text(text: str, style: str) -> str:
    """Translate text into specified style.
    
    Args:
        text: Text to translate
        style: Target style for translation
        
    Returns:
        str: Translated text
    """
    prompt = f"""Translate the text \
    that is delimited by triple backticks 
    into a style that is {style}.
    text: ```{text}```
    """
    return get_completion(prompt)

def parse_customer_response(email: str) -> CustomerServiceResponse:
    """Parse customer email and generate structured response.
    
    Args:
        email: Customer email content
        
    Returns:
        CustomerServiceResponse: Structured response with tone, subject, response text, and next steps
    """
    # Define response schemas
    response_schemas = [
        ResponseSchema(name="tone", description="The tone of the customer's email (e.g., angry, frustrated, polite)"),
        ResponseSchema(name="subject", description="The main subject or issue from the email"),
        ResponseSchema(name="response", description="A polite and professional response to the customer"),
        ResponseSchema(name="next_steps", description="List of recommended next steps or actions")
    ]
    
    # Create the parser
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()
    
    # Create the prompt template
    template = """Analyze the customer email below and provide a structured response.
    
    Customer Email: {email}
    
    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["email"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    # Get the formatted prompt
    formatted_prompt = prompt.format(email=email)
    
    # Get completion with JSON response format
    response = get_completion(
        formatted_prompt,
        response_format={"type": "json_object"}
    )
    
    # Parse the response into our Pydantic model
    parsed_response = parser.parse(response)
    return CustomerServiceResponse(**parsed_response)

def main() -> None:
    """Example usage of prompt parsing and text transformation."""
    try:
        # Example 1: Simple style translation
        customer_email = """
        Arrr, I be fuming that me blender lid 
        flew off and splattered me kitchen walls 
        with smoothie! And to make matters worse,
        the warranty don't cover the cost of 
        cleaning up me kitchen. I need yer help 
        right now, matey!
        """
        
        style = "American English in a calm and respectful tone"
        translated = translate_text(customer_email, style)
        print("\nTranslated Email:")
        print(translated)
        
        # Example 2: Structured response parsing
        response = parse_customer_response(customer_email)
        print("\nStructured Response:")
        print(f"Tone: {response.tone}")
        print(f"Subject: {response.subject}")
        print(f"Response: {response.response}")
        print("\nNext Steps:")
        for step in response.next_steps:
            print(f"- {step}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
