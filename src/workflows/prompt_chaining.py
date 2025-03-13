"""Module implementing a prompt-chaining pattern using Azure OpenAI API.

This module demonstrates the prompt chaining pattern for a calendar assistant
that processes requests through multiple stages: extraction, parsing, and confirmation.

Features:
    - Type-safe with full Python type hints and Pydantic models
    - Automatic retry mechanism with configurable retries and delays
    - Comprehensive error handling and logging
    - Response metadata tracking
    - Flexible system prompts for each chain step
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class ValidationResponse(BaseModel):
    """Pydantic model for validation response.

    Attributes:
        is_valid: Whether the input is a valid calendar request
        confidence: Confidence score between 0 and 1
        reasoning: Explanation for the decision
    """
    is_valid: bool = Field(..., description="Whether the input is a valid calendar request")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: str = Field(..., min_length=1, description="Explanation for the decision")


class CalendarRequest(BaseModel):
    """Pydantic model for calendar request details.

    Attributes:
        date: The date of the event in YYYY-MM-DD format
        time: The time of the event in HH:MM format
        duration: Duration of the event in minutes
        participants: List of participant email addresses
        title: Title of the event
        description: Optional description of the event
    """
    date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$', description="Date in YYYY-MM-DD format")
    time: str = Field(..., pattern=r'^\d{2}:\d{2}$', description="Time in HH:MM format")
    duration: int = Field(..., gt=0, description="Duration in minutes")
    participants: List[str] = Field(..., min_length=1, description="List of participant email addresses")
    title: str = Field(..., min_length=1, description="Title of the event")
    description: Optional[str] = Field(None, description="Optional description of the event")

    @field_validator('participants')
    @classmethod
    def validate_email_addresses(cls, v: List[str]) -> List[str]:
        """Validate that all participants have email addresses.

        Args:
            v: List of participant email addresses

        Returns:
            The validated list of email addresses

        Raises:
            ValueError: If any email address is invalid
        """
        for email in v:
            if '@' not in email:
                raise ValueError(f"Invalid email address: {email}")
        return v


class PromptChainProcessor:
    """Class implementing the prompt chaining pattern for calendar requests.

    This class demonstrates a 3-step prompt chain:
    1. Extract & Validate: Determines if input is a valid calendar request
    2. Parse Details: Extracts structured calendar information
    3. Generate Confirmation: Creates user-friendly confirmation

    Features:
    - Automatic retries with exponential backoff
    - Comprehensive error handling and logging
    - Response metadata tracking
    - Type-safe with Pydantic models
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 10.0
    ) -> None:
        """Initialize the PromptChainProcessor.

        Args:
            model: The model deployment name in Azure OpenAI (default: 'gpt-4o')
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            max_retries: Maximum number of retries for API calls
            initial_retry_delay: Initial delay between retries in seconds
            max_retry_delay: Maximum delay between retries in seconds
        
        Raises:
            ValueError: If Azure OpenAI credentials are not properly set
        """
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZ_OPENAI_ENDPOINT")

        if not api_key or not endpoint:
            raise ValueError("Azure OpenAI credentials must be set in environment variables")

        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-10-21",
            azure_endpoint=endpoint
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay

        # Configure retry decorator
        self._retry_decorator = retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(
                multiplier=initial_retry_delay,
                max=max_retry_delay
            ),
            reraise=True
        )

    def _extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON from a response that might be wrapped in markdown code blocks.

        Args:
            response: Response string from the API

        Returns:
            Parsed JSON dictionary

        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        logger.debug(f"Attempting to extract JSON from response: {response}")

        # Try parsing as pure JSON first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.debug("Failed to parse response as pure JSON")

        # Try extracting JSON from markdown code block
        if '```json' in response and '```' in response:
            try:
                json_str = response.split('```json')[1].split('```')[0].strip()
                return json.loads(json_str)
            except (IndexError, json.JSONDecodeError):
                logger.debug("Failed to extract JSON from markdown code block")

        # Try finding any JSON-like structure
        try:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                return json.loads(response[start:end + 1])
        except json.JSONDecodeError:
            logger.debug("Failed to extract JSON from response content")

        error_msg = "Could not extract valid JSON from response"
        logger.error(f"{error_msg}: {response}")
        raise json.JSONDecodeError(error_msg, response, 0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True
    )
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Make a call to Azure OpenAI API with automatic retries.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            The model's response as a string

        Raises:
            Exception: If the API call fails after all retries
        """
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if self.max_tokens is not None:
                params["max_tokens"] = self.max_tokens

            logger.debug(f"Making API call with params: {params}")
            start_time = time.time()

            completion = self.client.chat.completions.create(**params)
            response = completion.choices[0].message.content or ""

            elapsed_time = time.time() - start_time
            logger.info(f"API call completed in {elapsed_time:.2f} seconds")
            logger.debug(f"Received response: {response}")

            return response
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise

    def extract_and_validate(self, user_input: str) -> Tuple[bool, float, str]:
        """Step 1: Extract and validate if the input is a calendar request.

        Args:
            user_input: The user's input text

        Returns:
            Tuple containing:
            - Boolean indicating if input is a valid calendar request
            - Confidence score (0-1)
            - Reasoning for the decision
        """
        system_message = """You are a calendar request validator. 
        Analyze the input and determine if it's a valid calendar request.
        Respond in JSON format with three fields:
        {
            "is_valid": true/false,
            "confidence": 0.0-1.0,
            "reasoning": "explanation"
        }
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ]
        
        response = self._call_llm(messages)
        result = self._extract_json_from_response(response)
        return (result["is_valid"], result["confidence"], result["reasoning"])

    def parse_details(self, user_input: str) -> CalendarRequest:
        """Step 2: Parse the calendar request details.

        Args:
            user_input: The validated calendar request text

        Returns:
            CalendarRequest object containing structured event details

        Raises:
            ValueError: If the response cannot be parsed or is invalid
        """
        system_message = """You are a calendar request parser.
        Extract calendar event details from the input and respond in JSON format with:
        {
            "date": "YYYY-MM-DD",
            "time": "HH:MM",
            "duration": minutes_as_integer,
            "participants": ["email1", "email2"],
            "title": "event title",
            "description": "optional description"
        }
        """
        
        try:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ]
            
            response = self._call_llm(messages)
            result_dict = self._extract_json_from_response(response)
            
            # Validate and create CalendarRequest using Pydantic model
            calendar_request = CalendarRequest(**result_dict)
            logger.info(
                f"Parsed calendar request: {calendar_request.date} at {calendar_request.time}, "
                f"{len(calendar_request.participants)} participants"
            )
            
            return calendar_request
        except Exception as e:
            logger.error(f"Failed to parse calendar details: {str(e)}")
            raise ValueError(f"Failed to parse calendar details: {str(e)}")

    def generate_confirmation(self, calendar_request: CalendarRequest) -> str:
        """Step 3: Generate a user-friendly confirmation message.

        Args:
            calendar_request: The parsed calendar request details

        Returns:
            A natural language confirmation message

        Raises:
            ValueError: If the confirmation message cannot be generated
        """
        try:
            system_message = """You are a helpful calendar assistant.
            Generate a friendly confirmation message for the calendar event.
            Include all relevant details in a clear, concise format.
            When formatting dates, use both the ISO format (YYYY-MM-DD) and human-readable format.
            """
            
            event_details = f"""
            Event: {calendar_request.title}
            Date: {calendar_request.date}
            Time: {calendar_request.time}
            Duration: {calendar_request.duration} minutes
            Participants: {', '.join(calendar_request.participants)}
            Description: {calendar_request.description or 'N/A'}
            """
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Generate confirmation for:\n{event_details}"}
            ]
            
            confirmation = self._call_llm(messages)
            logger.info(f"Generated confirmation for event on {calendar_request.date}")
            
            return confirmation
        except Exception as e:
            logger.error(f"Failed to generate confirmation: {str(e)}")
            raise ValueError(f"Failed to generate confirmation: {str(e)}")

    def process_request(self, user_input: str) -> str:
        """Process a calendar request through the entire prompt chain.

        Args:
            user_input: The user's calendar request text

        Returns:
            A confirmation message if successful, or an error message if validation fails

        Note:
            This method will never return None. It will always return either a success
            or error message to provide clear feedback to the user.
        """
        try:
            # Step 1: Extract and Validate
            logger.info("Step 1: Validating calendar request")
            is_valid, confidence, reasoning = self.extract_and_validate(user_input)
            if not is_valid or confidence < 0.7:
                return f"Sorry, I couldn't process that as a calendar request. {reasoning}"

            # Step 2: Parse Details
            logger.info("Step 2: Parsing calendar details")
            calendar_details = self.parse_details(user_input)

            # Step 3: Generate Confirmation
            logger.info("Step 3: Generating confirmation message")
            return self.generate_confirmation(calendar_details)

        except ValueError as e:
            error_msg = str(e)
            logger.error(f"Failed to process request: {error_msg}")
            return f"Sorry, I encountered an error while processing your request: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Unexpected error while processing request: {error_msg}")
            return "Sorry, an unexpected error occurred while processing your request. Please try again."


def main() -> None:
    """Example usage of the prompt chaining pattern."""
    processor = PromptChainProcessor(
        model="gpt-4o",  # Make sure this matches your deployment name
        temperature=0.7
    )

    # Example calendar request
    user_input = """Schedule a team meeting for next Tuesday at 2 PM.
    It will be a 1-hour discussion with john@example.com and sarah@example.com
    about the Q2 planning."""

    try:
        result = processor.process_request(user_input)
        print("Response:")
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
