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
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class EventExtraction(BaseModel):
    """First LLM call: Extract basic event information.

    Attributes:
        description: Raw description of the event
        is_calendar_event: Whether this text describes a calendar event
        confidence_score: Confidence score between 0 and 1
    """
    description: str = Field(description="Raw description of the event")
    is_calendar_event: bool = Field(
        description="Whether this text describes a calendar event"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")


class EventDetails(BaseModel):
    """Second LLM call: Parse specific event details.

    Attributes:
        name: Name of the event
        date: Date and time of the event in ISO 8601 format
        duration_minutes: Expected duration in minutes
        participants: List of participants
    """
    name: str = Field(description="Name of the event")
    date: str = Field(
        description="Date and time of the event. Use ISO 8601 format."
    )
    duration_minutes: int = Field(description="Expected duration in minutes")
    participants: List[str] = Field(description="List of participants")

    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate that the date is in ISO 8601 format.

        Args:
            v: Date string to validate

        Returns:
            The validated date string

        Raises:
            ValueError: If date format is invalid
        """
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError("Date must be in ISO 8601 format")


class EventConfirmation(BaseModel):
    """Third LLM call: Generate confirmation message.

    Attributes:
        confirmation_message: Natural language confirmation message
        calendar_link: Optional generated calendar link
    """
    confirmation_message: str = Field(
        description="Natural language confirmation message"
    )
    calendar_link: Optional[str] = Field(
        description="Generated calendar link if applicable"
    )


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
            model: Azure OpenAI model deployment name
            temperature: Controls randomness in responses (0-1)
            max_tokens: Maximum tokens in response
            max_retries: Maximum number of retries for API calls
            initial_retry_delay: Initial delay between retries in seconds
            max_retry_delay: Maximum delay between retries in seconds

        Raises:
            ValueError: If Azure OpenAI credentials are not set
        """
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True
    )
    def extract_event_info(self, user_input: str) -> EventExtraction:
        """First LLM call to determine if input is a calendar event.

        Args:
            user_input: Raw user input text

        Returns:
            EventExtraction object with validation results

        Raises:
            ValueError: If the input is empty or invalid
            Exception: If the API call fails after all retries
        """
        if not user_input or not user_input.strip():
            raise ValueError("Input text cannot be empty")

        logger.info("Starting event extraction analysis")
        logger.debug(f"Input text: {user_input}")

        try:
            today = datetime.now()
            date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"{date_context} Analyze if the text describes a calendar event."
                    },
                    {"role": "user", "content": user_input}
                ],
                response_format=EventExtraction
            )
            result = completion.choices[0].message.parsed
            logger.info(
                f"Extraction complete - Is calendar event: {result.is_calendar_event}, "
                f"Confidence: {result.confidence_score:.2f}"
            )
            return result
        except Exception as e:
            logger.error(f"Error in event extraction: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True
    )
    def parse_event_details(self, description: str) -> EventDetails:
        """Second LLM call to extract specific event details.

        Args:
            description: Raw event description

        Returns:
            EventDetails object with parsed information

        Raises:
            ValueError: If the description is empty or invalid
            Exception: If the API call fails after all retries
        """
        if not description or not description.strip():
            raise ValueError("Event description cannot be empty")

        logger.info("Starting event details parsing")

        try:
            today = datetime.now()
            date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"{date_context} Extract detailed event information. "
                                   "When dates reference relative dates like 'next Tuesday', "
                                   "use this current date as reference."
                    },
                    {"role": "user", "content": description}
                ],
                response_format=EventDetails
            )
            result = completion.choices[0].message.parsed
            logger.info(
                f"Parsed event details - Name: {result.name}, Date: {result.date}, "
                f"Duration: {result.duration_minutes}min"
            )
            logger.debug(f"Participants: {', '.join(result.participants)}")
            return result
        except Exception as e:
            logger.error(f"Error in event parsing: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True
    )
    def generate_confirmation(self, event_details: EventDetails) -> EventConfirmation:
        """Third LLM call to generate a confirmation message.

        Args:
            event_details: Parsed event details

        Returns:
            EventConfirmation object with confirmation message

        Raises:
            ValueError: If event_details is None
            Exception: If the API call fails after all retries
        """
        if not event_details:
            raise ValueError("Event details cannot be None")

        logger.info("Generating confirmation message")

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate a natural confirmation message for the event. "
                                   "Include all relevant details in a clear format, making sure to:\n"
                                   "1. List all participant email addresses and names (if available) in Paricipants: line\n"
                                   "2. Use the following format for the Participants: Participants:" 
                                   " john@example.com; \"Sarah\" sarah@example.com"
                                   "3. Use 12-hour time format with AM/PM (e.g., 2:00 PM instead of 14:00)\n"
                                   "4. Do not use - or * for formatting of field names"
                                   "5. Sign off with your name; Calendar Assistant"
                    },
                    {"role": "user", "content": json.dumps(event_details.model_dump())}
                ],
                response_format=EventConfirmation
            )
            result = completion.choices[0].message.parsed
            logger.info("Confirmation message generated successfully")
            return result
        except Exception as e:
            logger.error(f"Error in confirmation generation: {str(e)}")
            raise

    def process_calendar_request(self, user_input: str) -> Tuple[bool, str]:
        """Process a calendar request through the entire prompt chain.

        Args:
            user_input: Raw user input string

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Step 1: Extract and validate event info
            event_info = self.extract_event_info(user_input)
            if not event_info.is_calendar_event:
                return False, f"Not a valid calendar request (confidence: {event_info.confidence_score:.2f})"

            # Step 2: Parse event details
            event_details = self.parse_event_details(event_info.description)

            # Step 3: Generate confirmation
            confirmation = self.generate_confirmation(event_details)

            return True, confirmation.confirmation_message

        except Exception as e:
            error_msg = f"Error processing calendar request: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


def main() -> None:
    """Example usage of the prompt chaining pattern."""
    # Initialize the processor
    processor = PromptChainProcessor(
        model="gpt-4o",  # Use your deployed model name
        temperature=0.7,
        max_tokens=1000
    )

    # Example calendar requests
    requests = [
        "Schedule a team meeting next Tuesday at 2pm for 1 hour with john@example.com "
        "and sarah@example.com to discuss the Q2 roadmap.",
        "Can you order pizza for dinner?",  # Non-calendar request
        "Set up a quick sync tomorrow morning at 9am with the dev team "
        "(dev-team@company.com) for 30 minutes."
    ]

    # Process each request
    for i, request in enumerate(requests, 1):
        print(f"\nProcessing Request #{i}:")
        print(f"Input: {request}")
        success, message = processor.process_calendar_request(request)
        print("Result: " + ("Success" if success else "Failed"))
        print(f"Message: {message}\n")
        print("-" * 80)


if __name__ == "__main__":
    main()
