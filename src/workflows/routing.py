"""Module implementing a routing pattern using Azure OpenAI API.

This module demonstrates the routing pattern for a calendar assistant that can:
1. Determine the type of calendar request (new event, modify event, other)
2. Extract relevant details based on request type
3. Generate appropriate responses with proper formatting

Features:
- Type-safe implementation using Pydantic models
- Automatic retries with exponential backoff
- Comprehensive error handling and logging
- Response metadata tracking
"""

import json
import logging
import os
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from openai import AzureOpenAI
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logger = logging.getLogger(__name__)


class RequestType(str, Enum):
    """Enumeration of supported calendar request types."""
    NEW_EVENT = "new_event"
    MODIFY_EVENT = "modify_event"
    OTHER = "other"


class RequestClassification(BaseModel):
    """Router LLM call: Determine the type of calendar request.

    Attributes:
        request_type: Type of calendar request being made
        confidence_score: Confidence score between 0 and 1
        description: Cleaned description of the request
    """
    request_type: RequestType = Field(description="Type of calendar request being made")
    confidence_score: float = Field(description="Confidence score between 0 and 1")
    description: str = Field(description="Cleaned description of the request")


class NewEventDetails(BaseModel):
    """Details for creating a new calendar event.

    Attributes:
        name: Name/title of the event
        date: Date and time in ISO 8601 format
        duration_minutes: Duration in minutes
        participants: List of participant email addresses
    """
    name: str = Field(description="Name of the event")
    date: str = Field(description="Date and time of the event in ISO 8601 format")
    duration_minutes: int = Field(description="Duration in minutes")
    participants: List[str] = Field(description="List of participants")

    @field_validator('date')
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
            raise ValueError("Date must be in ISO 8601 format (e.g., 2025-03-14T14:00:00)")


class Change(BaseModel):
    """Details for changing a specific field in an existing event.

    Attributes:
        field: Field to change (e.g., date, time, duration)
        new_value: New value for the field
    """
    field: str = Field(description="Field to change")
    new_value: str = Field(description="New value for the field")


class ModifyEventDetails(BaseModel):
    """Details for modifying an existing calendar event.

    Attributes:
        event_identifier: Description to identify the existing event
        changes: List of changes to make
        participants_to_add: New participants to add
        participants_to_remove: Participants to remove
    """
    event_identifier: str = Field(description="Description to identify the existing event")
    changes: List[Change] = Field(description="List of changes to make")
    participants_to_add: List[str] = Field(description="New participants to add")
    participants_to_remove: List[str] = Field(description="Participants to remove")


class CalendarResponse(BaseModel):
    """Final response format for calendar operations.

    Attributes:
        success: Whether the operation was successful
        message: User-friendly response message
        calendar_link: Optional calendar link
    """
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="User-friendly response message")
    calendar_link: Optional[str] = Field(description="Calendar link if applicable")


class RequestRouter:
    """Class implementing the routing pattern for calendar requests.

    This class demonstrates a multi-step routing pattern:
    1. Classify: Determine the type of calendar request
    2. Extract: Get relevant details based on request type
    3. Process: Handle the request appropriately
    4. Respond: Generate a user-friendly response

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
        """Initialize the RequestRouter.

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
    def classify_request(self, user_input: str) -> RequestClassification:
        """First LLM call to determine the type of calendar request.

        Args:
            user_input: Raw user input text

        Returns:
            RequestClassification object with request type and confidence

        Raises:
            ValueError: If the input is empty or invalid
            Exception: If the API call fails after all retries
        """
        if not user_input or not user_input.strip():
            raise ValueError("User input cannot be empty")

        logger.info("Starting request classification")
        logger.debug(f"Input text: {user_input}")

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze the text and determine if it's a request to:\n"
                                 "1. Create a new calendar event\n"
                                 "2. Modify an existing calendar event\n"
                                 "3. Something else\n"
                                 "Respond with the appropriate request type and confidence score."
                    },
                    {"role": "user", "content": user_input}
                ],
                response_format=RequestClassification
            )
            result = completion.choices[0].message.parsed
            logger.info(
                f"Classification complete - Type: {result.request_type}, "
                f"Confidence: {result.confidence_score:.2f}"
            )
            return result
        except Exception as e:
            logger.error(f"Error in request classification: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True
    )
    def extract_new_event_details(self, description: str) -> NewEventDetails:
        """Extract details for creating a new calendar event.

        Args:
            description: Cleaned description of the request

        Returns:
            NewEventDetails object with parsed information

        Raises:
            ValueError: If the description is empty or invalid
            Exception: If the API call fails after all retries
        """
        if not description or not description.strip():
            raise ValueError("Event description cannot be empty")

        logger.info("Starting new event details extraction")

        try:
            today = datetime.now()
            date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"{date_context} Extract details for creating a new calendar event. "
                                 "When dates reference relative dates like 'next Tuesday', "
                                 "use this current date as reference."
                    },
                    {"role": "user", "content": description}
                ],
                response_format=NewEventDetails
            )
            result = completion.choices[0].message.parsed
            logger.info(
                f"Extracted new event details - Name: {result.name}, "
                f"Date: {result.date}, Duration: {result.duration_minutes}min"
            )
            return result
        except Exception as e:
            logger.error(f"Error in new event details extraction: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True
    )
    def extract_modify_event_details(self, description: str) -> ModifyEventDetails:
        """Extract details for modifying an existing calendar event.

        Args:
            description: Cleaned description of the request

        Returns:
            ModifyEventDetails object with parsed information

        Raises:
            ValueError: If the description is empty or invalid
            Exception: If the API call fails after all retries
        """
        if not description or not description.strip():
            raise ValueError("Event description cannot be empty")

        logger.info("Starting modify event details extraction")

        try:
            today = datetime.now()
            date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"{date_context} Extract details for modifying an existing calendar event. "
                                 "Identify the event to modify and what changes to make."
                    },
                    {"role": "user", "content": description}
                ],
                response_format=ModifyEventDetails
            )
            result = completion.choices[0].message.parsed
            logger.info(
                f"Extracted modify event details - Event: {result.event_identifier}, "
                f"Changes: {len(result.changes)}, "
                f"Add participants: {len(result.participants_to_add)}, "
                f"Remove participants: {len(result.participants_to_remove)}"
            )
            return result
        except Exception as e:
            logger.error(f"Error in modify event details extraction: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True
    )
    def generate_response(self, success: bool, details: str) -> CalendarResponse:
        """Generate a user-friendly response message.

        Args:
            success: Whether the operation was successful
            details: Details about the operation result

        Returns:
            CalendarResponse object with response message

        Raises:
            ValueError: If details are empty
            Exception: If the API call fails after all retries
        """
        if not details or not details.strip():
            raise ValueError("Response details cannot be empty")

        logger.info("Generating response message")

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate a natural response message for the calendar operation. "
                                 "Include all relevant details in a clear format.\n"
                                 "1. Use 12-hour time format with AM/PM (e.g., 2:00 PM instead of 14:00)\n"
                                 "2. Format participant emails as 'Participants: email1@example.com; email2@example.com'\n"
                                 "3. Do not use - or * for formatting\n"
                                 "Sign off with your name; Calendar Assistant"
                    },
                    {
                        "role": "user",
                        "content": json.dumps({
                            "success": success,
                            "details": details
                        })
                    }
                ],
                response_format=CalendarResponse
            )
            result = completion.choices[0].message.parsed
            logger.info("Response message generated successfully")
            return result
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            raise

    def route_calendar_request(self, user_input: str) -> Tuple[bool, str]:
        """Route and process a calendar request through the routing pattern.

        Args:
            user_input: Raw user input string

        Returns:
            Tuple of (success: bool, message: str)

        Raises:
            ValueError: If the input is empty
        """
        if not user_input or not user_input.strip():
            raise ValueError("User input cannot be empty")

        try:
            # Step 1: Classify the request
            classification = self.classify_request(user_input)
            
            if classification.request_type == RequestType.NEW_EVENT:
                # Step 2a: Extract new event details
                details = self.extract_new_event_details(classification.description)
                participants_str = "; ".join(details.participants)
                response = self.generate_response(
                    True,
                    f"New event created: {details.name} on {details.date} "
                    f"for {details.duration_minutes} minutes. "
                    f"Participants: {participants_str}"
                )
            
            elif classification.request_type == RequestType.MODIFY_EVENT:
                # Step 2b: Extract modification details
                details = self.extract_modify_event_details(classification.description)
                changes_desc = ", ".join(f"{c.field}: {c.new_value}" for c in details.changes)
                add_participants = "; ".join(details.participants_to_add) if details.participants_to_add else "none"
                remove_participants = "; ".join(details.participants_to_remove) if details.participants_to_remove else "none"
                response = self.generate_response(
                    True,
                    f"Modified event: {details.event_identifier}\n"
                    f"Changes: {changes_desc}\n"
                    f"Add participants: {add_participants}\n"
                    f"Remove participants: {remove_participants}"
                )
            
            else:
                # Step 2c: Handle non-calendar requests
                response = self.generate_response(
                    False,
                    "This doesn't appear to be a calendar-related request. "
                    f"Confidence: {classification.confidence_score:.2f}"
                )

            return response.success, response.message

        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


def main() -> None:
    """Example usage of the routing pattern."""
    # Initialize the router
    router = RequestRouter(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1000
    )

    # Example requests
    requests = [
        "Schedule a team meeting next Tuesday at 2pm for 1 hour with john@example.com "
        "and sarah@example.com to discuss the Q2 roadmap.",
        
        "Move my 3pm meeting with the dev team to 4pm and add lisa@example.com "
        "to the participant list.",
        
        "What's the weather like today?",  # Non-calendar request
    ]

    # Process each request
    for i, request in enumerate(requests, 1):
        print(f"\nProcessing Request #{i}:")
        print(f"Input: {request}")
        success, message = router.route_calendar_request(request)
        print("Result: " + ("Success" if success else "Failed"))
        print(f"Message: {message}\n")
        print("-" * 80)


if __name__ == "__main__":
    main()
