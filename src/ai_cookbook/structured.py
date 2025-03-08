"""Module for structured outputs via Azure OpenAI API."""

import os
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, field_validator

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------------------------
# Step 1: Define the response format in a Pydantic model
# --------------------------------------------------------------
class CalendarEvent(BaseModel):
    """Structured format for calendar event information.
    
    Attributes:
        name: The name or title of the event
        date: The date of the event in YYYY-MM-DD format
        time: The time of the event in HH:MM 24-hour format, optional
        participants: List of people participating in the event
    """
    name: str
    date: str  # Format: YYYY-MM-DD
    time: Optional[str] = None  # Format: HH:MM (24-hour)
    participants: List[str]

    @field_validator('date')
    @classmethod
    def validate_date_format(cls, value: str) -> str:
        """Validate that the date is in YYYY-MM-DD format.

        Args:
            value: Date string to validate

        Returns:
            The validated date string

        Raises:
            ValueError: If the date format is invalid
        """
        try:
            datetime.strptime(value, '%Y-%m-%d')
            return value
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
            
    @field_validator('time')
    @classmethod
    def validate_time_format(cls, value: Optional[str]) -> Optional[str]:
        """Validate that the time is in HH:MM 24-hour format.

        Args:
            value: Time string to validate

        Returns:
            The validated time string or None

        Raises:
            ValueError: If the time format is invalid
        """
        if value is None:
            return None
        try:
            datetime.strptime(value, '%H:%M')
            return value
        except ValueError:
            raise ValueError('Time must be in HH:MM format (24-hour)')


def extract_event_info(text: str) -> CalendarEvent:
    """Extract structured calendar event information from text using Azure OpenAI.

    Args:
        text: The input text containing event information

    Returns:
        A CalendarEvent object containing the structured event information

    Raises:
        ValueError: If Azure OpenAI credentials are not properly set
        Exception: If the API call fails
    """
    # --------------------------------------------------------------
    # Step 2: Initialize Azure OpenAI client
    # --------------------------------------------------------------
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZ_OPENAI_ENDPOINT")
    if not api_key or not endpoint:
        raise ValueError("Azure OpenAI credentials must be set in environment variables")
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-10-21",
        azure_endpoint=endpoint
    )
    # Get current date for context
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Prepare system message with date context
    current_time = datetime.now().strftime("%H:%M")
    system_message = f"""Extract event information from the text. Today's date is {current_date} and current time is {current_time}.
    For the date field, if the text mentions a specific date, use it. 
    If the text only mentions a day of the week (e.g., 'Friday'), calculate the actual date based on the current date.
    Return the date in YYYY-MM-DD format.
    For the time field:
    - If a specific time is mentioned (e.g., '2pm', '14:00'), convert it to 24-hour HH:MM format
    - If no time is mentioned, return null
    - For relative times (e.g., 'this afternoon'), use appropriate business hours (e.g., 13:00 for afternoon)"""
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": text},
    ]

    # --------------------------------------------------------------
    # Step 3: Parse the response into a CalendarEvent Pydantic model
    # --------------------------------------------------------------
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        response_format=CalendarEvent
    )
    return completion.choices[0].message.parsed


def main() -> None:
    """Example usage of structured event extraction."""
    try:
        # Example 1: Event with relative date (next Friday)
        print("\nExample 1: Event with relative date")
        event1 = extract_event_info(
            text="Jim and Bob are going to a science fair next Friday at 2pm."
        )
        print("Extracted Event Information:")
        print(f"Name: {event1.name}")
        print(f"Date: {event1.date}")
        print(f"Time: {event1.time if event1.time else 'Not specified'}")
        print(f"Participants: {', '.join(event1.participants)}")

        # Example 2: Event with specific date
        print("\nExample 2: Event with specific date")
        event2 = extract_event_info(
            text="Carol and David have a team meeting on March 15, 2025 at 15:30."
        )
        print("Extracted Event Information:")
        print(f"Name: {event2.name}")
        print(f"Date: {event2.date}")
        print(f"Time: {event2.time if event2.time else 'Not specified'}")
        print(f"Participants: {', '.join(event2.participants)}")

        # Example 3: Event with day of week
        print("\nExample 3: Event with day of week")
        event3 = extract_event_info(
            text="James is presenting at the AI conference this Monday afternoon."
        )
        print("Extracted Event Information:")
        print(f"Name: {event3.name}")
        print(f"Date: {event3.date}")
        print(f"Time: {event3.time if event3.time else 'Not specified'}")
        print(f"Participants: {', '.join(event3.participants)}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
