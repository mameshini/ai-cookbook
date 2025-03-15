"""Tests for the prompt chaining module using real Azure OpenAI API calls."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from dotenv import load_dotenv

# Add src directory to Python path
src_dir = str(Path(__file__).parent.parent / 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from workflows.prompt_chaining import EventDetails, EventExtraction, EventConfirmation, PromptChainProcessor

# Load environment variables
load_dotenv()

@pytest.fixture
def processor() -> PromptChainProcessor:
    """Fixture that returns a PromptChainProcessor instance.
    
    Returns:
        PromptChainProcessor instance configured with Azure OpenAI
    """
    return PromptChainProcessor(
        model="gpt-4o",  # Make sure this matches your deployment name
        temperature=0.1  # Lower temperature for more consistent test results
    )


def test_extract_event_info(processor: PromptChainProcessor) -> None:
    """Test the extract_event_info step of the prompt chain with real API calls.
    
    Args:
        processor: PromptChainProcessor fixture
    """
    # Test valid calendar request
    valid_input = """Schedule a team meeting for next Tuesday at 2 PM.
    It will be a 1-hour discussion with john@example.com and sarah@example.com
    about the Q2 planning."""
    
    result = processor.extract_event_info(valid_input)
    
    assert isinstance(result, EventExtraction), "Expected EventExtraction instance"
    assert result.is_calendar_event is True, "Expected input to be validated as a valid calendar request"
    assert result.confidence_score > 0.7, "Expected good confidence score for valid request"
    assert result.description, "Expected non-empty description"
    
    # Test invalid calendar request
    invalid_input = "What's the weather like today?"
    
    result = processor.extract_event_info(invalid_input)
    
    assert isinstance(result, EventExtraction), "Expected EventExtraction instance"
    assert result.is_calendar_event is False, "Expected input to be validated as invalid calendar request"
    assert result.confidence_score > 0.7, "Expected high confidence score for clear non-calendar request"


def test_parse_event_details(processor: PromptChainProcessor) -> None:
    """Test the parse_event_details step of the prompt chain with real API calls.
    
    Args:
        processor: PromptChainProcessor fixture
    """
    description = """Schedule a team meeting for next Tuesday at 2 PM.
    It will be a 1-hour discussion with john@example.com and sarah@example.com
    about the Q2 planning."""
    
    result = processor.parse_event_details(description)
    
    assert isinstance(result, EventDetails), "Expected EventDetails instance"
    assert isinstance(result.name, str), "Expected name to be a string"
    assert isinstance(result.date, str), "Expected date to be in ISO 8601 format"
    assert isinstance(result.duration_minutes, int), "Expected duration_minutes to be an integer"
    assert isinstance(result.participants, list), "Expected participants to be a list"
    assert len(result.participants) == 2, "Expected two participants"
    assert all('@' in p for p in result.participants), "Expected email addresses in participants"
    assert result.name, "Expected non-empty name"


def test_generate_confirmation(processor: PromptChainProcessor) -> None:
    """Test the generate_confirmation step of the prompt chain with real API calls.
    
    Args:
        processor: PromptChainProcessor fixture
    """
    event_details = EventDetails(
        name="Q2 Planning Discussion",
        date="2025-03-19T14:00:00",
        duration_minutes=60,
        participants=["john@example.com", "sarah@example.com"]
    )
    
    result = processor.generate_confirmation(event_details)
    
    assert isinstance(result, EventConfirmation), "Expected EventConfirmation instance"
    assert isinstance(result.confirmation_message, str), "Expected confirmation_message to be a string"
    assert isinstance(result.calendar_link, (str, type(None))), "Expected calendar_link to be string or None"
    
    # Check that the confirmation includes key event details
    assert "March 19, 2025" in result.confirmation_message, "Expected date in human-readable format"
    assert "2:00 PM" in result.confirmation_message, "Expected time in 12-hour format"
    assert event_details.name in result.confirmation_message, "Expected name in confirmation"
    assert any(p in result.confirmation_message for p in event_details.participants), \
        "Expected participants in confirmation"
    assert "Calendar Assistant" in result.confirmation_message, "Expected Calendar Assistant signature"


def test_process_calendar_request_success(processor: PromptChainProcessor) -> None:
    """Test the complete prompt chain processing with a valid request using real API calls.
    
    Args:
        processor: PromptChainProcessor fixture
    """
    user_input = """Schedule a team meeting for next Tuesday at 2 PM.
    It will be a 1-hour discussion with john@example.com and sarah@example.com
    about the Q2 planning."""
    
    success, message = processor.process_calendar_request(user_input)
    
    assert success is True, "Expected successful processing"
    assert "PM" in message, "Expected time format in final response"
    assert any(email in message for email in ["john@example.com", "sarah@example.com"]), \
        "Expected participant emails in final response"
    assert "Q2" in message, "Expected meeting topic in final response"
    assert "Calendar Assistant" in message, "Expected Calendar Assistant signature"


def test_process_calendar_request_invalid(processor: PromptChainProcessor) -> None:
    """Test the prompt chain processing with an invalid request using real API calls.
    
    Args:
        processor: PromptChainProcessor fixture
    """
    invalid_input = "What's the weather like today?"
    
    success, message = processor.process_calendar_request(invalid_input)
    
    assert success is False, "Expected unsuccessful processing"
    assert "Not a valid calendar request" in message, "Expected invalid request message"
    assert "confidence" in message.lower(), "Expected confidence score in message"
