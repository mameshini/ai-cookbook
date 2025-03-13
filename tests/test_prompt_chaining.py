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

from workflows.prompt_chaining import CalendarRequest, PromptChainProcessor

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


def test_extract_and_validate(processor: PromptChainProcessor) -> None:
    """Test the extract_and_validate step of the prompt chain with real API calls.
    
    Args:
        processor: PromptChainProcessor fixture
    """
    # Test valid calendar request
    valid_input = """Schedule a team meeting for next Tuesday at 2 PM.
    It will be a 1-hour discussion with john@example.com and sarah@example.com
    about the Q2 planning."""
    
    is_valid, confidence, reasoning = processor.extract_and_validate(valid_input)
    
    assert is_valid is True, "Expected input to be validated as a valid calendar request"
    assert confidence > 0.7, "Expected good confidence score for valid request"
    assert reasoning, "Expected non-empty reasoning"
    
    # Test invalid calendar request
    invalid_input = "What's the weather like today?"
    
    is_valid, confidence, reasoning = processor.extract_and_validate(invalid_input)
    
    assert is_valid is False, "Expected input to be validated as invalid calendar request"
    assert confidence > 0.7, "Expected high confidence score for clear non-calendar request"
    assert "weather" in reasoning.lower() or "calendar" in reasoning.lower(), \
        "Expected reasoning to mention weather or calendar context"


def test_parse_details(processor: PromptChainProcessor) -> None:
    """Test the parse_details step of the prompt chain with real API calls.
    
    Args:
        processor: PromptChainProcessor fixture
    """
    user_input = """Schedule a team meeting for next Tuesday at 2 PM.
    It will be a 1-hour discussion with john@example.com and sarah@example.com
    about the Q2 planning."""
    
    result = processor.parse_details(user_input)
    
    assert isinstance(result, CalendarRequest), "Expected CalendarRequest instance"
    assert isinstance(result.date, str), "Expected date to be a string"
    assert isinstance(result.time, str), "Expected time to be a string"
    assert isinstance(result.duration, int), "Expected duration to be an integer"
    assert isinstance(result.participants, list), "Expected participants to be a list"
    assert len(result.participants) == 2, "Expected two participants"
    assert all('@' in p for p in result.participants), "Expected email addresses in participants"
    assert result.title, "Expected non-empty title"


def test_generate_confirmation(processor: PromptChainProcessor) -> None:
    """Test the generate_confirmation step of the prompt chain with real API calls.
    
    Args:
        processor: PromptChainProcessor fixture
    """
    calendar_request = CalendarRequest(
        date="2024-03-19",
        time="14:00",
        duration=60,
        participants=["john@example.com", "sarah@example.com"],
        title="Q2 Planning Discussion",
        description="Team meeting for Q2 planning"
    )
    
    result = processor.generate_confirmation(calendar_request)
    
    # Check that the confirmation includes key event details in both formats
    assert calendar_request.date in result or "March 19, 2024" in result, \
        "Expected date in ISO or human-readable format"
    assert "2:00 PM" in result or calendar_request.time in result, \
        "Expected time in 12-hour or 24-hour format"
    assert calendar_request.title in result, "Expected title in confirmation"
    assert any(p in result for p in calendar_request.participants), \
        "Expected participants in confirmation"


def test_process_request_success(processor: PromptChainProcessor) -> None:
    """Test the complete prompt chain processing with a valid request using real API calls.
    
    Args:
        processor: PromptChainProcessor fixture
    """
    user_input = """Schedule a team meeting for next Tuesday at 2 PM.
    It will be a 1-hour discussion with john@example.com and sarah@example.com
    about the Q2 planning."""
    
    result = processor.process_request(user_input)
    
    assert result is not None, "Expected successful processing"
    assert "PM" in result, "Expected time format in final response"
    assert any(email in result for email in ["john@example.com", "sarah@example.com"]), \
        "Expected participant emails in final response"
    assert "Q2" in result, "Expected meeting topic in final response"


def test_process_request_invalid(processor: PromptChainProcessor) -> None:
    """Test the prompt chain processing with an invalid request using real API calls.
    
    Args:
        processor: PromptChainProcessor fixture
    """
    invalid_input = "What's the weather like today?"
    
    result = processor.process_request(invalid_input)
    
    assert result is not None, "Expected response for invalid request"
    assert "couldn't process" in result.lower(), "Expected error message for invalid request"
    assert "calendar request" in result.lower(), "Expected mention of calendar request in error"
