"""Tests for the routing pattern implementation using real Azure OpenAI API calls.

This module contains tests for the RequestRouter class and its components,
following our established testing standards with real API integration.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from dotenv import load_dotenv
from pydantic import ValidationError

# Add src directory to Python path
src_dir = str(Path(__file__).parent.parent / 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from workflows.routing import (
    CalendarResponse,
    Change,
    ModifyEventDetails,
    NewEventDetails,
    RequestClassification,
    RequestRouter,
    RequestType,
)

# Load environment variables
load_dotenv()


@pytest.fixture
def router() -> RequestRouter:
    """Fixture that returns a RequestRouter instance.
    
    Returns:
        RequestRouter instance configured with Azure OpenAI
    """
    return RequestRouter(
        model="gpt-4o",  # Make sure this matches your deployment name
        temperature=0.1  # Lower temperature for more consistent test results
    )


def test_classify_request(router: RequestRouter) -> None:
    """Test the classify_request step of the routing pattern with real API calls.

    Args:
        router: RequestRouter fixture
    """
    # Test valid calendar request
    valid_input = "Schedule a team meeting next Tuesday at 2pm with john@example.com"
    result = router.classify_request(valid_input)

    assert isinstance(result, RequestClassification), "Expected RequestClassification instance"
    assert result.request_type == RequestType.NEW_EVENT, "Expected NEW_EVENT type"
    assert result.confidence_score > 0.7, "Expected good confidence score for valid request"
    assert result.description, "Expected non-empty description"


def test_extract_new_event_details(router: RequestRouter) -> None:
    """Test the extract_new_event_details step of the routing pattern with real API calls.

    Args:
        router: RequestRouter fixture
    """
    description = (
        "Schedule a team meeting next Tuesday at 2pm for 1 hour "
        "with john@example.com and sarah@example.com about Q2 planning"
    )
    result = router.extract_new_event_details(description)

    assert isinstance(result, NewEventDetails), "Expected NewEventDetails instance"
    assert result.name, "Expected non-empty event name"
    assert result.date, "Expected valid date"
    assert result.duration_minutes == 60, "Expected 1-hour duration"
    assert len(result.participants) == 2, "Expected two participants"
    assert "john@example.com" in result.participants, "Expected john@example.com in participants"
    assert "sarah@example.com" in result.participants, "Expected sarah@example.com in participants"


def test_extract_modify_event_details(router: RequestRouter) -> None:
    """Test the extract_modify_event_details step of the routing pattern with real API calls.

    Args:
        router: RequestRouter fixture
    """
    description = (
        "Move my 3pm dev team meeting to 4pm and add lisa@example.com "
        "to the participant list"
    )
    result = router.extract_modify_event_details(description)

    assert isinstance(result, ModifyEventDetails), "Expected ModifyEventDetails instance"
    assert result.event_identifier, "Expected non-empty event identifier"
    assert result.changes, "Expected at least one change"
    assert result.participants_to_add == ["lisa@example.com"], "Expected one participant to add"
    assert not result.participants_to_remove, "Expected no participants to remove"


def test_generate_response(router: RequestRouter) -> None:
    """Test the generate_response step of the routing pattern with real API calls.

    Args:
        router: RequestRouter fixture
    """
    details = (
        "Created new event: Q2 Planning Meeting at 2:00 PM. "
        "Participants: john@example.com; sarah@example.com"
    )
    result = router.generate_response(True, details)

    assert isinstance(result, CalendarResponse), "Expected CalendarResponse instance"
    assert result.success, "Expected success=True"
    assert result.message, "Expected non-empty message"
    assert "2:00 PM" in result.message, "Expected time in 12-hour format"
    assert "john@example.com; sarah@example.com" in result.message, "Expected participants in message"


def test_route_calendar_request_success(router: RequestRouter) -> None:
    """Test the complete routing pattern with a valid request using real API calls.

    Args:
        router: RequestRouter fixture
    """
    request = (
        "Schedule a team meeting next Tuesday at 2pm for 1 hour "
        "with john@example.com and sarah@example.com to discuss Q2 planning"
    )
    success, message = router.route_calendar_request(request)

    assert success, "Expected successful request processing"
    assert "2:00 PM" in message, "Expected time in 12-hour format"
    assert "john@example.com" in message, "Expected first participant in message"
    assert "sarah@example.com" in message, "Expected second participant in message"


def test_route_calendar_request_invalid(router: RequestRouter) -> None:
    """Test the routing pattern with an invalid request using real API calls.

    Args:
        router: RequestRouter fixture
    """
    # Test non-calendar request
    request = "What's the weather like today?"
    success, message = router.route_calendar_request(request)

    assert not success, "Expected unsuccessful request processing"
    assert "calendar" in message.lower(), "Expected message about non-calendar request"


def test_invalid_date_format() -> None:
    """Test validation of date format in NewEventDetails model."""
    with pytest.raises(ValidationError):
        NewEventDetails(
            name="Invalid Meeting",
            date="not-a-date",  # Invalid date format
            duration_minutes=60,
            participants=["test@example.com"]
        )


def test_empty_input(router: RequestRouter) -> None:
    """Test handling of empty input in the routing pattern.

    Args:
        router: RequestRouter fixture
    """
    with pytest.raises(ValueError, match="User input cannot be empty"):
        router.route_calendar_request("")
