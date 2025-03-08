"""Tests for the chat module."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# Add src directory to Python path
src_dir = str(Path(__file__).parent.parent / 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from ai_cookbook.chat import basic_chat


@dataclass
class MockMessage:
    """Mock message class to simulate OpenAI API response."""
    content: str


@dataclass
class MockChoice:
    """Mock choice class to simulate OpenAI API response."""
    message: MockMessage


@dataclass
class MockCompletion:
    """Mock completion class to simulate OpenAI API response."""
    choices: list[MockChoice]


class MockCompletions:
    """Mock completions class to simulate OpenAI API."""
    def create(self, **kwargs: Any) -> MockCompletion:
        """Mock create method that returns a static response."""
        return MockCompletion([MockChoice(MockMessage("3"))])


class MockChat:
    """Mock chat class to simulate OpenAI API."""
    completions = MockCompletions()


class MockClient:
    """Mock OpenAI client class."""
    chat = MockChat()


def test_basic_chat_letter_count() -> None:
    """Test the LLM's ability to count letters in a word.
    
    This test verifies that the model can accurately perform a simple
    letter counting task, specifically counting the occurrences of 'r'
    in the word 'strawberry'.
    """
    prompt = "Count ALL instances of the letter 'r' in the word 'strawberry' "\
             "Be thorough and count EVERY 'r'. Respond with just the number."
    response = basic_chat(
        prompt=prompt,
        temperature=0.0,  # Use 0 temperature for deterministic responses
        system_message="You are a helpful assistant. Always respond with just the number "\
                      "for counting questions."
    )
    
    # Clean the response to get just the number
    cleaned_response = response.strip().split()[0]
    assert cleaned_response == "3", f"Expected 3 r's in 'strawberry', got {cleaned_response}"


def test_basic_chat_unit_conversion() -> None:
    """Test the LLM's ability to perform unit conversion.
    
    This test verifies that the model can accurately convert 1 meter to feet
    with a precision of 2 decimal places.
    """
    prompt = "Convert 1 meter to feet. Please respond with just the number "\
             "rounded to 2 decimal places."
    response = basic_chat(
        prompt=prompt,
        temperature=0.0,  # Use 0 temperature for deterministic responses
        system_message="You are a helpful assistant. For unit conversions, "\
                      "always respond with just the number rounded to the "\
                      "requested decimal places."
    )
    
    # Clean the response to get just the number
    cleaned_response = float(response.strip().split()[0])
    assert abs(cleaned_response - 3.28) < 0.01, \
        f"Expected 3.28 feet (±0.01), got {cleaned_response}"


def test_basic_chat_letter_count_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the basic_chat function with a mocked OpenAI API response.
    
    This test verifies that the function correctly processes the API response
    without making an actual API call. It uses the same prompt as the real test
    but returns a static response.
    
    Args:
        monkeypatch: pytest fixture for mocking
    """
    # Mock the AzureOpenAI client
    def mock_azure_openai(*args: Any, **kwargs: Any) -> MockClient:
        return MockClient()
    
    monkeypatch.setattr('ai_cookbook.chat.AzureOpenAI', mock_azure_openai)
    
    prompt = "Count ALL instances of the letter 'r' in the word 'strawberry'. "\
             "Be thorough and count EVERY 'r'. Respond with just the number."
    response = basic_chat(
        prompt=prompt,
        temperature=0.0,
        system_message="You are a helpful assistant. Always respond with just the number "\
                      "for counting questions."
    )
    
    # Clean the response to get just the number
    cleaned_response = response.strip()
    assert cleaned_response == "3", f"Expected 3 r's in 'strawberry', got {cleaned_response}"
