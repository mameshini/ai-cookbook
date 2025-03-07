# AI Cookbook

A collection of AI patterns and recipes using Azure OpenAI GPT-4 and Claude 3.5 Sonnet v2 via AWS Bedrock.
This project can be opened used with Windsurf AI IDE.
You can subscribe to Windsurf at https://codeium.com/refer?referral_code=5o03sz1fshqlzdx0

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your credentials:

Create a `.env` file and add your credentials:
```bash
# Azure OpenAI credentials:
AZURE_OPENAI_API_KEY="<your-azure-openai-api-key>"
AZURE_OPENAI_ENDPOINT="<your-azure-openai-endpoint>"

# AWS credentials:
AWS_ACCESS_KEY_ID="<your-aws-access-key>"
AWS_SECRET_ACCESS_KEY="<your-aws-secret-key>"
AWS_REGION="<your-aws-region>"
```

## Usage

Run the basic chat example:
```bash
python -m ai_cookbook.chat
```

This will run a simple example that generates a poem about empathy. You can modify the example in `src/ai_cookbook/chat.py` to try different prompts and parameters:

- Change the `prompt` to ask different questions
- Adjust `temperature` (0-1) to control response randomness
- Modify `system_message` to set different contexts
- Set `max_tokens` to limit response length

## Testing

Run tests with pytest:
```bash
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_chat.py
```

The test suite includes:
- Real API tests that verify actual GPT-4 responses
- Mocked tests that validate functionality without making API calls
- Tests for letter counting and unit conversion capabilities

To debug tests:
1. Use VS Code's debugger with the "Python: Debug Tests" configuration
2. Set breakpoints in your test files
3. The debugger will stop at breakpoints and show variable values inline

## Project Structure

```
├── src/
│   └── ai_cookbook/       # Main package directory
│       ├── __init__.py
│       └── chat.py        # Chat completion implementation
├── tests/                 # Test directory
│   ├── __init__.py
│   └── test_chat.py      # Tests for chat functionality
├── .env.example          # Example environment variables
├── .gitignore            # Git ignore file
├── README.md             # This file
└── requirements.txt      # Project dependencies
```

### Key Components

- `chat.py`: Implements LLM chat completion with:
  - Support for Azure OpenAI GPT-4
  - Type annotations for better code clarity
  - Comprehensive docstrings following PEP 257
  - Error handling for missing credentials
  - Configurable parameters (temperature, max_tokens)

- `test_chat.py`: Test suite with:
  - Real API integration tests for Azure OpenAI
  - Mocked tests using pytest fixtures
  - Type annotations and detailed docstrings
