# AI Cookbook

A collection of AI patterns and recipes using Azure OpenAI GPT-4 and Claude 3.5 Sonnet v2 via AWS Bedrock.
These patterns can be used to provide examples and architecture guidance for AI Coding Agents when building AI applications.

You can subscribe to Windsurf coding agent at https://codeium.com/refer?referral_code=5o03sz1fshqlzdx0

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
│   ├── ai_cookbook/       # Core AI patterns
│   │   ├── chat.py        # Chat completion implementation
│   │   ├── structured.py  # Structured output pattern
│   │   ├── tools.py       # Function calling pattern
│   │   └── retrieval.py   # Knowledge base retrieval
│   ├── langchain/        # LangChain agent patterns
│   │   ├── chat_agent.py  # Conversational agent with tools
│   │   ├── chat_agent_eval.py  # Agent observability and evals
│   │   ├── kb_rag_eval.py # Offline RAG evaluation with with RAGAS metrics
│   └── workflows/         # Advanced workflow patterns
│       ├── orchestrator.py  # Blog post orchestration
│       ├── routing.py      # Request classification
│       └── parallelization.py  # Concurrent processing
├── tests/                 # Test directory
│   └── test_chat.py      # Tests for chat functionality
├── .env.example          # Example environment variables
├── .gitignore            # Git ignore file
├── README.md             # This file
└── requirements.txt      # Project dependencies
```

### Building Blocks and Patterns

This project demonstrates several key patterns for building AI agents and workflows.  

#### Agent Pattern (LangChain ReAct with Tools and RAG)
- `src/langchain/chat_agent.py`: Implements an intelligent conversational agent using LangChain with:
  - Azure OpenAI integration using GPT-4
  - Custom tools for weather lookup, Wikipedia search, RAG, and datetime functions
  - Pydantic models for type-safe input validation
  - ConversationBufferMemory for context retention
  - Gradio web interface for interactive chat
  - Environment variable management with dotenv

- `src/langchain/chat_agent_eval.py`: Implements agent observability and evals with:
  - LangSmith integration for run tracking and evaluation
  - Multiple evaluation criteria:
    - Response correctness
    - Tool usage appropriateness
    - Factual accuracy
  - Pydantic models for structured evaluation responses
  - Real-time evaluation metrics during chat interactions

#### Basic Chat Completion
- `chat.py`: Implements LLM chat completion with:
  - Support for Azure OpenAI GPT-4
  - Type annotations for better code clarity
  - Comprehensive docstrings following PEP 257
  - Error handling for missing credentials
  - Configurable parameters (temperature, max_tokens)

#### Structured Output Pattern
- `structured.py`: Implements structured data extraction with:
  - Pydantic models for data validation
  - Calendar event information parsing
  - Date extraction in YYYY-MM-DD format
  - Time extraction in 24-hour HH:MM format
  - Azure OpenAI's beta.chat.completions.parse functionality

#### Function Calling Pattern
- `tools.py`: Demonstrates function calling capabilities:
  - Real-time weather data retrieval 
  - Integration with open-meteo.com weather API
  - Structured responses with natural language context

#### Retrieval Pattern (see LangChain RAG for advanced retrieval)
- `retrieval.py`: Knowledge base interaction using:
  - JSON-based e-commerce knowledge store
  - Function calling for policy and service information
  - Graceful handling of out-of-scope questions
  - Source attribution with record IDs

#### Agentic Workflow Patterns (pure Python)
- `src/workflows/`: Advanced AI workflow implementations:
  - Orchestration + enriched LLMs form patterns of intelligent outcomes
  - The key patterns are: prompt chaining, routing parallelization orchestration, evaluator-optimizer, and service agent
  - Type-safe implementations using Python 3.11 and Pydantic
  - PEP 8 and PEP 257 compliant with comprehensive docstrings
  - Robust error handling and automatic retries with tenacity

##### 1. Prompt Chaining Pattern (`prompt_chaining.py`)
- Sequential processing of complex tasks
- Each step refines or builds upon previous outputs
- Strong validation between chain steps
- Demonstrated in calendar request processing

##### 2. Routing Pattern (`routing.py`)
- Request classification and specialized handling
- Calendar request routing implementation
- Confidence scoring for request types
- Clean separation of concerns

##### 3. Parallelization Pattern (`parallelization.py`)
- Concurrent LLM calls for validation
- Calendar and security check parallelization
- Efficient aggregation of validation results
- Improved response times through parallel processing

##### 4. Orchestrator Pattern (`orchestrator.py`)
- Blog post creation orchestration using Azure OpenAI
- Multi-phase workflow: Planning, Writing, Review, and Revision
- Type-safe implementation with Pydantic models:
  - `OrchestratorPlan`: Blog structure and tasks
  - `SectionContent`: Written content and key points
  - `ReviewFeedback`: Review score and suggested edits

For detailed documentation of these patterns, see `src/workflows/README.md`.

### Chat Agent with Tool Usage

The chat agent pattern demonstrates how to create an interactive AI assistant that can use external tools and memory.  
Agents use LLMs and have access to tools so that they can chain actions and reasoning to do larger and more complex tasks.
RAG is used to retrieve relevant information from a knowledge base.


```mermaid
graph TD
    A[User Input] --> B[LangChain Agent]
    B --> C{Reason \& Tools}
    
    C -->|Financial Data| R[RAG Search Tool]
    C -->|Weather| D[Weather API Tool]
    C -->|Information| E[Wikipedia Tool]
    C -->|Chat| L[LLM]

    R -->|Query| H[Bedrock Knowledge Base]
    H -->|Retrieve| R

    R --> M[Conversation Memory]
    D --> M[Conversation Memory]
    E --> M[Conversation Memory]
    L --> M[Conversation Memory]
    
    M --> EVAL{Agent Finish?}

    EVAL -->|Yes| F[Response Generation]
    EVAL -->|No| B

    F --> G[User Output]
```


#### Testing
- `test_chat.py`: Test suite with:
  - Real API integration tests for Azure OpenAI
  - Mocked tests using pytest fixtures
  - Type annotations and detailed docstrings
