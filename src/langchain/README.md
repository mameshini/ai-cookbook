# LangChain Patterns for Building AI Agents

This directory contains examples and patterns for building AI agents using LangChain. 

### Chat Agent with Tool Usage

The chat agent pattern demonstrates how to create an interactive AI assistant that can use external tools.  
Agents use LLMs and have access to tools so that they can chain actions and reasoning to do larger and more complex tasks.

Key features: chained prompt, model, AgentExecutor, tools, and memory.
This agent implements the "ReAct" (Reason + Act) pattern, where the agent can:

1. Reason about what tools it needs
2. Act by calling those tools
3. Remember previous actions through the scratchpad
4. Generate appropriate responses based on tool results


```mermaid
graph TD
    A[User Input] --> B[LangChain Agent]
    B --> C{Reason & Act}
    
    C -->|Weather| D[Weather API Tool]
    C -->|Information| E[Wikipedia Tool]
    C -->|Chat| L[LLM]

    D --> M[Conversation Memory]
    E --> M[Conversation Memory]
    L --> M[Conversation Memory]

    M --> EVAL{Agent Finish?}

    EVAL -->|Yes| F[Response Generation]
    EVAL -->|No| B

    F --> G[User Output]```

#### Components:
1. **Tool Definition**
   - Pydantic models for input validation
   - Type-safe function implementations
   - Clear documentation and error handling

2. **Agent Configuration**
   - Azure OpenAI integration
   - Conversation memory management
   - Tool registration and formatting

3. **UI Integration**
   - Gradio interface for web interaction
   - Async support for responsive UX
   - Error handling and user feedback

### Best Practices

1. **Type Safety**
   - Python 3.11 type hints throughout
   - Pydantic models for data validation
   - Clear return type definitions

2. **Error Handling**
   - Comprehensive try-except blocks
   - User-friendly error messages
   - Graceful fallbacks

3. **Memory Management**
   - Conversation history tracking
   - Context-aware responses
   - Efficient state management

4. **Tool Integration**
   - Modular tool definitions
   - Clear input/output contracts
   - Reusable components

## Implementation Examples

1. `chat_agent.py`: Demonstrates a full chat agent with:
   - Weather lookup capability
   - Wikipedia search integration
   - Conversation memory
   - Gradio web interface

## Getting Started

1. Ensure you have the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your environment variables:
   ```bash
   AZURE_OPENAI_API_KEY="your_key"
   AZURE_OPENAI_ENDPOINT="your_endpoint"
   AZURE_OPENAI_DEPLOYMENT="your_deployment"
   ```

3. Run the chat agent:
   ```bash
   python chat_agent.py
   ```

