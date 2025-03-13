# Workflow Patterns for building AI Agents


Learn more about the theory and practice behind these patterns:
- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - Anthropic's blog post

### Prompt Chaining

Prompt chaining is a powerful pattern that breaks down complex AI tasks into a sequence of smaller, more focused steps. Each step in the chain processes the output from the previous step, allowing for better control, validation, and reliability.

#### Calendar Assistant Example

Our calendar assistant demonstrates a 3-step prompt chain with validation:

```mermaid
graph LR
    A[User Input] --> B[LLM 1: Extract]
    B --> C{Gate Check}
    C -->|Pass| D[LLM 2: Parse Details]
    C -->|Fail| E[Exit]
    D --> F[LLM 3: Generate Confirmation]
    F --> G[Final Output]
```

#### Step 1: Extract & Validate

- Determines if the input is actually a calendar request
- Provides a confidence score
- Acts as an initial filter to prevent processing invalid requests

#### Step 2: Parse Details

- Extracts specific calendar information
- Structures the data (date, time, participants, etc.)
- Converts natural language to structured data

#### Step 3: Generate Confirmation

- Creates a user-friendly confirmation message
- Optionally generates calendar links
- Provides the final user response

### Routing

Routing is a pattern that directs different types of requests to specialized handlers. This allows for optimized processing of distinct request types while maintaining a clean separation of concerns.

#### Calendar Assistant Example

Our calendar assistant demonstrates routing between new event creation and event modification:

```mermaid
graph LR
    A[User Input] --> B[LLM Router]
    B --> C{Route}
    C -->|New Event| D[New Event Handler]
    C -->|Modify Event| E[Modify Event Handler]
    C -->|Other| F[Exit]
    D --> G[Response]
    E --> G
```

#### Router

- Classifies the request type (new/modify event)
- Provides confidence scoring
- Cleans and standardizes the input

#### Specialized Handlers

- New Event Handler: Creates calendar events
- Modify Event Handler: Updates existing events
- Each optimized for its specific task

### Parallelization

Parallelization runs multiple LLM calls concurrently to validate or analyze different aspects of a request simultaneously.

#### Calendar Assistant Example

Our calendar assistant implements parallel validation guardrails:

```mermaid
graph LR
    A[User Input] --> B[Calendar Check]
    A --> C[Security Check]
    B --> D{Aggregate}
    C --> D
    D -->|Valid| E[Continue]
    D -->|Invalid| F[Exit]
```

#### Parallel Checks

- Calendar Validation: Verifies valid calendar request
- Security Check: Screens for prompt injection
- Run simultaneously for better performance

#### Aggregation

- Combines validation results
- Applies validation rules
- Makes final accept/reject decision

### Orchestrator-Workers

The orchestrator-workers pattern uses a central LLM to dynamically analyze tasks, coordinate specialized workers, and synthesize their results. This creates a flexible system that can adapt to different types of requests while maintaining specialized processing.

#### Blog Writing Example

Our blog writing system demonstrates the orchestrator pattern for content creation:

```mermaid
graph LR
    A[Topic Input] --> B[Orchestrator]
    B --> C[Planning Phase]
    C --> D[Writing Phase]
    D --> E[Review Phase]
    style D fill:#f9f,stroke:#333,stroke-width:2px
```

#### Orchestrator

- Analyzes the blog topic and requirements
- Creates structured content plan
- Coordinates section writing
- Manages content flow and cohesion

#### Planning Phase

- Analyzes topic complexity
- Identifies target audience
- Breaks content into logical sections
- Assigns word count per section
- Defines writing style guidelines

#### Writing Phase

- Specialized workers write individual sections
- Each section maintains context from previous sections
- Follows style and length guidelines
- Captures key points for each section

#### Review Phase

- Evaluates overall cohesion
- Scores content flow (0-1)
- Suggests section-specific improvements
- Produces final polished version