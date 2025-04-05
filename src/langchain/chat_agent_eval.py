"""
Chat Agent with LangChain and Gradio UI.

This module implements a conversational agent using LangChain with OpenAI integration
and Gradio web interface. The agent can perform weather lookups and Wikipedia searches.
"""

import os
import datetime
import requests
import wikipedia
import gradio as gr
from typing import List, Dict, Any
import boto3
from botocore.config import Config
from langchain.retrievers import AmazonKnowledgeBasesRetriever
from langchain_community.chat_models import BedrockChat
from langchain.chains import RetrievalQA

# LangSmith imports
from langsmith import Client, tracing_context
from langsmith.evaluation import EvaluationResult
from pydantic import BaseModel, Field

# Initialize LangSmith client
langsmith_client = Client()

# Pydantic model for evaluation responses
class EvaluationResponse(BaseModel):
    """Response schema for evaluation functions."""
    score: float = Field(..., description="Score from 0-1 based on the evaluation criteria")
    reasoning: str = Field(..., description="Detailed explanation for the score")

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ai-cookbook-chat-agent"  # Project name in LangSmith
from dotenv import load_dotenv, find_dotenv
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.globals import set_debug

# Load environment variables
_ = load_dotenv(find_dotenv())  # read local .env file

# Validate Azure OpenAI credentials
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION", "2024-10-21")
model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")  # This should be your Azure OpenAI deployment name

if not api_key or not endpoint:
    raise ValueError("Azure OpenAI credentials must be set in environment variables")


class OpenMeteoInput(BaseModel):
    """Input schema for OpenMeteo weather API."""
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")


@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> str:
    """Get the current temperature for a location. You should:
    1. Convert city/location names to coordinates (e.g. San Francisco is at lat=37.7749, long=-122.4194)
    2. Use those coordinates to fetch the temperature
    
    Args:
        latitude: Latitude of the location (e.g. 37.7749 for San Francisco)
        longitude: Longitude of the location (e.g. -122.4194 for San Francisco)
    
    Returns:
        str: Current temperature in Fahrenheit"""
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
        'temperature_unit': 'fahrenheit'
    }

    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) 
                 for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']
    
    closest_time_index = min(range(len(time_list)), 
                           key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    
    return f'The current temperature is {current_temperature}°F'


@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[:3]:
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)


@tool
def get_current_datetime() -> str:
    """Get the current date and time in a human-readable format."""
    current_time = datetime.datetime.now()
    return current_time.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")


@tool
def search_rag(query: str) -> str:
    """Search financial statements and real estate information using RAG."""
    # Initialize Bedrock clients
    region_name = os.getenv('AWS_DEFAULT_REGION', 'us-west-2')
    kb_id = os.getenv('KNOWLEDGE_BASE_ID', 'Y5UHFTTWAT')  # Default KB ID
    
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=region_name,
        config=Config(retries={'max_attempts': 3})
    )
    
    bedrock_agent_client = boto3.client(
        service_name='bedrock-agent-runtime',
        region_name=region_name
    )
    
    # Initialize Bedrock chat model
    llm = BedrockChat(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        client=bedrock_client,
        region_name=region_name,
        model_kwargs={"temperature": 0.0, "max_tokens": 2000}
    )
    
    # Initialize retriever
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,
        client=bedrock_agent_client,
        region_name=region_name,
        retrieval_config={
            "vectorSearchConfiguration": {"numberOfResults": 5},
            "overrideSearchType": "HYBRID"
        }
    )
    
    # Initialize QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    # Get response
    response = qa_chain.invoke(query)
    return response["result"]


def setup_agent() -> AgentExecutor:
    """Set up the LangChain agent with tools and model."""
    tools = [get_current_temperature, search_wikipedia, get_current_datetime, search_rag]
    set_debug(True)  # Enable debug mode for detailed logging
    functions = [format_tool_to_openai_function(f) for f in tools]
    
    model = AzureChatOpenAI(
        openai_api_version=api_version,
        azure_deployment=model_name,
        temperature=0,
        tags=["chat-agent"],  # Add tags for better filtering in LangSmith
        metadata={"agent_type": "weather_wiki_datetime"}
    ).bind(functions=functions)
    
    memory = ConversationBufferMemory(
        return_messages=True, 
        memory_key="chat_history",
        input_key="input",
        output_key="output"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are helpful but sassy assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    chain = RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
    ) | prompt | model | OpenAIFunctionsAgentOutputParser()
    
    return AgentExecutor(
        agent=chain,
        tools=tools,
        verbose=True,
        memory=memory,
        return_intermediate_steps=True,  # Enable returning intermediate steps
        tags=["chat-agent"],  # Add tags for better filtering in LangSmith
        metadata={
            "agent_type": "weather_wiki_datetime",
            "tools": ["weather", "wikipedia", "datetime"],
            "memory_type": "conversation_buffer"
        }
    )


# Initialize the agent globally
_agent = setup_agent()

def chat_response(message: str, history: List[List[str]]) -> str:
    """Process user message and return agent's response."""
    try:
        # Convert Gradio chat history to format suitable for memory
        formatted_history = []
        for human, ai in history:
            formatted_history.extend([("human", human), ("ai", ai)])
        
        # Add run metadata for better tracking
        run_metadata = {
            "conversation_id": str(hash(str(formatted_history))),
            "message_type": "user_input",
            "history_length": len(formatted_history)
        }
        
        response = _agent.invoke(
            {
                "input": message,
                "chat_history": formatted_history
            },
            config={
                "metadata": run_metadata,
                "tags": ["chat-interaction"]
            }
        )
        return response["output"]
    except Exception as e:
        # Log error to LangSmith
        langsmith_client.create_run(
            name="chat_error",
            run_type=RunTypeEnum.chain,
            error=str(e),
            inputs={"message": message, "history": history},
            tags=["error", "chat-agent"],
            metadata={"error_type": type(e).__name__}
        )
        return f"Error: {str(e)}"


def create_ui() -> gr.Interface:
    """Create and configure the Gradio UI."""
    css = """
    .example {
        padding: 8px 12px !important;
        border: 1px solid #ccc !important;
        border-radius: 8px !important;
        margin: 4px !important;
        background-color: #f8f9fa !important;
        transition: all 0.3s ease !important;
    }
    .example:hover {
        background-color: #e9ecef !important;
        border-color: #adb5bd !important;
        transform: translateY(-1px) !important;
    }
    .examples {
        gap: 8px !important;
        flex-wrap: wrap !important;
        padding: 10px !important;
    }
    """
    
    interface = gr.ChatInterface(
        fn=chat_response,
        title="AI Assistant",
        description="""I can help you with:
        - 🌤️ Weather information for any city
        - 📚 Wikipedia searches and summaries
        - 💬 Engaging conversation on various topics""",
        theme="soft",
        css=css,
        examples=[
            "What's the weather in San Francisco?",
            "What was the primary reason for the increase in net cash provided by operating activities for Octank Financial in 2021?",
            "Tell me about the history of artificial intelligence",
            "Search Wikipedia for quantum computing",
            "What's the current temperature in London?",
            "Tell me about the Python programming language",
        ]
    )
    return interface


def evaluate_response_correctness(run_data: dict) -> EvaluationResult:
    """Evaluate if the response correctly addresses the user's query.
    
    Args:
        run_data: Dictionary containing run information
        
    Returns:
        EvaluationResult with score and feedback
    """
    # Extract tool usage information
    intermediate_steps = run_data.get('intermediate_steps', [])
    tool_info = []
    for action, observation in intermediate_steps:
        tool_info.append(f"Tool: {action.tool}\nInput: {action.tool_input}\nOutput: {observation}")
    tool_usage = "\n\n".join(tool_info) if tool_info else "No tools were used"

    prompt = f"""
    You are evaluating the correctness of an AI assistant's response.

    User Query: {run_data.get('input')}
    
    Tools Used and Their Output:
    {tool_usage}
    
    Assistant's Response: {run_data.get('output')}
    
    Evaluate if the response correctly addresses the user's query, taking into account:
    1. Does the response use the tool outputs correctly?
    2. Is the information provided accurate based on the tool outputs?
    3. Is the response complete and well-formatted?
    
    Score from 0-1, where:
    - 1.0: Perfect response that fully addresses the query and correctly uses tool outputs
    - 0.7-0.9: Good response with minor issues (e.g. slightly verbose or missing minor details)
    - 0.4-0.6: Partial response with significant issues (e.g. incomplete or partially incorrect)
    - 0.0-0.3: Poor response (completely incorrect or irrelevant)
    """
    
    eval_chain = AzureChatOpenAI(
        temperature=0,
        model="gpt-4o",
        name="AgentEval"
    ).bind(
        functions=[{
            "name": "evaluate",
            "description": "Evaluate the response and provide a score with reasoning",
            "parameters": EvaluationResponse.model_json_schema()
        }],
        function_call={"name": "evaluate"}
    )
    
    # Disable tracing for evaluation runs
    with tracing_context(enabled=False):
        result = eval_chain.invoke(prompt)
    
    # Parse function call result
    function_args = result.additional_kwargs["function_call"]["arguments"]
    response = EvaluationResponse.model_validate_json(function_args)
    
    return EvaluationResult(
        key="response_correctness",
        score=response.score,
        comment=response.reasoning
    )


def evaluate_tool_usage(run_data: dict) -> EvaluationResult:
    """Evaluate if tools were used appropriately.
    
    Args:
        run_data: Dictionary containing run information
        
    Returns:
        EvaluationResult with score and feedback
    """
    # Extract tool usage from intermediate steps
    intermediate_steps = run_data.get('intermediate_steps', [])
    tool_usage = []
    
    for action, observation in intermediate_steps:
        tool_usage.append({
            'tool': action.tool,
            'input': action.tool_input,
            'output': observation
        })
    
    prompt = f"""
    You are evaluating if the AI assistant used available tools appropriately for the user's query.
    
    User Query: {run_data.get('input')}
    Available Tools:
    - get_current_temperature: Gets current temperature for a location (sufficient for weather queries)
    - search_wikipedia: Searches Wikipedia articles
    - get_current_datetime: Gets current date and time
    
    Tools Used: {[usage['tool'] for usage in tool_usage]}
    Tool Calls:
    {chr(10).join([f'- {usage["tool"]}: {usage["input"]} -> {usage["output"]}' for usage in tool_usage])}
    Final Response: {run_data.get('output')}
    
    Evaluation Criteria:
    1. Tool Selection: Did the assistant choose the appropriate tool(s) for the query?
    2. Tool Input: Were the tool inputs well-formatted and relevant?
    3. Tool Output Usage: Was the tool output used effectively in the response?
    
    Important Notes:
    - For weather queries, using get_current_temperature alone is perfectly valid
    - Multiple tool calls are only necessary if the query requires different types of information
    
    Score from 0-1, where:
    - 1.0: Perfect tool usage - right tool(s) selected and used effectively
    - 0.7-0.9: Good tool usage with minor issues (e.g., slightly inefficient but still effective)
    - 0.4-0.6: Significant issues (wrong tool choice or poor usage of correct tool)
    - 0.0-0.3: Major problems (unnecessary tools or completely wrong usage)
    """
    
    eval_chain = AzureChatOpenAI(
        temperature=0,
        model="gpt-4o",
        name="AgentEval"
    ).bind(
        functions=[{
            "name": "evaluate",
            "description": "Evaluate the response and provide a score with reasoning",
            "parameters": EvaluationResponse.model_json_schema()
        }],
        function_call={"name": "evaluate"}
    )
    
    # Disable tracing for evaluation runs
    with tracing_context(enabled=False):
        result = eval_chain.invoke(prompt)
    
    # Parse function call result
    function_args = result.additional_kwargs["function_call"]["arguments"]
    response = EvaluationResponse.model_validate_json(function_args)
    
    return EvaluationResult(
        key="tool_usage",
        score=response.score,
        comment=response.reasoning
    )


def evaluate_factual_accuracy(run_data: dict) -> EvaluationResult:
    """Evaluate the factual accuracy of the response.
    
    Args:
        run_data: Dictionary containing run information
        
    Returns:
        EvaluationResult with score and feedback
    """
    intermediate_steps = run_data.get('intermediate_steps', [])
    tool_outputs = [step[1] for step in intermediate_steps if step and step[1]]
    
    # Extract user query for context
    user_query = run_data.get('input', '')
    
    prompt = f"""
    You are evaluating the factual accuracy of an AI assistant's response.
    
    User Query: {user_query}
    Tool Outputs: {tool_outputs}
    Assistant's Response: {run_data.get('output')}
    
    Evaluate the factual accuracy of the response using these criteria:
    1. Core Information: Is all information from tool outputs used correctly?
    2. Additional Context: Does any additional information (if present) enhance the response without contradicting tool outputs?
    3. Relevance: Is all information (both from tools and additional) relevant to the user's query?
    
    Scoring Guide:
    - 1.0: Response is fully grounded in tool outputs AND any additional information is relevant and enhances the response
    - 0.7-0.9: Response uses tool outputs correctly but additional information may be slightly off or less relevant
    - 0.4-0.6: Response partially misuses tool outputs or includes irrelevant/distracting additional information
    - 0.0-0.3: Response contradicts tool outputs or includes incorrect/misleading information
    
    Note: A response can score 1.0 even if it includes information beyond the tool outputs, as long as that information is relevant and enhances the response without contradicting the tool outputs.
    """
    
    eval_chain = AzureChatOpenAI(
        temperature=0,
        model="gpt-4o",
        name="AgentEval"
    ).bind(
        functions=[{
            "name": "evaluate",
            "description": "Evaluate the response and provide a score with reasoning",
            "parameters": EvaluationResponse.model_json_schema()
        }],
        function_call={"name": "evaluate"}
    )
    
    # Disable tracing for evaluation runs
    with tracing_context(enabled=False):
        result = eval_chain.invoke(prompt)
    
    # Parse function call result
    function_args = result.additional_kwargs["function_call"]["arguments"]
    response = EvaluationResponse.model_validate_json(function_args)
    
    return EvaluationResult(
        key="factual_accuracy",
        score=response.score,
        comment=response.reasoning
    )


def run_evaluations(run_data: dict) -> List[EvaluationResult]:
    """Run all evaluations on a chat interaction.
    
    Args:
        run_data: Dictionary containing run information
        
    Returns:
        List of evaluation results. Empty list if ONLINE_EVAL is False.
    """
    # Check if online evaluation is enabled
    online_eval = os.getenv("ONLINE_EVAL", "False").lower() == "true"
    if not online_eval:
        return []
    
    evaluators = [
        evaluate_response_correctness,
        evaluate_tool_usage,
        evaluate_factual_accuracy
    ]
    
    results = []
    for evaluator in evaluators:
        try:
            result = evaluator(run_data)
            results.append(result)
        except Exception as e:
            print(f"Error in {evaluator.__name__}: {str(e)}")
    
    # Store evaluation results in the run data only if we have results
    if results:
        run_data['feedback_stats'] = {
            result.key: {
                "score": result.score,
                "comment": result.comment
            } for result in results
        }
    
    return results


def chat_response(message: str, history: List[List[str]]) -> str:
    """Process user message and return agent's response with evaluation."""
    try:
        formatted_history = []
        for human, ai in history:
            formatted_history.extend([("human", human), ("ai", ai)])
        
        run_metadata = {
            "conversation_id": str(hash(str(formatted_history))),
            "message_type": "user_input",
            "history_length": len(formatted_history)
        }
        
        # Get response from agent with run tracking
        from langchain.callbacks.tracers.langchain import LangChainTracer
        import asyncio
        
        tracer = LangChainTracer()
        response = _agent.invoke(
            {
                "input": message,
                "chat_history": formatted_history
            },
            config={
                "metadata": run_metadata,
                "tags": ["chat-interaction"],
                "callbacks": [tracer]
            }
        )
        
        # Schedule evaluations to run asynchronously
        async def process_evaluations() -> None:
            try:
                # Run evaluations
                eval_results = run_evaluations(response)
                
                # Only process evaluations if we have results
                if eval_results:
                    avg_score = sum(result.score for result in eval_results) / len(eval_results)
                    
                    run_id = tracer.latest_run.id if tracer.latest_run else None
                    if run_id:
                        # Create feedback in LangSmith
                        langsmith_client.create_feedback(
                            run_id=run_id,
                            key="accuracy",
                            score=avg_score,
                            comment=f"Average eval score: {avg_score:.2f}. " +
                                   f"Individual scores: {', '.join(f'{r.key}: {r.score:.2f}' for r in eval_results)}"
                        )
                    else:
                        print("Warning: Could not find run_id in response")
                    
                    # Log evaluation summary
                    print(f"\nEvaluation Summary (avg score: {avg_score:.2f}):")
                    for result in eval_results:
                        print(f"{result.key}: {result.score:.2f} - {result.comment}")
                else:
                    print("\nEvaluations skipped (ONLINE_EVAL=False)")
            except Exception as e:
                print(f"Error in async evaluation: {str(e)}")
        
        def run_async_eval():
            """Run async evaluation in a separate thread."""
            asyncio.run(process_evaluations())
        
        # Start evaluation in a separate thread
        import threading
        eval_thread = threading.Thread(target=run_async_eval)
        eval_thread.daemon = True  # Allow the program to exit even if thread is running
        eval_thread.start()
        
        # Return response immediately
        return response["output"]
    except Exception as e:
        return f"Error: {str(e)}"


def main() -> None:
    """Main entry point of the application."""
    ui = create_ui()
    ui.launch(share=False)


if __name__ == "__main__":
    main()
