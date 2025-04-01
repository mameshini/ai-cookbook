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
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain_community.chat_models import BedrockChat
from langchain.chains import RetrievalQA
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
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

# Initialize AWS clients
region_name = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
kb_id = os.getenv("KNOWLEDGE_BASE_ID", "Y5UHFTTWAT")

bedrock_config = Config(
    region_name=region_name,
    connect_timeout=120,
    read_timeout=120,
    retries={"max_attempts": 0}
)

bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=region_name,
    config=bedrock_config
)

bedrock_agent_client = boto3.client(
    "bedrock-agent-runtime",
    region_name=region_name,
    config=bedrock_config
)


class OpenMeteoInput(BaseModel):
    """Input schema for OpenMeteo weather API."""
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")


@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> str:
    """Fetch current temperature for given latitude and longitude in Fahrenheit"""
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
def search_rag(query: str) -> str:
    """Search financial statements and real estate information using RAG."""
    # Initialize Bedrock chat model
    llm = BedrockChat(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        client=bedrock_client,
        region_name=region_name,
        model_kwargs={"temperature": 0.0, "max_tokens": 2000}
    )

    # Initialize retriever with explicit client
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,
        client=bedrock_agent_client,
        region_name=region_name,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}}
    )

    # Initialize QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Get response
    response = qa_chain.invoke({"query": query})
    return response["result"]


@tool
def get_current_datetime() -> str:
    """Get the current date and time in a human-readable format."""
    current_time = datetime.datetime.now()
    return current_time.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")


def setup_agent() -> AgentExecutor:
    """Set up the LangChain agent with tools and model."""
    tools = [get_current_temperature, search_wikipedia, get_current_datetime, search_rag]
    set_debug(True)  # Enable debug mode for detailed logging
    functions = [format_tool_to_openai_function(f) for f in tools]
    
    model = AzureChatOpenAI(
        openai_api_version=api_version,
        azure_deployment=model_name,
        temperature=0
    ).bind(functions=functions)
    
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    
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
        memory=memory
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
        
        response = _agent.invoke({
            "input": message,
            "chat_history": formatted_history
        })
        return response["output"]
    except Exception as e:
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


def main() -> None:
    """Main entry point of the application."""
    ui = create_ui()
    ui.launch(share=False)


if __name__ == "__main__":
    main()
