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
    """Fetch current temperature for given coordinates in Fahrenheit."""
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


def setup_agent() -> AgentExecutor:
    """Set up the LangChain agent with tools and model."""
    tools = [get_current_temperature, search_wikipedia]
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


def chat_response(message: str, history: List[List[str]]) -> str:
    """Process user message and return agent's response."""
    try:
        # Convert Gradio chat history to format suitable for memory
        formatted_history = []
        for human, ai in history:
            formatted_history.extend([("human", human), ("ai", ai)])
        
        agent = setup_agent()
        response = agent.invoke({
            "input": message,
            "chat_history": formatted_history
        })
        return response["output"]
    except Exception as e:
        return f"Error: {str(e)}"


def create_ui() -> gr.Interface:
    """Create and configure the Gradio UI."""
    interface = gr.ChatInterface(
        fn=chat_response,
        title="AI Assistant",
        description="Chat with an AI assistant that can check weather and search Wikipedia",
        theme="soft",
        examples=[
            "What's the weather in San Francisco?",
            "Tell me about the history of artificial intelligence",
            "What's the current temperature in London?",
        ]
    )
    return interface


def main() -> None:
    """Main entry point of the application."""
    ui = create_ui()
    ui.launch(share=False)


if __name__ == "__main__":
    main()
