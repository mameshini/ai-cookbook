# %% [markdown]
# # LangChain: Models, Prompts and Output Parsers
# 
# 
# ## Outline
# 
#  * Direct API calls to OpenAI
#  * API calls through LangChain:
#    * Prompts
#    * Models
#    * Output parsers

# %% [markdown]
# ## Get your [OpenAI API Key](https://platform.openai.com/account/api-keys)

# %%
#!pip install python-dotenv
#!pip install openai

# %%
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()  # read local .env file

# Initialize Azure OpenAI client
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION", "2024-10-21")

if not api_key or not endpoint:
    raise ValueError("Azure OpenAI credentials must be set in environment variables")

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint
)

# %% [markdown]
# Note: LLM's do not always produce the same results. When executing the code in your notebook, you may get slightly different answers that those in the video.

# %%
# Set Azure OpenAI model deployment name
llm_model = "gpt-4o"  # Azure OpenAI deployment name

# %% [markdown]
# ## Chat API : OpenAI
# 
# Let's start with a direct API calls to OpenAI.

# %%
def get_completion(prompt: str, model: str = llm_model) -> str:
    """Get completion from Azure OpenAI.
    
    Args:
        prompt: The input prompt
        model: The model deployment name in Azure OpenAI
        
    Returns:
        str: The model's response
    """
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return completion.choices[0].message.content or ""


# %%
get_completion("What is 1+1?")

# %%
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

# %%
style = """American English \
in a calm and respectful tone
"""

# %%
prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

print(prompt)

# %%
response = get_completion(prompt)

# %%
response

# %% [markdown]
# ## Chat API : LangChain
# 
# Let's try how we can do the same using LangChain.

# %%
!pip install -qU langchain-openai
!pip install -qU langchain

# %% [markdown]
# ### Model

# %%
from langchain_openai import AzureChatOpenAI

# %%
# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0
chat = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version="2024-10-21",
    azure_deployment=llm_model,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.0
)
chat

# %% [markdown]
# ### Prompt template

# %%
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

# %%
from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(template_string)


# %%
prompt_template.messages[0].prompt

# %%
prompt_template.messages[0].prompt.input_variables

# %%
customer_style = """American English \
in a calm and respectful tone
"""

# %%
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

# %%
customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)

# %%
print(type(customer_messages))
print(type(customer_messages[0]))

# %%
print(customer_messages[0])

# %%
# Call the LLM to translate to the style of the customer message
customer_response = chat(customer_messages)

# %%
print(customer_response.content)

# %%
service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""

# %%
service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""

# %%
service_messages = prompt_template.format_messages(
    style=service_style_pirate,
    text=service_reply)

print(service_messages[0].content)

# %%
service_response = chat(service_messages)
print(service_response.content)

# %% [markdown]
# ## Output Parsers
# 
# Let's start with defining how we would like the LLM output to look like:

# %%
{
  "gift": False,
  "delivery_days": 5,
  "price_value": "pretty affordable!"
}

# %%
customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""

# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

prompt_template = ChatPromptTemplate.from_template(review_template)
print(prompt_template)

# %%
messages = prompt_template.format_messages(text=customer_review)
chat = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    temperature=0.0,
    openai_api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key
)
response = chat(messages)
print(response.content)

# %%
type(response.content)

# %%
# You will get an error by running this line of code 
# because'gift' is not a dictionary
# 'gift' is a string
response.content.get('gift')

# %% [markdown]
# ### Parse the LLM output string into a Python dictionary

# %%
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

# %%
gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]

# %%
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# %%
format_instructions = output_parser.get_format_instructions()

# %%
print(format_instructions)

# %%
review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=review_template_2)

messages = prompt.format_messages(text=customer_review, 
                                format_instructions=format_instructions)

# %%
print(messages[0].content)

# %%
response = chat(messages)

# %%
print(response.content)

# %%
output_dict = output_parser.parse(response.content)

# %%
output_dict

# %%
type(output_dict)

# %%
output_dict.get('delivery_days')



# %%
