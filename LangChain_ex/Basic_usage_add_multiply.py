from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os

# Load enviornemnt variables from .env file
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    print("API key for Open AI not found")

llm = ChatOpenAI(model="llama3.2", api_key="ollama", base_url="http://localhost:11434/v1", temperature=0)
# llm = init_chat_model("gpt-4o-mini", model_provider="openai")
#  model="llama3.1:8b-text-q8_0"
# model="llama3.2"
@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]

query = query = "What is 3 * 12? Also, what is 11 + 49?"

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

result = agent_executor.invoke({"input": query})

print(result)

llm_with_tools = llm.bind_tools(tools)
result2 = llm_with_tools.invoke(query).tool_calls

print(result2)
