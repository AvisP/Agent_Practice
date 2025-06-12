# https://python.langchain.com/docs/how_to/tool_calling/
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load enviornemnt variables from .env file
load_dotenv()

llm = ChatOpenAI(model="llama3.2", api_key="ollama", base_url="http://localhost:11434/v1", temperature=0)
# model="llama3.1:8b-text-q8_0"
# model="llama3.2"
@tool
def magic_function(input: int) -> int:
    """Applies a magic functio to an input"""
    return input+2

tools = [magic_function]
query = "What is the value of magic_function(3)"

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