from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
# Load enviornemnt variables from .env file
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    print("API key for Open AI not found")

from langchain.chat_models import init_chat_model
llm = ChatOpenAI(model="llama3.2", api_key="ollama", base_url="http://localhost:11434/v1", temperature=0)

# llm = init_chat_model("gpt-4o-mini", model_provider="openai")

from typing import Optional

from pydantic import BaseModel, Field


# Pydantic
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


structured_llm = llm.with_structured_output(Joke)

result = structured_llm.invoke("Tell me a joke about cats")

print(result)