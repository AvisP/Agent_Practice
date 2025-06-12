from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
# Load enviornemnt variables from .env file
load_dotenv()

from typing import Union
from typing import Optional
from pydantic import BaseModel, Field

class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


class ConversationalResponse(BaseModel):
    """Respond in a conversational manner. Be kind and helpful."""

    response: str = Field(description="A conversational response to the user's query")


class FinalResponse(BaseModel):
    final_output: Union[Joke, ConversationalResponse]

if not os.environ.get("OPENAI_API_KEY"):
    print("API key for Open AI not found")

from langchain.chat_models import init_chat_model
llm = ChatOpenAI(model="llama3.2", api_key="ollama", base_url="http://localhost:11434/v1", temperature=0)


structured_llm = llm.with_structured_output(FinalResponse)

result_joke = structured_llm.invoke("Tell me a joke about cats") # Test with Joke
print(result_joke)

result_normal =  structured_llm.invoke("How are you today?") 
print(result_normal)

### To stream
for chunk in structured_llm.stream("Tell me a joke about cats"):
    print(chunk)