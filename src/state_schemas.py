
from typing import List, Union
from pydantic import Field

from config.models import llm

from langgraph.prebuilt.chat_agent_executor import AgentState

class ApplicationState(AgentState):
    documents: List[str] = Field(description="raw file contents of the documents")
    images_or_error: Union[List[str], str] = Field(description="array of image paths returned from the visualizer agent or error string why it wasn't generated")