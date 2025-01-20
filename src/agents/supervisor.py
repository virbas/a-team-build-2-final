"""
Supervisor will dedicate tasks to the workes.
Supervisor does not write to the state, just dedicates the tasks (except a special case where the question is too simple)
"""
from pydantic import Field
from typing import Final, TypedDict, Literal, Union

from langgraph.constants import END
from langgraph.types import Command
from langchain_core.messages import AIMessage

from config.models import llm
from state_schemas import ApplicationState

import logging;

_logger = logging.getLogger(__name__)

agents = Literal["retriever_node", "analyst_node", "visualizer_node", "__end__"]


# _CONST_NEEDS_CLARIFICATION = "TOO_SIMPLE"

_CONST_SUPERVISOR_SYSTEM_PROMPT: Final[str] = f"""
    You are a supervisor tasked with managing a conversation between the
    following agents: {agents}. Given the following user request,
    respond with the worker to act next. Each worker will perform a
    task and respond with their results and status.
    
    Agents have access to iii.org (Insurance Information Institute) database about historical auto insurance data.
    
    The data must be first retrieved and analyzed and only then it can be visualized.
    
    Data is visualized ONLY if the user has asked for it.
    
    When all the agents are finished with their job - respond with {END}.
    """

class GraphRouter(TypedDict):
    f"""Worker to route to next and the reason why this worker was chosen"""

    reason: str = Field(description="A short explanation why this worker was needed")
    next: Union[agents] = Field(description="next worker name")

def supervisor_node(state: ApplicationState) -> Command[agents]:
    
    system_message =  {"role": "system", "content": _CONST_SUPERVISOR_SYSTEM_PROMPT}
    messages = [system_message] + state["messages"]

    # llm = config.get('configurable', {}).get("llm")
    #print(f"Supervisor got ", state["messages"][-1].content)
    response = llm.with_structured_output(GraphRouter).invoke(messages)
    #print("supervisor response", response)
    goto = response["next"]
    
    # # Special case where the user question is too simple. For example what's 2+2
    # # Todo craft a better response message. Right now it's too system-like-answer
    # if goto == _CONST_NEEDS_CLARIFICATION:
    #     return Command(
    #         update={
    #             "messages": AIMessage(response["reason"])
    #         },
    #         goto=END
    #     )

    return Command(goto=goto)