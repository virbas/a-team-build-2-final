"""
Retriever will atempt to retrieve the information from the underlying DB.
The reAct agent will try to reason and retrieve different data for several times before returning the data.

After the similar results are retieved from the DB, retriever will read the content of the original files from the file storage and update the state with them
so that the downstream nodes can make use of the full data.
"""
from typing import Annotated, Literal

from langchain.tools import tool
from langchain.chains import RetrievalQAWithSourcesChain

from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage, HumanMessage

from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

from state_schemas import ApplicationState
from config.models import llm
from config.db import vector_store

from agents.document_processor import get_source_contents

import logging

_logger = logging.getLogger(__name__)

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,
                                            retriever=vector_store.as_retriever(),
                                            return_source_documents=False)

@tool()
def _data_retrieval(query: Annotated[str, "A text to perform a search in the vector database"], tool_call_id: Annotated[str, InjectedToolCallId]):
    """Use this tool to search the database."""
    result = chain.invoke(query, debug=_logger.getEffectiveLevel() < logging.WARNING)
    documents = [get_source_contents(source) for source in result["sources"].split(',')]
    
    return Command(
        update={
            "messages": [ToolMessage(content=str(result), tool_call_id=tool_call_id)],
            "documents": documents
        }
    )

_retrieval_agent = create_react_agent(llm,
                                    tools=[_data_retrieval],
                                    state_schema=ApplicationState, 
                                    state_modifier="""
                                    You are an intelligent search assistant specialized in querying vector-based databases to retrieve relevant information about iii.org (Insurance Information Institute) historical auto insurance data.

                                    You're only focus is to retieve data is to understand user inputs, interpret their intent, and translate the query into a form suitable for searching a vector store.
                                    
                                    The retrieved data will be forwarded further for processing.
                                    You don't draw, sing, you don't talk about yourself or do anything else except search for data and return it.
                                    
                                    Try to be clever about questions. For example:
                                    
                                    User: Retrieve data for the current year
                                    Answer: Here's data for "FILL_IN_THE_CURRENT_YEAR"
                                    
                                    Follow these guidelines to fulfill your role:

                                    - Understand User Intent:
                                        Accurately interpret user queries, identifying key entities, topics, or semantic goals.
                                        Support natural language queries and handle ambiguous inputs gracefully by asking clarifying questions.
                                    
                                    - Construct Vector Queries:
                                        Convert user inputs into similarity-friendly prompts.
                                        Leverage semantic similarity to find the most relevant results in the vector store.
                                    
                                    - Search and Retrieval:
                                        Perform efficient searches in the vector store, focusing on retrieving relevant, accurate, and contextually appropriate results.
                                        Rank results based on similarity scores or other metadata as appropriate.
                                    
                                    - Provide Results:
                                        Present retrieved results in a clear and organized manner.
                                    
                                    - Minimize unnecessary back-and-forth communication by providing comprehensive responses when possible.
                                        Use plain language and avoid jargon unless the user specifies otherwise.
                                    
                                    - Error Handling:
                                        Gracefully handle situations where no results are found by suggesting alternative queries or asking for clarification.
                                        Inform the user if a query cannot be processed or if additional context is required.
                                        You are integrated with a vector store and have access to an embedding model for query transformation.
                                        Your responses should prioritize clarity, relevance, and user satisfaction at all times.
                                    """,
                                        
                                        debug=_logger.getEffectiveLevel() < logging.WARNING)


def retriever_node(state: ApplicationState) -> Command[Literal["supervisor_node"]]:
    """
    langgraph node
    """
    result = _retrieval_agent.invoke(state, debug=_logger.getEffectiveLevel() < logging.WARNING)

    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="retriever")
            ],
            "documents": result.get("documents", [])
        },
        goto="supervisor_node",
    )