#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

import argparse
import logging
import os

import langchain;
import langchain.globals
from langgraph.graph import START, StateGraph
from langgraph.errors import GraphRecursionError

import state_schemas

from typing import Dict, Union
from agents.supervisor import supervisor_node
from agents.visualizer import visualizer_node
from agents.retriever import retriever_node
from agents.analyst import analyst_node

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
langchain.globals.set_verbose(logger.getEffectiveLevel() < logging.WARNING)

builder = StateGraph(state_schemas.ApplicationState)
builder.add_node(supervisor_node)
builder.add_node(retriever_node)
builder.add_node(visualizer_node)
builder.add_node(analyst_node)

builder.add_edge(START, "supervisor_node");

app = builder.compile(debug=logger.getEffectiveLevel() < logging.WARNING);

def query(query: str, initialState: state_schemas.ApplicationState = None ) -> Union[state_schemas.ApplicationState, GraphRecursionError]:
    """
    ask a question to ask our system.
    Args: 
        query: str - that's the prompt
        initiaState: ApplicationState - path to pass to the graph as an "initial". However, we can use it intead of checkpointers and memory.
                                        For eaxmple we will pass a previous chat as initialState from the UI to continue with the user conversation.
    """
    
    if initialState is None: 
        initialState =  {
                "messages": [("user", query)],
                "documents": [],
                "images_or_error": [],
        }
    else:
        initialState["messages"].append(("user", query))
    
    config = {"recursion_limit": 30} 
    try:
        state = app.invoke(initialState, config=config, stream_mode="values")
    except GraphRecursionError as e:
        logger.error("Langgraph recursion error")
        return e;
    
    return state

def process_document(path: str):
    """
    Processes the file and embeds it into the vectore base
    
    Args:
        path: full path to the file to read from.
    """
    
    from agents import document_processor
    
    result = document_processor.insert_file_into_vector(path)
    logger.info(f"Inserting {path} result: \n {result}")
    
    return result;
    
    
def process_directotry(directory: str):
    """
        Process and vectorize the whole directory. 
        File extensions or directory location is not checked. So please be carefull 
    """
    
    from agents import document_processor
    
    for file in os.listdir(directory):        
        result = process_document(os.path.join(directory, file))
        
        # Yeah, we simply print this one. Enabling debugging would produce too much noise
        print(f"File {file} inserted with result: {result}")
        
    update_summary();

 

def update_summary():
    """
    We maintain the overall short summary on what our vector state conains.
    Call update_summary() when new docs are added.
    """
    from agents import document_processor
    logger.info("Updating documents overview")
    document_processor.update_overview()

    
def main():
    parser = argparse.ArgumentParser(description='Insurance Data Analysis Pipeline')
    parser.add_argument('--insert-file', type=str, help='Path to data file for processing')
    parser.add_argument('--insert-directory', type=str, help='Path to directory with file for processing')
    parser.add_argument('--query', type=str, help='Analysis query to run')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--update-summary', action='store_true', help='Update database summary file')

    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        langchain.globals.set_verbose(logger.getEffectiveLevel() < logging.WARNING)

    if args.query:
        state = query(args.query)
        for message in state["messages"]:
            message.pretty_print();
        return;
        
    if args.insert_file:
        process_document(args.insert_file)
        return;
    
    if args.insert_directory:
        process_directotry(args.insert_directory)
        return;
    
    if args.update_summary:
        update_summary()
        return;
        
if __name__ == "__main__":
    main()