"""
Visualizer atemps to create matplot charts and save them as tempfiles.
Code is executed using PythonREPL - so use with caution.
Further more - if it keeps on failing, enable DEBUG loging to scan for generated python code - chances are you're missing some lib the generated code is trying to run.

Alternative (and safer) approach could be using https://quickchart.io/, but that's for another time
"""
from typing import Annotated, Literal, List
from langgraph.constants import END

import ast
from langchain_experimental.utilities import PythonREPL

from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

from state_schemas import ApplicationState
from config.models import llm

import logging

_CONST_VISUALIZER_ERROR = "nothing to visualize"

_logger = logging.getLogger(__name__)

_repl = PythonREPL()

@tool
def _python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."]):
    """Use this to execute python code."""
    
    _logger.log(logging.INFO, "Visualizer code: \n", code)
    
   # print(logging.INFO, "Visualizer code: \n", code)
    
    try:
        result = _repl.run(code)
    except BaseException as e:
        #print(f"error _python_repl_tool failed running visualizations: \n", repr(e))
        _logger.log(logging.INFO, "_python_repl_tool failed running visualizations: \n", repr(e))
        return f"Error: {repr(e)}"

    return f"Successfully executed:\nstdout: {result}"

code_agent = create_react_agent(llm, tools=[_python_repl_tool], state_modifier=f"""
    You are a developer tasked with visualizing the data.
    Please provide the python code to visualize the data using matplotlib.pyplot.
    IMPORTANT - your code always starts with:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('agg') 
    
    Image should be saved in the tempfile. When the image is saved to the tempfile, the script should print(tempfile path).
     
    Return just the python code that can be evaluated using exec(). No code fences, triple ticks or language identifiers.
    
    If success, return an array of of paths to the tempfiles generated.
    
    If there is nothing to visualized - return "{_CONST_VISUALIZER_ERROR}"
    
    """, debug=_logger.getEffectiveLevel() < logging.WARNING)

def visualizer_node(state: ApplicationState) -> Command[Literal["__end__"]]:
    """ Vizualizer langgraph node.
        returns: Array of paths to the image saved in the system tmp dir or an error string
    """
    
    documents_message = HumanMessage(content="Here are the raw contents of the data: \n" + "\n---------\n".join(state["documents"]), name="documents")
    
    result = code_agent.invoke({"messages": state["messages"][-1:] + [documents_message]}, debug=_logger.getEffectiveLevel() < logging.WARNING)
    
    response_content: str = result["messages"][-1].content;
    
    if _CONST_VISUALIZER_ERROR not in response_content:
        try:
            # response_content will be a string representing python list. Use ast to parce it, since JSON will not make it
            images =  ast.literal_eval(response_content)
        except Exception as e:
            _logger.log(logging.ERROR, str(e))
            images = _CONST_VISUALIZER_ERROR
    else:
        images = _CONST_VISUALIZER_ERROR
        
    # return Command(
    #     update={
    #         "messages": AIMessage("Here are the visualizations"),
    #         "images_or_error": images
    #     },
    #     goto=END
    # )
    
    return { "images_or_error": images }

   
    # Alternative solution for using quickchart.io.
    # Initial generated links to quickchart seems to be quite broken and llm.with_structured_output seems to fail to parse simple List[str] where strings contains '}]' at the end.
    # Futhermore, on a first shot openAI 4o fails to create valid links, but it can then easily fix itself.
    # So another few hours which I don't have and a second reActive agent would be in place
    # ------------------------------------------------------------------------------------------------------------------    
    
    # class _ImageLinks(BaseModel):
    #     """List of image links to quickchart.io"""
    #     images: List[str] = Field(description="List of links to quickchart.io generated images")
    #     error: str = Field(description="Description if no visualization is possible")  
    
    # system_message = SystemMessage(content=f"""You are a developer tasked with visualizing the data.
    
    # Please provide links to https://quickchart.io/ to reflect provided data.
    # There can be multiple images generated for the data.
    
    # If there is nothing to visualized - return "{_CONST_VISUALIZER_ERROR}""")
    
    # result = llm.with_structured_output(_ImageLinks).invoke([system_message] + state["messages"] + [documents_message])
    # result.images = [url.replace("+data", ",data") for url in result.images]
    # if len(result.images):
    #     return { "images_or_error": result.images }
    # else:
    #     return { "images_or_error": result.error }