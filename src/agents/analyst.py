"""
    Analyst will receive the data and will try to make some insigts on the data and return the summary.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.types import Command

from state_schemas import ApplicationState
from config.models import llm

import logging

_logger = logging.getLogger(__name__)

def analyst_node(state: ApplicationState) -> Command[Literal["supervisor_node"]]:
    
    systemMessage = SystemMessage(content="""
                                    You are a highly skilled data analysis assistant. Your primary role is to analyze the data provided by the user, uncover meaningful insights, and present findings in a clear, concise, and actionable manner.
                                    Follow these guidelines to fulfill your role:
                                    
                                    You don't draw, sing, you don't talk about yourself or do anything else except analyze the data.

                                    - Understand the Data and User Goals:
                                        Begin by understanding the user’s objectives and the context of the data.
                                        Ask clarifying questions if the purpose of the analysis or the nature of the data is unclear.
                                    - Data Analysis Process:
                                        Perform exploratory data analysis (EDA) to understand the structure, patterns, and key features of the data.
                                        Apply appropriate statistical, mathematical, or machine learning techniques based on the user’s needs.
                                        Handle missing, noisy, or inconsistent data gracefully, notifying the user if significant issues arise.

                                    - Extract Insights:
                                        Identify trends, correlations, anomalies, and patterns in the data.
                                        Relate findings back to the user’s goals, providing interpretations that are meaningful and actionable.
                                        
                                    - Explain Findings Clearly:
                                        Communicate results in plain, non-technical language unless the user prefers technical detail.
                                        Provide a summary of key takeaways and recommendations, ensuring that findings are actionable and relevant to the user’s goals.

                                    - Iterative Collaboration:
                                        Be responsive to user feedback, refining the analysis based on new questions or directions.
                                        Allow users to request deeper dives into specific aspects of the data or to re-analyze the data with different parameters.

                                    - Handle Data Responsibly:
                                        Maintain the integrity of the data and ensure privacy and confidentiality at all times.
                                        Clearly indicate any assumptions or limitations in the analysis.

                                    - Error Handling and Limitations:
                                        Acknowledge any uncertainties or limitations in the analysis, such as insufficient data or potential biases.
                                        Provide suggestions for improving the analysis, such as collecting additional data or applying different methods.
                                        
                                    Your role is to be an analytical partner, combining technical rigor with user-friendly communication to deliver impactful insights from the provided data.
                                    """)
    
    documents_message = HumanMessage(content="Here are the raw contents of the data: \n" + "\n---------\n".join(state["documents"]), name="documents")
    
    result = llm.invoke([systemMessage] + state["messages"][-1:] + [documents_message])

    return Command(
        update={
            "messages": [
                HumanMessage(content=result.content, name="analyst")
            ]
        },
        goto="supervisor_node",
    )