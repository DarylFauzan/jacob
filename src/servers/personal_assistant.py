import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from tools import *
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name = "Personal Assistant"
)

@mcp.tool()
def daryl_cv(question:str):
    """This is a function to retrieve information about Muhammad Daryl Fauzan. 
       To use this tool, you just need to pass the question you want to ask."""
    
    return fetch_cv(question)

@mcp.prompt(name = "system message")
def system_message(text: str = None) -> str:
    return f"""
            You are Muhamad Daryl Fauzan personal assistant. 
            Your role is to help user to answer things that they want to know about Daryl and run demo of one of Daryl engine. 
                                        
            You will be given tools:
            1. daryl_info
            Please use the tools based on their functionalities. 
                                                                        
            You are not allowed to answer any question that are out of context.
            You are only allowed to give users information about Daryl based on the information that you retrieve. 
            If the user ask something and Daryl does not provide the information, politely say that they don't have access to those information.
            If the user ask something that out of the context, please refuse it politely.
            """

if __name__ == "__main__":
    mcp.run(transport="stdio")