import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from tools import *
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name = "Jacob Hackathon"
)

@mcp.tool()
def product_recommendation(text):
    """use this function to fetch product list that suits the user needs. this function will return the list of recommendation product based on the user description
    input:
    1. text: str = a string that describe the user company.
    
    output:
    output format will be in dictionary where the key is 
    1. product_id
    2. product_description
    3. relevant_score: measure the relevancy of the product with the user description. the score is [0, 1] where 1 is the most relevant product"""

    return product_rec(text)

@mcp.tool()
def registration_procedure():
    """use this function if you want to explain the registration process to the user"""
    return registration()

def chat_registration_procedure():
    pass

@mcp.tool()
def document_list(lob: str, legal_code: str):
    f"""use this function if the user ask about what document that they need to submit if they want to join DOKU.
    input:
    1. lob: user line of business. The value are {legal_doc["business_line"].keys()}
    2. legal_code: user legal business code. The value are {legal_doc["legal_code"].keys()}
    """
    return document_map(lob, legal_code)

if __name__ == "__main__":
    mcp.run(transport="stdio")