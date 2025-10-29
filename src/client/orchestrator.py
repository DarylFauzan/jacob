# import package to create agent
import re
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage

# import tools from mcp
import asyncio, json, time
from servers.tools import system_message
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

memory = {}

# use client to fetch tools
async def get_tools():
    print("🛠️fetching tools and prompt...")
    client = MultiServerMCPClient(
        {
            "Jacob Hackathon": {
                "command": "python",
                "args": ["src/servers/hackathon_jacob.py"],
                "transport": "stdio",
            }
    },
    )
    tools = await client.get_tools()
    print("fetching tools success")
    return tools

model_name = "gpt-4o-mini"

# create langgraph agent
async def create_agent(model_name = "gpt-4o-mini"):
    tools = await get_tools()
    llm = ChatOpenAI(model=model_name, temperature = 1.2).bind_tools(tools)

    # create the llm node
    def call_model(state:MessagesState):
        system_prompt = SystemMessage(content = system_message)

        response = llm.invoke([system_prompt] + state["messages"])

        return {"messages": response}

    # Create react agent
    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    return graph

graph = asyncio.run(create_agent())
print("Agent has been initialized")

# build the orchestrator
async def orchestrator(user_id, question, model_name="gpt-4o-mini"):
    start_time = time.time()

    # initialize memory for user
    if user_id not in memory:
        memory[user_id] = [HumanMessage(content = question)]
    else:
        memory[user_id].append(HumanMessage(content = question))

    # invoke the agent
    response = await graph.ainvoke({"messages": memory[user_id]})
    memory[user_id] = response["messages"]

    answer = response["messages"][-1].content
    log = {'response': answer, 'token': response["messages"][-1].response_metadata, "duration": time.time() - start_time}
    print(f"{log}")

    return answer