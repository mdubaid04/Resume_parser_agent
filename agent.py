import os

from langgraph.graph import StateGraph,START,END, add_messages
from pydantic import BaseModel, Field
from langgraph.prebuilt.tool_node import tools_condition,ToolNode
from typing import Literal, TypedDict,Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from dotenv import load_dotenv
from langchain_core.tools import tool,InjectedToolCallId
import asyncio
from langchain_core.messages import HumanMessage,ToolMessage
from pymongo import AsyncMongoClient
from langchain_core.runnables import RunnableConfig
from bson import ObjectId
from bson.errors import InvalidId
from langgraph.types import Command
import json


load_dotenv()


adzuna_url="https://api.adzuna.com/v1/api/jobs/in/search/1"
adzuna_params={
    "app_id": os.getenv("ADZUNA_ID"),
    "app_key": os.getenv("ADZUNA_API_KEY"),
    "what": "python developer",
    "where":"bangalore",
    "results_per_page": 5,
    "content-type": "application/json",
    "max_days_old": 7,
}

class AgentState(TypedDict):
  messages: Annotated[list[BaseMessage],add_messages]
  profile:dict


@tool
async def retrive_user_resume_from_db( config:RunnableConfig,tool_call_id:Annotated[str,InjectedToolCallId])->Command:
   """"Get the current user's resume profile (skills, experience, education, etc.) from the database.
       Use this tool when the user asks for job or anything that needs their resume."""

   conf=config["configurable"]["collection"]
   id=config["configurable"]["id"]
   try:
    obj_id=ObjectId(id)
   except (InvalidId, TypeError ,KeyError):
    return {"error":"Invalid ID format missing resume id. Please provide a valid ObjectId string."}
   try:
      doc=await conf.find_one({"_id":obj_id},{"_id":0,"profile":1})
      if not doc or "profile" not in doc:
        return Command(update={
            "messages":[ToolMessage("Error:No resume found for the provided ID.",tool_call_id=tool_call_id)]
        })
      return Command(update={
            "profile":doc["profile"],
            "messages":[ToolMessage(json.dumps(doc,default=str),tool_call_id=tool_call_id)]
        })
   except Exception as e:
    return Command(update={
        "messages":[ToolMessage(f"Error: {str(e)}",tool_call_id=tool_call_id)]
    })


router_llm=ChatGoogleGenerativeAI(model="gemma-4-31b-it")
tools=[retrive_user_resume_from_db]
tool_binded_llm=router_llm.bind_tools(tools)
tool_node=ToolNode(tools=tools)


prompt=ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant. Use tools when needed, otherwise answer the user's question directly."),
    MessagesPlaceholder("messages")
  ])

router_chain=prompt|tool_binded_llm

async def router(state:AgentState):
  response=await router_chain.ainvoke({"messages": state["messages"]})
  return {"messages":[response]}

  



workflow=StateGraph(AgentState)

workflow.add_node("router",router)
workflow.add_node("tool_node",tool_node)


workflow.add_edge(START,"router")
workflow.add_conditional_edges("router",tools_condition ,{"tools":"tool_node",END:END})
workflow.add_edge("tool_node","router")





agent=workflow.compile()

png_bytes=agent.get_graph().draw_mermaid_png()

with open("agent.png","wb") as f:
    f.write(png_bytes)

  