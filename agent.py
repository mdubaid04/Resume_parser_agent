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
from langgraph.prebuilt import InjectedState
from langchain_core.messages import HumanMessage,ToolMessage
from pymongo import AsyncMongoClient
from langchain_core.runnables import RunnableConfig
from bson import ObjectId
from bson.errors import InvalidId
from langgraph.types import Command
import json,httpx
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()


ADZUNA_URL = "https://api.adzuna.com/v1/api/jobs/in/search/1"
ADZUNA_ID = os.getenv("ADZUNA_ID")
ADZUNA_API_KEY = os.getenv("ADZUNA_API_KEY")


@tool
async def fetch_jobs(
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Fetch job listings that match the user's target roles.

    IMPORTANT: This tool depends on the output of retrive_user_resume_from_db.
    Always call retrive_user_resume_from_db FIRST and wait for its result.
    Only after the resume profile is loaded, call this tool in a SEPARATE,
    LATER step. NEVER call both tools in the same step or in parallel.

    Takes no arguments. Target roles are read automatically from the loaded profile.
    Use this when the user asks for jobs, openings, or vacancies."""

    profile = state.get("profile") or {}
    roles = profile.get("target_roles") or []
    if isinstance(roles, str):          # agar string aa gayi to list bana do
        roles = [roles]

    if not roles:
        return Command(update={"messages": [ToolMessage(
            "Error: profile not loaded or has no target_roles. "
            "Call retrive_user_resume_from_db first.",
            tool_call_id=tool_call_id)]})

    where = config["configurable"].get("location", "Noida")
    all_jobs, seen = [], set()

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            for role in roles:
                r = await client.get(ADZUNA_URL, params={
                    "app_id": ADZUNA_ID,
                    "app_key": ADZUNA_API_KEY,
                    "what": role,
                    "where": where,
                    "results_per_page": 5,
                    "max_days_old": 7,
                })
                r.raise_for_status()

                for j in r.json().get("results", []):
                    if j["id"] in seen:
                        continue
                    seen.add(j["id"])
                    all_jobs.append({
                        "role_searched": role,
                        "title": j.get("title"),
                        "company": (j.get("company") or {}).get("display_name"),
                        "description": j.get("description"),
                        "salary_min": j.get("salary_min"),
                        "salary_max": j.get("salary_max"),
                        "salary_is_predicted": j.get("salary_is_predicted"),
                        "location": (j.get("location") or {}).get("display_name"),
                        "url": j.get("redirect_url"),
                    })
        llm_view = [{**job, "description": (job["description"] or "")[:200]} for job in all_jobs]

        return Command(update={
            "jobs": all_jobs,
            "messages": [ToolMessage(json.dumps(llm_view, default=str), tool_call_id=tool_call_id)],
        })
    except Exception as e:
        return Command(update={"messages": [ToolMessage(
            f"Error: {e!r}", tool_call_id=tool_call_id)]})

class AgentState(TypedDict):
  messages: Annotated[list[BaseMessage],add_messages]
  profile:dict


@tool
async def retrive_user_resume_from_db( config:RunnableConfig,tool_call_id:Annotated[str,InjectedToolCallId])->Command:
   """Load the current user's resume profile (skills, experience, education,
    target roles) from the database into the agent state.

    Call this FIRST whenever the user's resume or jobs are needed.
    Must complete before fetch_jobs is called. Takes no arguments."""

   conf=config["configurable"]["collection"]
   id=config["configurable"]["resume_id"]
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
tools=[retrive_user_resume_from_db,fetch_jobs]
tool_binded_llm=router_llm.bind_tools(tools,parallel_tool_calls=False)
tool_node=ToolNode(tools=tools)


prompt=ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful job-search assistant.\n\n"
     "TOOL RULES:\n"
     "- Call tools ONE AT A TIME. Never call two tools in the same step.\n"
     "- To find jobs: first call retrive_user_resume_from_db and wait for its result. "
     "Only in the NEXT step call fetch_jobs.\n"
     "- If the resume was already fetched earlier in this conversation, "
     "do not fetch it again. Call fetch_jobs directly.\n"
     "- fetch_jobs and retrive_user_resume_from_db take no arguments.\n\n"
     "After fetch_jobs returns, summarise the best matches briefly: "
     "title, company, location, salary (if available) and the link.\n\n"
     "For general questions (e.g. 'what is machine learning') answer directly "
     "without calling any tool."),
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





agent=workflow.compile(checkpointer=MemorySaver())

png_bytes=agent.get_graph().draw_mermaid_png()

with open("agent.png","wb") as f:
    f.write(png_bytes)

  