"""
Research Agent for Logseq Lens.
Implements the agentic research loop using LangGraph.
"""
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
import json

from config import config
from tools import get_all_tools, TOOL_MAP
from services import hybrid_search, SearchResult, build_context
from rag import get_llm, RAGResponse
from agents.prompts import RESEARCH_AGENT_PROMPT, EVALUATE_CONTEXT_PROMPT


# ---- State Definition ----

class ResearchState(TypedDict):
    """State for the research agent subgraph."""
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    search_query: str
    initial_documents: List[SearchResult]
    research_iterations: int
    max_iterations: int
    research_activity: List[str]
    sufficient: bool
    final_answer: Optional[str]
    provider: str


# ---- Nodes ----

def research_agent_node(state: ResearchState) -> Dict[str, Any]:
    """
    Research agent decides what tool to call next based on current evidence.
    """
    llm = get_llm(state["provider"])
    tools = get_all_tools()
    llm_with_tools = llm.bind_tools(tools)
    
    # Build context from research so far
    context_parts = []
    if state["initial_documents"]:
        context_parts.append("Initial retrieval results:")
        for i, doc in enumerate(state["initial_documents"], 1):
            context_parts.append(f"  {i}. [{doc.stable_id}] {doc.content[:300]}...")
    
    if state["research_activity"]:
        context_parts.append("\nResearch steps so far:")
        for activity in state["research_activity"]:
            context_parts.append(f"  - {activity}")
    
    context_str = "\n".join(context_parts) if context_parts else "No previous research."
    
    messages = [
        HumanMessage(content=RESEARCH_AGENT_PROMPT.format(
            question=state["question"],
            search_query=state["search_query"],
            context=context_str,
            iterations=state["research_iterations"],
            max_iterations=state["max_iterations"]
        ))
    ]
    
    response = llm_with_tools.invoke(messages)
    
    # Track tool calls in activity
    activity = list(state["research_activity"])
    if response.tool_calls:
        for tc in response.tool_calls:
            tool_name = tc["name"]
            args = tc.get("args", {})
            arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
            activity.append(f"{tool_name}({arg_str})")
    
    return {
        "messages": [response],
        "research_activity": activity,
        "research_iterations": state["research_iterations"] + (1 if response.tool_calls else 0)
    }


def execute_tools_node(state: ResearchState) -> Dict[str, Any]:
    """
    Execute the tools called by the research agent.
    """
    tool_node = ToolNode(get_all_tools())
    
    # Get the last AI message with tool calls
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {}
    
    # Execute tools
    result = tool_node.invoke({"messages": state["messages"]})
    
    # Format tool results as activity
    activity = list(state["research_activity"])
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            activity.append(f"Tool result: {msg.content[:200]}...")
    
    return {
        "messages": result["messages"],
        "research_activity": activity
    }


def evaluate_research_node(state: ResearchState) -> Dict[str, Any]:
    """
    Evaluate if the research has gathered sufficient evidence.
    """
    llm = get_llm(state["provider"])
    
    # Build context from all gathered evidence
    context_parts = []
    if state["initial_documents"]:
        context_parts.append("Initial retrieval:")
        for i, doc in enumerate(state["initial_documents"], 1):
            context_parts.append(f"  {i}. [{doc.stable_id}] {doc.content[:300]}...")
    
    # Extract tool results from messages
    tool_results = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            tool_results.append(msg.content)
    
    if tool_results:
        context_parts.append("\nTool results:")
        for i, result in enumerate(tool_results, 1):
            context_parts.append(f"  {i}. {result[:500]}...")
    
    context_str = "\n".join(context_parts) if context_parts else "No evidence gathered."
    
    eval_prompt = EVALUATE_CONTEXT_PROMPT.format(
        question=state["question"],
        context=context_str
    )
    
    response = llm.invoke([HumanMessage(content=eval_prompt)])
    
    # Parse response (expecting "SUFFICIENT" or "INSUFFICIENT" with reasoning)
    content = response.content.strip().upper()
    sufficient = "SUFFICIENT" in content and "INSUFFICIENT" not in content
    
    return {
        "sufficient": sufficient,
        "research_activity": state["research_activity"] + [f"Evaluation: {'Sufficient' if sufficient else 'Insufficient'}"]
    }


def should_continue_research(state: ResearchState) -> Literal["continue", "generate", "max_iterations"]:
    """
    Decide whether to continue research, generate answer, or stop due to max iterations.
    """
    if state["research_iterations"] >= state["max_iterations"]:
        return "max_iterations"
    if state["sufficient"]:
        return "generate"
    return "continue"


def generate_final_answer_node(state: ResearchState) -> Dict[str, Any]:
    """
    Generate the final answer using all gathered evidence.
    """
    from agents.prompts import FINAL_ANSWER_PROMPT
    from rag import RAGResponse
    
    llm = get_llm(state["provider"])
    structured_llm = llm.with_structured_output(RAGResponse)
    
    # Build complete context
    context_parts = []
    if state["initial_documents"]:
        context_parts.append("Initial retrieval:")
        for i, doc in enumerate(state["initial_documents"], 1):
            context_parts.append(f"  {i}. [{doc.stable_id}] {doc.content}")
    
    tool_results = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            tool_results.append(msg.content)
    
    if tool_results:
        context_parts.append("\nResearch findings:")
        for i, result in enumerate(tool_results, 1):
            context_parts.append(f"  {i}. {result}")
    
    context_str = "\n".join(context_parts) if context_parts else "No context available."
    
    response = structured_llm.invoke([
        HumanMessage(content=FINAL_ANSWER_PROMPT.format(
            question=state["question"],
            context=context_str
        ))
    ])
    
    # Extract sources from all evidence
    sources = []
    for doc in state["initial_documents"]:
        if doc.source not in sources:
            sources.append(doc.source)
    
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            # Try to extract source references from tool results
            pass  # Tool results already contain source info
    
    return {
        "final_answer": response.answer,
        "messages": [AIMessage(content=response.answer)],
        "research_activity": state["research_activity"] + ["Final answer generated"]
    }


# ---- Research Subgraph ----

def route_after_agent(state: ResearchState) -> Literal["execute_tools", "generate_final"]:
    """Route to tools if LLM called them, otherwise generate final answer."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"
    return "generate_final"


def create_research_subgraph() -> StateGraph:
    """Create the research agent subgraph."""
    workflow = StateGraph(ResearchState)
    
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("evaluate_research", evaluate_research_node)
    workflow.add_node("generate_final", generate_final_answer_node)
    
    workflow.add_edge(START, "research_agent")
    workflow.add_conditional_edges(
        "research_agent",
        route_after_agent,
        {
            "execute_tools": "execute_tools",
            "generate_final": "generate_final"
        }
    )
    workflow.add_edge("execute_tools", "evaluate_research")
    
    workflow.add_conditional_edges(
        "evaluate_research",
        should_continue_research,
        {
            "continue": "research_agent",
            "generate": "generate_final",
            "max_iterations": "generate_final"
        }
    )
    
    workflow.add_edge("generate_final", END)
    
    return workflow.compile()


# ---- Helper Functions ----

def run_research(
    question: str,
    search_query: str,
    initial_documents: List[SearchResult],
    provider: str,
    max_iterations: int = None
) -> Dict[str, Any]:
    """
    Run the research agent subgraph.
    
    Returns:
        Dict with final_answer, research_activity, sufficient
    """
    if max_iterations is None:
        max_iterations = config.max_research_iterations
    
    subgraph = create_research_subgraph()
    
    initial_state = ResearchState(
        messages=[],
        question=question,
        search_query=search_query,
        initial_documents=initial_documents,
        research_iterations=0,
        max_iterations=max_iterations,
        research_activity=["Initial hybrid search"],
        sufficient=False,
        final_answer=None,
        provider=provider
    )
    
    result = subgraph.invoke(initial_state)
    
    return {
        "answer": result.get("final_answer", "Research completed but no answer generated."),
        "research_activity": result.get("research_activity", []),
        "sufficient": result.get("sufficient", False)
    }