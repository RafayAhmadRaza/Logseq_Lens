"""
Main LangGraph workflow for Logseq Lens.
Implements: query rewrite -> retrieve -> evaluate -> (research agent) -> generate
"""
from typing import TypedDict, Annotated, List, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import json

from config import config
from services import hybrid_search, SearchResult, build_context
from rag import get_llm, RAGResponse
from agents import run_research
from agents.prompts import QUERY_REWRITE_PROMPT, EVALUATE_INITIAL_CONTEXT_PROMPT

load_dotenv()

langfuse = get_client()
langfuse_handler = CallbackHandler()


# ---- State Definition ----

class RAGState(TypedDict):
    question: str
    search_query: str
    documents: List[SearchResult]
    answer: RAGResponse
    messages: Annotated[List[BaseMessage], add_messages]
    provider: str
    research_enabled: bool
    logseq_path: str
    research_iterations: int
    research_activity: List[str]
    context_sufficient: bool
    researched: bool


# ---- Nodes ----

def rewrite_query_node(state: RAGState) -> dict:
    """Rewrite the user's question into a standalone search query."""
    history = "\n".join(
        f"{message.type}: {message.content}"
        for message in state["messages"]
    )
    
    llm = get_llm(state["provider"])
    query_rewriter = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
    
    search_query = query_rewriter.invoke({
        "history": history,
        "question": state["question"]
    })
    
    return {"search_query": search_query}


def retrieve_node(state: RAGState) -> dict:
    """Perform initial hybrid retrieval."""
    documents = hybrid_search(state["search_query"], k=config.default_top_k)
    return {
        "documents": documents,
        "research_activity": ["Initial hybrid search"]
    }


def evaluate_context_node(state: RAGState) -> dict:
    """Evaluate if initial retrieval is sufficient."""
    # If research is disabled, always proceed to generate
    if not state.get("research_enabled", False):
        return {"context_sufficient": True, "researched": False}
    
    # If no documents found, definitely need research
    if not state["documents"]:
        return {"context_sufficient": False, "researched": True}
    
    # Use LLM to evaluate
    llm = get_llm(state["provider"])
    
    context_str = build_context(state["documents"])
    
    eval_prompt = EVALUATE_INITIAL_CONTEXT_PROMPT.format(
        question=state["question"],
        search_query=state["search_query"],
        context=context_str
    )
    
    response = llm.invoke([HumanMessage(content=eval_prompt)])
    content = response.content.strip().upper()
    sufficient = "SUFFICIENT" in content and "INSUFFICIENT" not in content
    
    return {
        "context_sufficient": sufficient,
        "researched": not sufficient
    }


def route_after_evaluation(state: RAGState) -> Literal["research", "generate"]:
    """Route to research agent or direct generation."""
    if state.get("researched", False) and state.get("research_enabled", False):
        return "research"
    return "generate"


def research_node(state: RAGState) -> dict:
    """Run the agentic research subgraph."""
    max_iterations = config.max_research_iterations
    
    research_result = run_research(
        question=state["question"],
        search_query=state["search_query"],
        initial_documents=state["documents"],
        provider=state["provider"],
        max_iterations=max_iterations
    )
    
    return {
        "answer": research_result.get("answer"),
        "research_activity": research_result.get("research_activity", []),
        "context_sufficient": research_result.get("sufficient", False),
        "researched": True
    }


def generate_node(state: RAGState) -> dict:
    """Generate final answer (either from research or direct RAG)."""
    # If research already produced an answer, use it
    if state.get("answer") and isinstance(state["answer"], str):
        # Research agent produced a string answer, wrap in RAGResponse
        sources = []
        for doc in state["documents"]:
            if doc.source not in sources:
                sources.append(doc.source)
        
        answer_obj = RAGResponse(
            answer=state["answer"],
            sufficient_context=state.get("context_sufficient", True),
            sources=sources
        )
        return {
            "answer": answer_obj,
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=state["answer"])
            ]
        }
    
    # Direct RAG generation
    context = build_context(state["documents"])
    
    rag_chain = get_llm(state["provider"]).with_structured_output(RAGResponse)
    
    # Use the same prompt as in rag.py
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an assistant that answers questions using the user's Logseq notes.

Use ONLY the provided context.

Rules:
- Do not invent information.
- If the context does not contain enough information, say so.
- Set sufficient_context to true only when the context contains enough information to answer.
- Include only sources that were actually used.
- The answer should be concise and directly answer the question.

Context:

{context}
"""
        ),
        (
            "human",
            "{question}"
        )
    ])
    
    chain = prompt | rag_chain
    answer = chain.invoke({
        "context": context,
        "question": state["question"]
    })
    
    return {
        "answer": answer,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer.answer)
        ]
    }


# ---- Graph Construction ----

graph = StateGraph(RAGState)

graph.add_node("rewrite_query", rewrite_query_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("evaluate_context", evaluate_context_node)
graph.add_node("research", research_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "rewrite_query")
graph.add_edge("rewrite_query", "retrieve")
graph.add_edge("retrieve", "evaluate_context")

graph.add_conditional_edges(
    "evaluate_context",
    route_after_evaluation,
    {
        "research": "research",
        "generate": "generate"
    }
)

graph.add_edge("research", "generate")
graph.add_edge("generate", END)

rag_graph = graph.compile()


# ---- Main for testing ----

if __name__ == "__main__":
    messages = []
    
    while True:
        question = input("\nYou: ")
        
        if question.lower() in {"exit", "quit"}:
            break
        
        result = rag_graph.invoke({
            "question": question,
            "search_query": "",
            "documents": [],
            "answer": None,
            "messages": messages,
            "provider": "Ollama",
            "research_enabled": True,
            "logseq_path": config.logseq_path,
            "research_iterations": 0,
            "research_activity": [],
            "context_sufficient": False,
            "researched": False
        },
        config={'callbacks': [langfuse_handler]}
        )
        
        messages = result["messages"]
        
        print("\nSearch query:", result["search_query"])
        print("\nResearch activity:")
        for activity in result.get("research_activity", []):
            print(f"  - {activity}")
        print("\nAssistant:", result["answer"].answer)
        print("Sufficient context:", result["answer"].sufficient_context)
        print("Sources:", result["answer"].sources)
        print("Researched:", result.get("researched", False))