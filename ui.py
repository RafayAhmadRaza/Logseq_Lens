"""
Streamlit UI for Logseq Lens.
Main application interface with chat, sidebar configuration, and maintenance.
"""
import streamlit as st
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langfuse.langchain import CallbackHandler

from graph import rag_graph
from config import config
from services import get_index_stats, sync_graph, check_graph_changes, init_graph_manager, get_graph_manager
from agents import run_research
from rag import get_ollama_models
from langfuse import get_client


# Initialize Langfuse client and callback handler
langfuse = get_client()
langfuse_handler = CallbackHandler()


def format_sources(documents: List[Any]) -> List[Dict[str, Any]]:
    """Format documents for display."""
    sources = []
    for i, doc in enumerate(documents, 1):
        source_path = doc.metadata.get("source", "Unknown")
        source_name = Path(source_path).name
        # Handle both Document and SearchResult
        content = getattr(doc, 'page_content', getattr(doc, 'content', ''))
        sources.append({
            "index": i,
            "stable_id": doc.metadata.get("stable_id", ""),
            "source": source_path,
            "source_name": source_name,
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "content": content,
            "score": doc.metadata.get("search_score", doc.metadata.get("rrf_score")),
            "tags": doc.metadata.get("tags", ""),
            "page_links": doc.metadata.get("page_links", ""),
            "journal_date": doc.metadata.get("journal_date", ""),
        })
    return sources


def get_node_display_name(node_name: str) -> str:
    """Map internal node names to human-readable display names."""
    node_names = {
        "rewrite_query": "🔄 Rewriting query",
        "classify_question": "🎯 Classifying question",
        "retrieve": "🔍 Retrieving documents",
        "evaluate_context": "📊 Evaluating context",
        "research": "🔬 Deep research",
        "generate": "✍️ Generating answer",
        "generate_direct": "💬 Direct answer",
        "research_agent": "🤖 Research agent thinking",
        "execute_tools": "🔧 Executing tools",
        "evaluate_research": "📋 Evaluating research",
        "generate_final": "📝 Finalizing answer",
    }
    return node_names.get(node_name, f"⚙️ {node_name}")


def extract_prompt_from_node(node_name: str, state: Dict[str, Any]) -> str:
    """Extract prompt information for debugging from node execution context."""
    # This is a placeholder - actual prompt extraction would need to be done
    # at the point of LLM invocation inside the graph nodes
    prompts_map = {
        "rewrite_query": "Query rewrite with conversation history",
        "classify_question": "Classification prompt with user question",
        "retrieve": "Hybrid search (no LLM prompt)",
        "evaluate_context": "Context sufficiency evaluation",
        "generate": "RAG generation with context",
        "generate_direct": "Direct answer without retrieval",
        "research_agent": "Research agent with tools",
        "evaluate_research": "Research sufficiency evaluation",
        "generate_final": "Final answer synthesis",
    }
    return prompts_map.get(node_name, "Unknown prompt")


# ---- Page Configuration ----

st.set_page_config(
    page_title="Logseq Lens",
    page_icon="",
    layout="wide"
)


# ---- Session State Initialization ----

def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "graph_manager" not in st.session_state:
        st.session_state.graph_manager = init_graph_manager(config.logseq_path)
    
    if "current_graph_path" not in st.session_state:
        st.session_state.current_graph_path = config.logseq_path
    
    if "research_enabled" not in st.session_state:
        st.session_state.research_enabled = True
    
    if "top_k" not in st.session_state:
        st.session_state.top_k = config.default_top_k
    
    if "max_iterations" not in st.session_state:
        st.session_state.max_iterations = config.max_research_iterations
    
    if "last_sync_result" not in st.session_state:
        st.session_state.last_sync_result = None
    
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = config.ollama_model
    
    if "ollama_models" not in st.session_state:
        st.session_state.ollama_models = []
    
    # Generation state
    if "generating" not in st.session_state:
        st.session_state.generating = False
    
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


init_session_state()


# ---- Session State Initialization ----

def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "graph_manager" not in st.session_state:
        st.session_state.graph_manager = init_graph_manager(config.logseq_path)
    
    if "current_graph_path" not in st.session_state:
        st.session_state.current_graph_path = config.logseq_path
    
    if "research_enabled" not in st.session_state:
        st.session_state.research_enabled = True
    
    if "top_k" not in st.session_state:
        st.session_state.top_k = config.default_top_k
    
    if "max_iterations" not in st.session_state:
        st.session_state.max_iterations = config.max_research_iterations
    
    if "last_sync_result" not in st.session_state:
        st.session_state.last_sync_result = None
    
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = config.ollama_model
    
    if "ollama_models" not in st.session_state:
        st.session_state.ollama_models = []
    
    # Generation state
    if "generating" not in st.session_state:
        st.session_state.generating = False
    
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    
    # Streaming state
    if "stream_state" not in st.session_state:
        st.session_state.stream_state = {
            "active": False,
            "current_node": None,
            "node_output": None,
            "final_answer": None,
            "sources": [],
            "research_activity": [],
            "researched": False,
            "error": None,
        }


init_session_state()


# ---- Handle Pending Generation (runs before sidebar renders) ----
if st.session_state.generating and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    
    # Initialize stream state
    st.session_state.stream_state = {
        "active": True,
        "current_node": "starting",
        "node_output": None,
        "final_answer": None,
        "sources": [],
        "research_activity": [],
        "researched": False,
        "error": None,
    }
    st.rerun()


# ---- Process Streaming Generation ----
if st.session_state.stream_state.get("active", False) and not st.session_state.stream_state.get("final_answer"):
    # Prepare graph messages
    graph_messages = []
    for message in st.session_state.messages:
        if message["role"] == "user":
            graph_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            graph_messages.append(AIMessage(content=message["content"]))
    
    selected_model = st.session_state.ollama_model if st.session_state.get("provider_select") == "Ollama" else None
    
    initial_state = {
        "question": st.session_state.stream_state.get("prompt", ""),
        "search_query": "",
        "documents": [],
        "answer": None,
        "messages": graph_messages,
        "provider": st.session_state.get("provider_select", "Ollama"),
        "research_enabled": st.session_state.research_enabled,
        "logseq_path": st.session_state.current_graph_path,
        "research_iterations": 0,
        "research_activity": [],
        "context_sufficient": False,
        "researched": False,
        "needs_retrieval": True,
        "model": selected_model
    }
    
    # Store the prompt for the stream handler
    st.session_state.stream_state["prompt"] = st.session_state.stream_state.get("prompt", "")
    
    # Stream the graph execution
    try:
        config = {"callbacks": [langfuse_handler]}
        for chunk in rag_graph.stream(initial_state, config=config, stream_mode="updates"):
            # Process each node update
            for node_name, node_output in chunk.items():
                st.session_state.stream_state["current_node"] = node_name
                st.session_state.stream_state["node_output"] = node_output
                
                # Extract relevant info from node output
                if node_output and isinstance(node_output, dict):
                    if "research_activity" in node_output:
                        st.session_state.stream_state["research_activity"] = node_output["research_activity"]
                    if "documents" in node_output:
                        st.session_state.stream_state["sources"] = format_sources(node_output["documents"])
                    if "answer" in node_output and node_output["answer"]:
                        answer_obj = node_output["answer"]
                        if hasattr(answer_obj, 'answer'):
                            st.session_state.stream_state["final_answer"] = answer_obj
                            st.session_state.stream_state["sources"] = format_sources(
                                answer_obj.sources if hasattr(answer_obj, 'sources') else []
                            )
                        elif isinstance(answer_obj, dict) and "answer" in answer_obj:
                            st.session_state.stream_state["final_answer"] = answer_obj
                            st.session_state.stream_state["sources"] = format_sources(
                                answer_obj.get("sources", [])
                            )
                    if "researched" in node_output:
                        st.session_state.stream_state["researched"] = node_output["researched"]
            
            st.session_state.stream_state["active"] = False
            
    except Exception as e:
        st.session_state.stream_state["error"] = str(e)
        st.session_state.stream_state["active"] = False
    
    # Generation complete - save to history
    if st.session_state.stream_state.get("final_answer"):
        answer_obj = st.session_state.stream_state["final_answer"]
        if hasattr(answer_obj, 'answer'):
            answer_text = answer_obj.answer
            sources = st.session_state.stream_state.get("sources", [])
            researched = st.session_state.stream_state.get("researched", False)
            research_activity = st.session_state.stream_state.get("research_activity", [])
            sufficient_context = getattr(answer_obj, 'sufficient_context', True)
        else:
            answer_text = answer_obj.get("answer", "")
            sources = st.session_state.stream_state.get("sources", [])
            researched = st.session_state.stream_state.get("researched", False)
            research_activity = st.session_state.stream_state.get("research_activity", [])
            sufficient_context = answer_obj.get("sufficient_context", True)
        
        assistant_msg = {
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "research_activity": research_activity,
            "researched": researched,
            "sufficient_context": sufficient_context
        }
        st.session_state.messages.append(assistant_msg)
    elif st.session_state.stream_state.get("error"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error: {st.session_state.stream_state['error']}"
        })
    
    # Reset stream state
    st.session_state.generating = False
    st.session_state.stream_state = {
        "active": False,
        "current_node": None,
        "node_output": None,
        "final_answer": None,
        "sources": [],
        "research_activity": [],
        "researched": False,
        "error": None,
    }
    st.rerun()


# ---- Helper Functions ----

def validate_graph_path(path: str) -> tuple[bool, str]:
    """Validate a graph path."""
    expanded = os.path.expanduser(path)
    path_obj = Path(expanded)
    
    if not path_obj.exists():
        return False, f"Path does not exist: {expanded}"
    
    if not path_obj.is_dir():
        return False, f"Path is not a directory: {expanded}"
    
    # Check for Logseq indicators
    has_pages = (path_obj / "pages").exists()
    has_journals = (path_obj / "journals").exists()
    has_config = (path_obj / "logseq" / "config.edn").exists() or (path_obj / "config.edn").exists()
    
    if not (has_pages or has_journals or has_config):
        return False, f"Directory does not appear to be a Logseq graph (missing pages/, journals/, or config.edn)"
    
    return True, ""


def apply_graph_path(path: str):
    """Apply a new graph path."""
    is_valid, error = validate_graph_path(path)
    if not is_valid:
        st.sidebar.error(error)
        return False
    
    manager = st.session_state.graph_manager
    success, error, graph = manager.validate_and_set(path)
    
    if success:
        st.session_state.current_graph_path = os.path.expanduser(path)
        st.session_state.messages = []  # Clear chat on graph change
        st.sidebar.success(f"Graph changed to: {graph.name}")
        st.rerun()
    else:
        st.sidebar.error(error)
    return False


def run_sync(full: bool = False):
    """Run sync and store result."""
    with st.spinner("Syncing..." if not full else "Full rebuild..."):
        result = sync_graph(full=full)
        st.session_state.last_sync_result = result
    st.rerun()


# ---- Sidebar ----

with st.sidebar:
    st.header("Logseq Lens")
    
    # LLM Provider
    st.subheader("LLM Provider")
    provider = st.selectbox(
        "Provider",
        ["Ollama", "OpenRouter"],
        index=0,
        key="provider_select",
        disabled=st.session_state.generating
    )
    
    # Model selection
    if provider == "Ollama":
        st.caption("Runs locally via Ollama")
        
        # Fetch models button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Refresh", key="refresh_models_btn", use_container_width=True, disabled=st.session_state.generating):
                with st.spinner("Fetching models..."):
                    st.session_state.ollama_models = get_ollama_models()
        
        # Show model dropdown if models are available
        if st.session_state.ollama_models:
            # Ensure current selection is in the list, otherwise default to first
            current_model = st.session_state.ollama_model
            if current_model not in st.session_state.ollama_models:
                current_model = st.session_state.ollama_models[0]
                st.session_state.ollama_model = current_model
            
            selected_model = st.selectbox(
                "Model",
                st.session_state.ollama_models,
                index=st.session_state.ollama_models.index(current_model),
                key="ollama_model_select",
                disabled=st.session_state.generating
            )
            st.session_state.ollama_model = selected_model
        else:
            st.warning("No Ollama models found. Click Refresh or check if Ollama is running.")
            st.session_state.ollama_model = config.ollama_model
    else:
        st.caption("Uses cloud model via OpenRouter")
        st.session_state.ollama_model = config.openrouter_model
    
    st.divider()
    
    # Research Settings
    st.subheader("Research")
    research_enabled = st.checkbox(
        "Enable deep research",
        value=st.session_state.research_enabled,
        help="Allow agentic research with multiple search iterations for complex questions",
        disabled=st.session_state.generating
    )
    st.session_state.research_enabled = research_enabled
    
    if research_enabled:
        max_iter = st.number_input(
            "Max research iterations",
            min_value=1,
            max_value=10,
            value=st.session_state.max_iterations,
            key="max_iter_input",
            disabled=st.session_state.generating
        )
        st.session_state.max_iterations = max_iter
    
    st.divider()
    
    # Logseq Graph Selection
    st.subheader("Logseq Graph")
    
    current_graph = st.session_state.graph_manager.get_current()
    st.text_input(
        "Graph path",
        value=st.session_state.current_graph_path,
        key="graph_path_input",
        help="Path to your Logseq graph directory",
        disabled=st.session_state.generating
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply Graph", key="apply_graph_btn", use_container_width=True, disabled=st.session_state.generating):
            apply_graph_path(st.session_state.graph_path_input)
    with col2:
        if st.button("Validate", key="validate_graph_btn", use_container_width=True, disabled=st.session_state.generating):
            is_valid, error = validate_graph_path(st.session_state.graph_path_input)
            if is_valid:
                st.sidebar.success("Valid Logseq graph")
            else:
                st.sidebar.error(error)
    
    st.caption(f"Current: {current_graph.name}")
    st.caption(f"Path: {current_graph.path}")
    
    st.divider()
    
    # Maintenance
    st.subheader("Maintenance")
    
    # Index Status
    stats = get_index_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pages", stats["page_count"])
        st.metric("Chunks", stats["chunk_count"])
    with col2:
        st.metric("Vectors", stats["vector_count"])
        st.metric("Synced", "Yes" if stats["is_synced"] else "No")
    
    if stats["last_sync"]:
        st.caption(f"Last sync: {stats['last_sync'][:19].replace('T', ' ')}")
    else:
        st.caption("Last sync: Never")
    
    # Sync buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Resync", key="resync_btn", use_container_width=True, disabled=st.session_state.generating):
            run_sync(full=False)
    with col2:
        if st.button("Rebuild", key="rebuild_btn", use_container_width=True, type="secondary", disabled=st.session_state.generating):
            if st.session_state.get("confirm_rebuild", False):
                run_sync(full=True)
                st.session_state.confirm_rebuild = False
            else:
                st.session_state.confirm_rebuild = True
                st.warning("Click again to confirm full rebuild")
    
    if st.session_state.last_sync_result:
        result = st.session_state.last_sync_result
        if result.errors:
            st.error(f"Errors: {', '.join(result.errors)}")
        else:
            st.success(f"Added: {result.added}, Updated: {result.updated}, Deleted: {result.deleted} ({result.duration_seconds:.1f}s)")
    
    st.divider()
    
    # Settings
    st.subheader("Settings")
    top_k = st.number_input(
        "Top K results",
        min_value=1,
        max_value=20,
        value=st.session_state.top_k,
        key="top_k_input",
        disabled=st.session_state.generating
    )
    st.session_state.top_k = top_k
    
    st.caption(f"Embedding: {config.embedding_model}")
    st.caption(f"Ollama model: {config.ollama_model}")
    st.caption(f"OpenRouter model: {config.openrouter_model}")


# ---- Main Chat Interface ----

st.title("Logseq Lens")
st.caption("Ask questions about your Logseq notes.")

# ---- Streaming Activity Panel ----
if st.session_state.stream_state.get("active", False):
    with st.container():
        st.markdown("---")
        st.subheader("🔄 Generation in Progress")
        
        current_node = st.session_state.stream_state.get("current_node")
        if current_node:
            node_display = get_node_display_name(current_node)
            st.markdown(f"**Current step:** {node_display}")
            
            # Show node-specific details
            node_output = st.session_state.stream_state.get("node_output")
            if node_output and isinstance(node_output, dict):
                if "search_query" in node_output:
                    st.caption(f"Search query: {node_output['search_query']}")
                if "needs_retrieval" in node_output:
                    st.caption(f"Needs retrieval: {node_output['needs_retrieval']}")
                if "documents" in node_output and node_output["documents"]:
                    st.caption(f"Retrieved {len(node_output['documents'])} documents")
                if "research_activity" in node_output:
                    activity = node_output["research_activity"]
                    if activity:
                        st.caption(f"Latest: {activity[-1]}")
                if "context_sufficient" in node_output:
                    st.caption(f"Context sufficient: {node_output['context_sufficient']}")
                if "researched" in node_output:
                    st.caption(f"Researched: {node_output['researched']}")
        
        # Show current research activity
        research_activity = st.session_state.stream_state.get("research_activity", [])
        if research_activity:
            with st.expander("Research Activity", expanded=True):
                for i, step in enumerate(research_activity, 1):
                    st.markdown(f"{i}. {step}")
        
        st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            sources = message["sources"]
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for src in sources:
                        st.markdown(f"**{src['index']}. {src['source_name']}**")
                        st.caption(f"Chunk {src['chunk_index']}" + (f" | Score: {src['score']:.4f}" if src['score'] else ""))
                        if src.get("tags"):
                            st.caption(f"Tags: {src['tags']}")
                        if src.get("page_links"):
                            st.caption(f"Links: {src['page_links']}")
                        if src.get("journal_date"):
                            st.caption(f"Journal: {src['journal_date']}")
                        with st.expander("View content"):
                            st.text(src["content"][:2000])
        
        # Show research activity
        if message["role"] == "assistant" and "research_activity" in message:
            activity = message["research_activity"]
            if activity and len(activity) > 1:  # More than just initial search
                with st.expander("Research performed"):
                    for step in activity:
                        st.markdown(f"- {step}")


# ---- Developer Prompt Inspection Panel ----
with st.expander("🔧 Developer: Prompt Inspection", expanded=False):
    st.caption("Shows prompts sent to LLM at each node during the last generation")
    
    if st.session_state.stream_state.get("node_output"):
        node_name = st.session_state.stream_state.get("current_node")
        node_output = st.session_state.stream_state.get("node_output")
        
        if node_name:
            st.markdown(f"**Last node:** {get_node_display_name(node_name)}")
            
            # Show node-specific prompt info
            if node_name == "rewrite_query" and "search_query" in node_output:
                st.markdown("**Search query:**")
                st.code(node_output["search_query"])
            
            elif node_name == "classify_question":
                st.markdown("**Classification:**")
                st.code(f"needs_retrieval: {node_output.get('needs_retrieval', 'N/A')}")
            
            elif node_name == "retrieve" and "documents" in node_output:
                docs = node_output.get("documents", [])
                st.markdown(f"**Retrieved {len(docs)} documents:**")
                for i, doc in enumerate(docs[:3], 1):
                    st.caption(f"{i}. {doc.stable_id}: {doc.content[:100]}...")
                if len(docs) > 3:
                    st.caption(f"... and {len(docs) - 3} more")
            
            elif node_name in ["generate", "generate_direct", "generate_final"]:
                if "answer" in node_output:
                    answer = node_output["answer"]
                    if hasattr(answer, 'answer'):
                        st.markdown("**Model output:**")
                        st.code(answer.answer)
                    elif isinstance(answer, dict):
                        st.markdown("**Model output:**")
                        st.code(answer.get("answer", "N/A"))
            
            elif node_name in ["evaluate_context", "evaluate_research"]:
                st.markdown("**Evaluation:**")
                st.code(f"Sufficient: {node_output.get('context_sufficient', node_output.get('sufficient', 'N/A'))}")
            
            elif node_name == "research_agent":
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            st.markdown("**Tool calls:**")
                            for tc in msg.tool_calls:
                                st.code(f"{tc['name']}({tc.get('args', {})})")
        
        # Show full state for debugging
        st.markdown("---")
        st.markdown("**Full stream state (JSON):**")
        import json
        # Filter out non-serializable objects
        debug_state = {k: v for k, v in st.session_state.stream_state.items() 
                      if k not in ["node_output", "final_answer"]}
        st.code(json.dumps(debug_state, default=str, indent=2), language="json")

# Chat input
if prompt := st.chat_input("Ask your Logseq notes...", disabled=st.session_state.generating):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Set generation state and rerun
    st.session_state.generating = True
    st.session_state.pending_prompt = prompt
    st.session_state.stream_state["prompt"] = prompt
    st.rerun()