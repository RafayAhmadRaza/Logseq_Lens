"""
Streamlit UI for Logseq Lens.
Main application interface with chat, sidebar configuration, and maintenance.
"""
import streamlit as st
import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

from graph import rag_graph
from config import config
from services import get_index_stats, sync_graph, check_graph_changes, init_graph_manager, get_graph_manager
from agents import run_research
from rag import get_ollama_models


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


# ---- Handle Pending Generation (runs before sidebar renders) ----
if st.session_state.generating and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    
    # Prepare graph messages
    graph_messages = []
    for message in st.session_state.messages:
        if message["role"] == "user":
            graph_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            graph_messages.append(AIMessage(content=message["content"]))
    
    # Generate response
    try:
        selected_model = st.session_state.ollama_model if st.session_state.get("provider_select") == "Ollama" else None
        
        result = rag_graph.invoke({
            "question": prompt,
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
            "model": selected_model
        })
        
        answer = result["answer"]
        sources = format_sources(result.get("documents", []))
        research_activity = result.get("research_activity", [])
        researched = result.get("researched", False)
        
        # Store assistant message with metadata
        assistant_msg = {
            "role": "assistant",
            "content": answer.answer,
            "sources": sources,
            "research_activity": research_activity,
            "researched": researched,
            "sufficient_context": answer.sufficient_context
        }
        st.session_state.messages.append(assistant_msg)
        
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error: {str(e)}"
        })
    
    # Generation complete
    st.session_state.generating = False
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


# Chat input
if prompt := st.chat_input("Ask your Logseq notes...", disabled=st.session_state.generating):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Set generation state and rerun
    st.session_state.generating = True
    st.session_state.pending_prompt = prompt
    st.rerun()