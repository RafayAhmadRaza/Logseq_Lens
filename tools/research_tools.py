"""
LangGraph-compatible tools for the research agent.
Wraps the retrieval services as callable tools.
"""
from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from services import (
    keyword_search,
    vector_search,
    hybrid_search,
    get_document,
    get_full_page,
    search_by_tag,
    search_by_page_title,
    get_index_stats,
    SearchResult,
)
from services.logseq_parser import find_related_pages, build_page_index
from services.indexer import sync_graph, check_graph_changes
from services.graph_config import get_graph_manager


# ---- Input Models ----

class KeywordSearchInput(BaseModel):
    query: str = Field(description="Search query string")
    k: int = Field(default=5, description="Maximum number of results")


class VectorSearchInput(BaseModel):
    query: str = Field(description="Search query string")
    k: int = Field(default=5, description="Maximum number of results")


class HybridSearchInput(BaseModel):
    query: str = Field(description="Search query string")
    k: int = Field(default=5, description="Maximum number of results")


class GetDocumentInput(BaseModel):
    stable_id: str = Field(description="Stable identifier (source:chunk_index)")


class GetFullPageInput(BaseModel):
    source_path: str = Field(description="Path to the source markdown file")


class SearchByTagInput(BaseModel):
    tag: str = Field(description="Tag to search for (without #)")
    k: int = Field(default=10, description="Maximum number of results")


class SearchByPageInput(BaseModel):
    title: str = Field(description="Page title to search for")
    k: int = Field(default=10, description="Maximum number of results")


class FindRelatedPagesInput(BaseModel):
    stable_id: str = Field(description="Stable ID of the page to find related pages for")
    max_results: int = Field(default=10, description="Maximum number of related pages")


class SyncGraphInput(BaseModel):
    full: bool = Field(default=False, description="Force full rebuild instead of incremental")


# ---- Tool Output Helpers ----

def format_search_results(results: List[SearchResult], tool_name: str) -> str:
    """Format search results for the agent."""
    if not results:
        return f"{tool_name}: No results found."
    
    lines = [f"{tool_name}: Found {len(results)} result(s)."]
    for i, r in enumerate(results, 1):
        source_name = Path(r.source).name
        preview = r.content[:200].replace('\n', ' ')
        lines.append(f"  {i}. [{r.stable_id}] {source_name} (chunk {r.chunk_index})")
        lines.append(f"     Preview: {preview}...")
        if r.score is not None:
            lines.append(f"     Score: {r.score:.4f}")
    return "\n".join(lines)


def format_document_result(result: Optional[SearchResult], tool_name: str) -> str:
    """Format a single document result."""
    if not result:
        return f"{tool_name}: Document not found."
    
    source_name = Path(result.source).name
    return (
        f"{tool_name}: Retrieved document [{result.stable_id}]\n"
        f"Source: {source_name} (chunk {result.chunk_index})\n"
        f"Content:\n{result.content}"
    )


from pathlib import Path


# ---- Tools ----

@tool("keyword_search", args_schema=KeywordSearchInput, return_direct=False)
def keyword_search_tool(query: str, k: int = 5) -> str:
    """
    Search the SQLite FTS5 keyword index.
    Best for exact term matching, proper nouns, specific keywords.
    Returns structured results with source, chunk, and matching content.
    """
    results = keyword_search(query, k)
    return format_search_results(results, "Keyword Search")


@tool("vector_search", args_schema=VectorSearchInput, return_direct=False)
def vector_search_tool(query: str, k: int = 5) -> str:
    """
    Search the Chroma vector database using semantic similarity.
    Best for conceptual queries, paraphrased questions, finding related ideas.
    Returns structured results with similarity scores.
    """
    results = vector_search(query, k)
    return format_search_results(results, "Vector Search")


@tool("hybrid_search", args_schema=HybridSearchInput, return_direct=False)
def hybrid_search_tool(query: str, k: int = 5) -> str:
    """
    Hybrid search combining keyword and vector search using Reciprocal Rank Fusion.
    Best default search - combines exact matching with semantic understanding.
    Returns ranked results from both indexes.
    """
    results = hybrid_search(query, k)
    return format_search_results(results, "Hybrid Search")


@tool("get_document", args_schema=GetDocumentInput, return_direct=False)
def get_document_tool(stable_id: str) -> str:
    """
    Retrieve a specific document/chunk by its stable ID.
    Use when you need to see the full content of a specific search result.
    """
    result = get_document(stable_id)
    return format_document_result(result, "Get Document")


@tool("get_full_page", args_schema=GetFullPageInput, return_direct=False)
def get_full_page_tool(source_path: str) -> str:
    """
    Retrieve all chunks for a given source page (full page content).
    Use when you need to read an entire page rather than individual chunks.
    """
    results = get_full_page(source_path)
    if not results:
        return f"Get Full Page: No chunks found for {source_path}"
    
    lines = [f"Get Full Page: Retrieved {len(results)} chunk(s) from {Path(source_path).name}"]
    for r in results:
        lines.append(f"\n--- Chunk {r.chunk_index} ---")
        lines.append(r.content)
    return "\n".join(lines)


@tool("search_by_tag", args_schema=SearchByTagInput, return_direct=False)
def search_by_tag_tool(tag: str, k: int = 10) -> str:
    """
    Search for pages containing a specific tag (#tag).
    Use when the question relates to a specific topic tag.
    """
    results = search_by_tag(tag, k)
    return format_search_results(results, f"Tag Search (#{tag})")


@tool("search_by_page", args_schema=SearchByPageInput, return_direct=False)
def search_by_page_tool(title: str, k: int = 10) -> str:
    """
    Search for pages by title/filename.
    Use when looking for a specific page by name.
    """
    results = search_by_page_title(title, k)
    return format_search_results(results, f"Page Title Search ({title})")


@tool("find_related_pages", args_schema=FindRelatedPagesInput, return_direct=False)
def find_related_pages_tool(stable_id: str, max_results: int = 10) -> str:
    """
    Find pages related to a given page via links, tags, and backlinks.
    Use to explore the knowledge graph around a specific page.
    """
    # Get the source page
    doc = get_document(stable_id)
    if not doc:
        return f"Find Related Pages: Document {stable_id} not found."
    
    # We need the parsed pages to find related - this is a limitation
    # For now, return a message indicating this needs the full graph
    return (
        f"Find Related Pages: This tool requires the full parsed graph to be loaded. "
        f"Source page: {Path(doc.source).name}. "
        f"Use hybrid_search with terms from this page to find related content."
    )


@tool("get_index_status", return_direct=False)
def get_index_status_tool() -> str:
    """
    Get the current index status: page count, chunk count, vector count, last sync.
    """
    stats = get_index_stats()
    lines = [
        "Index Status:",
        f"  Graph: {stats['graph_name']} ({stats['graph_path']})",
        f"  Pages: {stats['page_count']}",
        f"  Chunks: {stats['chunk_count']}",
        f"  Vectors: {stats['vector_count']}",
        f"  Last Sync: {stats['last_sync'] or 'Never'}",
        f"  Synced: {'Yes' if stats['is_synced'] else 'No'}",
    ]
    return "\n".join(lines)


@tool("resync_graph", args_schema=SyncGraphInput, return_direct=False)
def resync_graph_tool(full: bool = False) -> str:
    """
    Trigger a resync of the Logseq graph.
    Use 'full=true' to force a complete rebuild.
    """
    from services.indexer import sync_graph
    result = sync_graph(full=full)
    
    lines = [
        f"Resync {'Full' if full else 'Incremental'} Complete:",
        f"  Added: {result.added}",
        f"  Updated: {result.updated}",
        f"  Deleted: {result.deleted}",
        f"  Duration: {result.duration_seconds:.1f}s",
    ]
    if result.errors:
        lines.append(f"  Errors: {', '.join(result.errors)}")
    return "\n".join(lines)


@tool("check_graph_changes", return_direct=False)
def check_graph_changes_tool() -> str:
    """
    Check if the Logseq graph has changed since last sync.
    """
    has_changes, message = check_graph_changes()
    return f"Graph Changes: {'Yes' if has_changes else 'No'} - {message}"


# ---- Tool Registry ----

RESEARCH_TOOLS = [
    hybrid_search_tool,
    keyword_search_tool,
    vector_search_tool,
    get_document_tool,
    get_full_page_tool,
    search_by_tag_tool,
    search_by_page_tool,
    find_related_pages_tool,
    get_index_status_tool,
    resync_graph_tool,
    check_graph_changes_tool,
]

TOOL_MAP = {tool.name: tool for tool in RESEARCH_TOOLS}


def get_tool(name: str):
    """Get a tool by name."""
    return TOOL_MAP.get(name)


def get_all_tools() -> List:
    """Get all research tools."""
    return RESEARCH_TOOLS