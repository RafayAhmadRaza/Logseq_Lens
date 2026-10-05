"""
MCP Server for Logseq Lens.
Exposes Logseq capabilities as MCP resources and tools.
"""
import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from config import config
from services import (
    keyword_search, vector_search, hybrid_search, get_document, get_full_page,
    search_by_tag, search_by_page_title, get_index_stats, sync_graph, check_graph_changes,
    SearchResult, load_logseq_graph, parse_documents, chunk_parsed_pages
)
from services.graph_config import get_graph_manager, GraphManager
from services.logseq_parser import find_related_pages, build_page_index, ParsedPage


# Initialize FastMCP server
mcp = FastMCP("Logseq Lens")


# ---- Resource Definitions ----

@mcp.resource("logseq://graph/info")
def graph_info() -> str:
    """Get information about the currently loaded Logseq graph."""
    manager = get_graph_manager()
    current = manager.get_current()
    
    info = {
        "name": current.name,
        "path": current.path,
        "page_count": current.page_count,
        "chunk_count": current.chunk_count,
        "vector_count": current.vector_count,
        "last_sync": current.last_sync,
        "is_synced": current.is_synced,
    }
    return json.dumps(info, indent=2)


@mcp.resource("logseq://graph/index-status")
def index_status() -> str:
    """Get detailed index status."""
    stats = get_index_stats()
    return json.dumps(stats, indent=2)


@mcp.resource("logseq://page/{stable_id}")
def page_resource(stable_id: str) -> str:
    """Get a specific page/chunk by stable ID."""
    result = get_document(stable_id)
    if not result:
        return json.dumps({"error": f"Document not found: {stable_id}"}, indent=2)
    
    return json.dumps({
        "stable_id": result.stable_id,
        "source": result.source,
        "chunk_index": result.chunk_index,
        "content": result.content,
        "metadata": result.metadata
    }, indent=2)


@mcp.resource("logseq://pages/by-tag/{tag}")
def pages_by_tag_resource(tag: str) -> str:
    """Get pages containing a specific tag."""
    results = search_by_tag(tag, k=50)
    return json.dumps([{
        "stable_id": r.stable_id,
        "source": r.source,
        "chunk_index": r.chunk_index,
        "preview": r.content[:200]
    } for r in results], indent=2)


# ---- Tool Definitions ----

class SearchLogseqInput(BaseModel):
    query: str = Field(description="Search query")
    method: str = Field(default="hybrid", description="Search method: hybrid, keyword, or vector")
    k: int = Field(default=5, description="Maximum results")


@mcp.tool()
def search_logseq(query: str, method: str = "hybrid", k: int = 5) -> str:
    """
    Search the Logseq knowledge base.
    
    Methods:
    - hybrid: Combines keyword and semantic search (default, best for most queries)
    - keyword: Exact term matching (best for proper nouns, specific terms)
    - vector: Semantic similarity (best for conceptual queries)
    """
    if method == "keyword":
        results = keyword_search(query, k)
    elif method == "vector":
        results = vector_search(query, k)
    else:
        results = hybrid_search(query, k)
    
    return json.dumps([{
        "stable_id": r.stable_id,
        "source": r.source,
        "chunk_index": r.chunk_index,
        "content": r.content,
        "score": r.score
    } for r in results], indent=2)


class GetPageInput(BaseModel):
    stable_id: str = Field(description="Stable ID of the page/chunk to retrieve")


@mcp.tool()
def get_page(stable_id: str) -> str:
    """Retrieve a specific page/chunk by its stable ID."""
    result = get_document(stable_id)
    if not result:
        return json.dumps({"error": f"Document not found: {stable_id}"})
    
    return json.dumps({
        "stable_id": result.stable_id,
        "source": result.source,
        "chunk_index": result.chunk_index,
        "content": result.content,
        "metadata": result.metadata
    }, indent=2)


class GetFullPageInput(BaseModel):
    source_path: str = Field(description="Path to the source markdown file")


@mcp.tool()
def get_full_page(source_path: str) -> str:
    """Retrieve all chunks for a given source page (full page content)."""
    results = get_full_page(source_path)
    if not results:
        return json.dumps({"error": f"No chunks found for {source_path}"})
    
    return json.dumps([{
        "stable_id": r.stable_id,
        "source": r.source,
        "chunk_index": r.chunk_index,
        "content": r.content
    } for r in results], indent=2)


class GetRelatedPagesInput(BaseModel):
    stable_id: str = Field(description="Stable ID of the page to find related pages for")
    max_results: int = Field(default=10, description="Maximum related pages to return")


@mcp.tool()
def get_related_pages(stable_id: str, max_results: int = 10) -> str:
    """Find pages related to a given page via links, tags, and backlinks."""
    doc = get_document(stable_id)
    if not doc:
        return json.dumps({"error": f"Document not found: {stable_id}"})
    
    # Note: This requires the full parsed graph. For now, return a helpful message.
    return json.dumps({
        "message": "Related page search requires full graph parsing. Use search_logseq with terms from this page to find related content.",
        "source_page": doc.source,
        "source_preview": doc.content[:300]
    }, indent=2)


class SearchTagsInput(BaseModel):
    tag: str = Field(description="Tag to search for (without #)")
    k: int = Field(default=10, description="Maximum results")


@mcp.tool()
def search_tags(tag: str, k: int = 10) -> str:
    """Search for pages containing a specific tag."""
    results = search_by_tag(tag, k)
    return json.dumps([{
        "stable_id": r.stable_id,
        "source": r.source,
        "chunk_index": r.chunk_index,
        "preview": r.content[:200]
    } for r in results], indent=2)


@mcp.tool()
def get_index_status() -> str:
    """Get the current index status."""
    stats = get_index_stats()
    return json.dumps(stats, indent=2)


class ResyncGraphInput(BaseModel):
    full: bool = Field(default=False, description="Force full rebuild")


@mcp.tool()
def resync_graph(full: bool = False) -> str:
    """Trigger a resync of the Logseq graph."""
    result = sync_graph(full=full)
    return json.dumps({
        "added": result.added,
        "updated": result.updated,
        "deleted": result.deleted,
        "duration_seconds": result.duration_seconds,
        "errors": result.errors
    }, indent=2)


@mcp.tool()
def check_graph_changes() -> str:
    """Check if the Logseq graph has changed since last sync."""
    has_changes, message = check_graph_changes()
    return json.dumps({
        "has_changes": has_changes,
        "message": message
    }, indent=2)


class SetGraphPathInput(BaseModel):
    path: str = Field(description="Path to the Logseq graph directory")


@mcp.tool()
def set_graph_path(path: str) -> str:
    """Set the active Logseq graph path."""
    manager = get_graph_manager()
    success, error, graph = manager.validate_and_set(path)
    if not success:
        return json.dumps({"success": False, "error": error})
    
    return json.dumps({
        "success": True,
        "graph": graph.to_dict()
    }, indent=2)


# ---- Server Entry Point ----

def run_mcp_server():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    run_mcp_server()