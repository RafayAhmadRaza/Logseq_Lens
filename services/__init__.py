"""
Services package for Logseq Lens.
"""
from services.retrieval import (
    keyword_search,
    vector_search,
    hybrid_search,
    get_document,
    get_full_page,
    search_by_tag,
    search_by_page_title,
    get_index_stats,
    setup_keyword_db,
    SearchResult,
    build_context,
    content_hash,
    get_chroma_db,
    get_keyword_db,
    keyword_add,
    keyword_update,
    keyword_delete,
)

from services.indexer import (
    load_logseq_graph,
    parse_documents,
    chunk_parsed_pages,
    sync_graph,
    sync_incremental,
    sync_full,
    check_graph_changes,
    SyncResult,
)

from services.graph_config import (
    GraphConfig,
    GraphManager,
    init_graph_manager,
    get_graph_manager,
)

from services.logseq_parser import (
    ParsedPage,
    parse_logseq_page_enhanced,
    parse_documents_enhanced,
    extract_page_links,
    extract_tags,
    find_related_pages,
    build_page_index,
    get_all_page_titles,
    find_pages_by_title,
    find_pages_by_tag,
)

__all__ = [
    # Retrieval
    "keyword_search",
    "vector_search",
    "hybrid_search",
    "get_document",
    "get_full_page",
    "search_by_tag",
    "search_by_page_title",
    "get_index_stats",
    "setup_keyword_db",
    "SearchResult",
    "build_context",
    "content_hash",
    "get_chroma_db",
    "get_keyword_db",
    "keyword_add",
    "keyword_update",
    "keyword_delete",
    # Indexer
    "load_logseq_graph",
    "parse_documents",
    "chunk_parsed_pages",
    "sync_graph",
    "sync_incremental",
    "sync_full",
    "check_graph_changes",
    "SyncResult",
    # Graph Config
    "GraphConfig",
    "GraphManager",
    "init_graph_manager",
    "get_graph_manager",
    # Logseq Parser
    "ParsedPage",
    "parse_logseq_page_enhanced",
    "parse_documents_enhanced",
    "extract_page_links",
    "extract_tags",
    "find_related_pages",
    "build_page_index",
    "get_all_page_titles",
    "find_pages_by_title",
    "find_pages_by_tag",
]