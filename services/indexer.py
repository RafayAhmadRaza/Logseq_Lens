"""
Index synchronization service for Logseq graphs.
Handles incremental and full sync of Logseq pages to vector and keyword indexes.
"""
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import config
from services.graph_config import get_graph_manager
from services.logseq_parser import parse_logseq_page_enhanced, ParsedPage
from services.retrieval import (
    get_chroma_db, get_keyword_db, setup_keyword_db,
    keyword_add, keyword_update, keyword_delete,
    content_hash, get_embeddings
)


@dataclass
class SyncResult:
    """Result of a sync operation."""
    added: int = 0
    updated: int = 0
    deleted: int = 0
    errors: List[str] = None
    duration_seconds: float = 0.0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def load_logseq_graph(graph_path: str) -> List[Dict]:
    """
    Load all markdown files from a Logseq graph.
    
    Args:
        graph_path: Path to the Logseq graph directory
        
    Returns:
        List of dicts with 'content' and 'source' keys
    """
    graph = Path(graph_path)
    documents = []
    
    if not graph.exists():
        return documents
    
    # Search in pages/ and journals/ directories, plus root
    search_dirs = []
    if (graph / "pages").exists():
        search_dirs.append(graph / "pages")
    if (graph / "journals").exists():
        search_dirs.append(graph / "journals")
    # Also check root for any .md files
    search_dirs.append(graph)
    
    for search_dir in search_dirs:
        for file in search_dir.rglob("*.md"):
            # Skip backup files
            if "bak" in file.parts:
                continue
            # Skip hidden files
            if any(part.startswith('.') for part in file.parts):
                continue
            
            try:
                content = file.read_text(encoding="utf-8")
                documents.append({
                    "content": content,
                    "source": str(file)
                })
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
    return documents


def parse_documents(raw_docs: List[Dict]) -> List[ParsedPage]:
    """Parse raw documents into enhanced ParsedPages."""
    return [parse_logseq_page_enhanced(doc) for doc in raw_docs]


def chunk_parsed_pages(parsed_pages: List[ParsedPage]) -> List[Document]:
    """Chunk parsed pages into documents for indexing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=25,
    )
    chunked_documents = []
    
    for page in parsed_pages:
        metadata = {
            "source": page.source,
            "page_title": Path(page.source).stem,
            "page_properties": str(page.page_properties),  # Store as string for metadata
            "tags": ",".join(page.tags),
            "page_links": ",".join(page.page_links),
            "journal_date": page.journal_date or "",
        }
        
        texts = text_splitter.create_documents(
            [page.page_content],
            metadatas=[metadata]
        )
        
        for i, chunk in enumerate(texts):
            chunk.metadata.update({
                "stable_id": f"{page.source}:{i}",
                "source": page.source,
                "chunk_index": i,
                "content_hash": content_hash(chunk.page_content),
                "page_title": Path(page.source).stem,
                "tags": ",".join(page.tags),
                "page_links": ",".join(page.page_links),
                "journal_date": page.journal_date or "",
            })
            chunked_documents.append(chunk)
    
    return chunked_documents


def sync_incremental(chunked_documents: List[Document]) -> SyncResult:
    """
    Perform incremental sync: detect new, modified, deleted chunks.
    
    Args:
        chunked_documents: Current chunks from the graph
        
    Returns:
        SyncResult with counts
    """
    start_time = time.time()
    result = SyncResult()
    
    chroma_dir = get_graph_manager().get_chroma_dir()
    keyword_db_path = get_graph_manager().get_keyword_db()
    
    to_add = []
    to_update = []
    to_delete = []
    
    # Check if index exists
    if Path(chroma_dir).exists() and Path(keyword_db_path).exists():
        try:
            vector_store = get_chroma_db()
            existing = vector_store._collection.get(include=['metadatas'])
            
            existing_docs = {
                metadata['stable_id']: metadata["content_hash"]
                for metadata in existing['metadatas']
            }
            
            current_ids = {
                chunk.metadata["stable_id"] for chunk in chunked_documents
            }
            
            # Detect additions and updates
            for chunk in chunked_documents:
                stable_id = chunk.metadata["stable_id"]
                new_hash = chunk.metadata["content_hash"]
                
                if stable_id not in existing_docs:
                    to_add.append(chunk)
                elif existing_docs[stable_id] != new_hash:
                    to_update.append(chunk)
            
            # Detect deletions
            for stable_id in existing_docs:
                if stable_id not in current_ids:
                    to_delete.append(stable_id)
            
            # Apply changes
            if to_add:
                vector_store.add_documents(
                    documents=to_add,
                    ids=[chunk.metadata["stable_id"] for chunk in to_add]
                )
                keyword_add(to_add)
                result.added = len(to_add)
            
            if to_update:
                vector_store.update_documents(
                    ids=[chunk.metadata["stable_id"] for chunk in to_update],
                    documents=to_update
                )
                keyword_update(to_update)
                result.updated = len(to_update)
            
            if to_delete:
                vector_store.delete(ids=to_delete)
                keyword_delete(to_delete)
                result.deleted = len(to_delete)
                
        except Exception as e:
            result.errors.append(f"Incremental sync error: {e}")
            # Fall back to full sync
            return sync_full(chunked_documents)
    else:
        # No existing index, do full sync
        return sync_full(chunked_documents)
    
    result.duration_seconds = time.time() - start_time
    return result


def sync_full(chunked_documents: List[Document]) -> SyncResult:
    """
    Perform full sync: recreate indexes from scratch.
    
    Args:
        chunked_documents: All chunks from the graph
        
    Returns:
        SyncResult with counts
    """
    start_time = time.time()
    result = SyncResult()
    
    try:
        # Recreate vector store
        chroma_dir = get_graph_manager().get_chroma_dir()
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
        
        vector_store = Chroma(
            collection_name="logseq_docs",
            embedding_function=get_embeddings(),
            persist_directory=chroma_dir
        )
        
        vector_store.add_documents(
            documents=chunked_documents,
            ids=[chunk.metadata["stable_id"] for chunk in chunked_documents]
        )
        
        # Recreate keyword database
        setup_keyword_db()
        keyword_add(chunked_documents)
        
        result.added = len(chunked_documents)
        
    except Exception as e:
        result.errors.append(f"Full sync error: {e}")
    
    result.duration_seconds = time.time() - start_time
    return result


def sync_graph(graph_path: Optional[str] = None, full: bool = False) -> SyncResult:
    """
    Main sync function: load graph, parse, chunk, and sync indexes.
    
    Args:
        graph_path: Optional path to graph (uses current graph if not provided)
        full: If True, force full rebuild instead of incremental
        
    Returns:
        SyncResult with statistics
    """
    manager = get_graph_manager()
    
    if graph_path:
        manager.set_graph(graph_path)
    
    current_graph = manager.get_current()
    
    # Load and parse
    raw_docs = load_logseq_graph(current_graph.path)
    parsed_pages = parse_documents(raw_docs)
    chunked_documents = chunk_parsed_pages(parsed_pages)
    
    # Sync
    if full:
        result = sync_full(chunked_documents)
    else:
        result = sync_incremental(chunked_documents)
    
    # Update graph status
    manager.update_status(
        page_count=len(parsed_pages),
        chunk_count=len(chunked_documents),
        vector_count=len(chunked_documents),
        last_sync=datetime.now().isoformat(),
        is_synced=len(result.errors) == 0
    )
    
    return result


def check_graph_changes(graph_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Check if the graph has changed since last sync.
    
    Returns:
        (has_changes, message)
    """
    manager = get_graph_manager()
    current = manager.get_current()
    
    if graph_path:
        expanded = os.path.expanduser(graph_path)
        if expanded != current.path:
            return True, "Different graph selected"
    
    if not current.is_synced:
        return True, "Not yet synchronized"
    
    if not current.last_sync:
        return True, "Never synchronized"
    
    # Check for file modifications since last sync
    try:
        last_sync_time = datetime.fromisoformat(current.last_sync).timestamp()
        graph = Path(current.path)
        
        search_dirs = []
        if (graph / "pages").exists():
            search_dirs.append(graph / "pages")
        if (graph / "journals").exists():
            search_dirs.append(graph / "journals")
        search_dirs.append(graph)
        
        for search_dir in search_dirs:
            for file in search_dir.rglob("*.md"):
                if "bak" in file.parts:
                    continue
                if file.stat().st_mtime > last_sync_time:
                    return True, f"Modified file detected: {file.name}"
    except Exception:
        pass
    
    return False, "No changes detected"