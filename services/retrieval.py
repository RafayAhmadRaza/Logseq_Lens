"""
Unified retrieval services for Logseq Lens.
Provides keyword, vector, hybrid search and document retrieval.
"""
import os
import re
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import config
from services.graph_config import get_graph_manager


def build_context(docs) -> str:
    """Build context string from documents (Document or SearchResult)."""
    context = []
    for i, doc in enumerate(docs, start=1):
        # Handle both Document and SearchResult
        content = getattr(doc, 'page_content', getattr(doc, 'content', ''))
        metadata = getattr(doc, 'metadata', {})
        context.append(
            f"[Source {i}]\n"
            f"Source: {metadata.get('source', 'Unknown')}\n"
            f"{content}"
        )
    return "\n\n".join(context)


# Global embeddings instance
_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create the embeddings model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    return _embeddings


def get_chroma_db() -> Chroma:
    """Get the Chroma vector store for the current graph."""
    graph_manager = get_graph_manager()
    chroma_dir = graph_manager.get_chroma_dir()
    
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    
    return Chroma(
        collection_name="logseq_docs",
        embedding_function=get_embeddings(),
        persist_directory=chroma_dir
    )


def get_keyword_db() -> sqlite3.Connection:
    """Get the SQLite FTS5 keyword search database for the current graph."""
    graph_manager = get_graph_manager()
    db_path = graph_manager.get_keyword_db()
    
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    db = sqlite3.connect(db_path)
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS documents
        USING fts5(
            stable_id,
            source,
            chunk_index UNINDEXED,
            content
        )
    """)
    db.commit()
    return db


def setup_keyword_db():
    """Initialize the keyword database."""
    db = get_keyword_db()
    db.close()


@dataclass
class SearchResult:
    """Structured search result."""
    stable_id: str
    source: str
    chunk_index: int
    content: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def to_document(self) -> Document:
        """Convert to LangChain Document."""
        meta = {
            "stable_id": self.stable_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
        }
        if self.metadata:
            meta.update(self.metadata)
        if self.score is not None:
            meta["search_score"] = self.score
        return Document(page_content=self.content, metadata=meta)


def content_hash(text: str) -> str:
    """Generate SHA256 hash of content."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


# ===================== KEYWORD SEARCH =====================

def keyword_search(query: str, k: int = 5) -> List[SearchResult]:
    """
    Search the SQLite FTS5 keyword database.
    
    Args:
        query: Search query string
        k: Maximum number of results
        
    Returns:
        List of SearchResult objects
    """
    db = get_keyword_db()
    
    # Extract meaningful terms
    terms = re.findall(r"\b[\w]+\b", query.lower())
    
    stopwords = {
        "what", "is", "are", "do", "does", "did", "i", "me", "my",
        "the", "a", "an", "to", "of", "for", "how", "much", "many", "need"
    }
    
    terms = [term for term in terms if term not in stopwords and len(term) > 1]
    
    if not terms:
        db.close()
        return []
    
    # Build FTS query with OR between terms
    fts_query = " OR ".join(f'"{term}"' for term in terms)
    
    cursor = db.execute(
        """
        SELECT stable_id, source, chunk_index, content
        FROM documents
        WHERE documents MATCH ?
        LIMIT ?
        """,
        (fts_query, k)
    )
    rows = cursor.fetchall()
    db.close()
    
    return [
        SearchResult(
            stable_id=stable_id,
            source=source,
            chunk_index=chunk_index,
            content=content
        )
        for stable_id, source, chunk_index, content in rows
    ]


# ===================== VECTOR SEARCH =====================

def vector_search(query: str, k: int = 5) -> List[SearchResult]:
    """
    Search the Chroma vector database using semantic similarity.
    
    Args:
        query: Search query string
        k: Maximum number of results
        
    Returns:
        List of SearchResult objects with similarity scores
    """
    vector_store = get_chroma_db()
    
    results = vector_store.similarity_search_with_score(query, k=k)
    
    search_results = []
    for doc, score in results:
        search_results.append(SearchResult(
            stable_id=doc.metadata.get("stable_id", ""),
            source=doc.metadata.get("source", ""),
            chunk_index=doc.metadata.get("chunk_index", 0),
            content=doc.page_content,
            score=score,
            metadata=doc.metadata
        ))
    
    return search_results


# ===================== HYBRID SEARCH =====================

def hybrid_search(query: str, k: int = 5, rrf_k: int = 60) -> List[SearchResult]:
    """
    Hybrid search using Reciprocal Rank Fusion (RRF).
    
    Args:
        query: Search query string
        k: Maximum number of results
        rrf_k: RRF constant (default 60)
        
    Returns:
        List of SearchResult objects ranked by combined score
    """
    semantic_results = vector_search(query, k)
    keyword_results = keyword_search(query, k)
    
    scores: Dict[str, float] = {}
    documents: Dict[str, SearchResult] = {}
    
    # Semantic search results
    for rank, result in enumerate(semantic_results, start=1):
        sid = result.stable_id
        scores[sid] = scores.get(sid, 0) + (1 / (rrf_k + rank))
        documents[sid] = result
    
    # Keyword search results
    for rank, result in enumerate(keyword_results, start=1):
        sid = result.stable_id
        scores[sid] = scores.get(sid, 0) + (1 / (rrf_k + rank))
        # Prefer the result with more metadata (usually from vector search)
        if sid not in documents or result.metadata:
            documents[sid] = result
    
    # Sort by combined score
    ranked_ids = sorted(scores.keys(), key=scores.get, reverse=True)
    
    results = []
    for sid in ranked_ids[:k]:
        result = documents[sid]
        result.score = scores[sid]
        result.metadata = result.metadata or {}
        result.metadata["rrf_score"] = scores[sid]
        results.append(result)
    
    return results


# ===================== DOCUMENT RETRIEVAL =====================

def get_document(stable_id: str) -> Optional[SearchResult]:
    """
    Retrieve a specific document/chunk by its stable_id.
    
    Args:
        stable_id: The stable identifier (source:chunk_index)
        
    Returns:
        SearchResult if found, None otherwise
    """
    # Try keyword DB first (has full content)
    db = get_keyword_db()
    cursor = db.execute(
        "SELECT stable_id, source, chunk_index, content FROM documents WHERE stable_id = ?",
        (stable_id,)
    )
    row = cursor.fetchone()
    db.close()
    
    if row:
        return SearchResult(
            stable_id=row[0],
            source=row[1],
            chunk_index=row[2],
            content=row[3]
        )
    
    # Fallback to vector store
    vector_store = get_chroma_db()
    try:
        results = vector_store.get(ids=[stable_id], include=["documents", "metadatas"])
        if results["ids"]:
            return SearchResult(
                stable_id=results["ids"][0],
                source=results["metadatas"][0].get("source", ""),
                chunk_index=results["metadatas"][0].get("chunk_index", 0),
                content=results["documents"][0],
                metadata=results["metadatas"][0]
            )
    except Exception:
        pass
    
    return None


def get_full_page(source_path: str) -> List[SearchResult]:
    """
    Retrieve all chunks for a given source page.
    
    Args:
        source_path: Path to the source markdown file
        
    Returns:
        List of SearchResult objects (all chunks for that page)
    """
    db = get_keyword_db()
    cursor = db.execute(
        "SELECT stable_id, source, chunk_index, content FROM documents WHERE source = ? ORDER BY chunk_index",
        (source_path,)
    )
    rows = cursor.fetchall()
    db.close()
    
    return [
        SearchResult(
            stable_id=row[0],
            source=row[1],
            chunk_index=row[2],
            content=row[3]
        )
        for row in rows
    ]


# ===================== ADVANCED SEARCH =====================

def search_by_tag(tag: str, k: int = 10) -> List[SearchResult]:
    """
    Search for pages/documents by tag.
    Note: This requires tags to be indexed. For now, uses keyword search on #tag.
    """
    # Search for the tag in content
    return keyword_search(f"#{tag}", k)


def search_by_page_title(title: str, k: int = 10) -> List[SearchResult]:
    """
    Search for pages by title/filename.
    """
    return keyword_search(title, k)


# ===================== INDEXING HELPERS =====================

def keyword_add(docs: List[Document]):
    """Add documents to keyword database."""
    db = get_keyword_db()
    for doc in docs:
        stable_id = doc.metadata.get("stable_id", "")
        if not stable_id:
            continue
        db.execute("DELETE FROM documents WHERE stable_id = ?", (stable_id,))
        db.execute(
            "INSERT INTO documents (stable_id, source, chunk_index, content) VALUES (?, ?, ?, ?)",
            (stable_id, doc.metadata.get("source", ""), doc.metadata.get("chunk_index", 0), doc.page_content)
        )
    db.commit()
    db.close()


def keyword_update(docs: List[Document]):
    """Update documents in keyword database (same as add for FTS5)."""
    keyword_add(docs)


def keyword_delete(stable_ids: List[str]):
    """Delete documents from keyword database."""
    db = get_keyword_db()
    for stable_id in stable_ids:
        db.execute("DELETE FROM documents WHERE stable_id = ?", (stable_id,))
    db.commit()
    db.close()


def get_index_stats() -> Dict[str, Any]:
    """Get statistics about the current index."""
    graph_manager = get_graph_manager()
    current = graph_manager.get_current()
    
    stats = {
        "graph_path": current.path,
        "graph_name": current.name,
        "page_count": current.page_count,
        "chunk_count": current.chunk_count,
        "vector_count": current.vector_count,
        "last_sync": current.last_sync,
        "is_synced": current.is_synced,
        "chroma_dir": current.chroma_dir,
        "keyword_db": current.keyword_db,
    }
    
    # Try to get actual counts from databases
    try:
        vector_store = get_chroma_db()
        count = vector_store._collection.count()
        stats["vector_count"] = count
        graph_manager.update_status(vector_count=count)
    except Exception:
        pass
    
    try:
        db = get_keyword_db()
        cursor = db.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        stats["chunk_count"] = count
        graph_manager.update_status(chunk_count=count)
        db.close()
    except Exception:
        pass
    
    return stats