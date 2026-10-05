"""
Graph configuration management.
Handles the currently active Logseq graph path and its associated index paths.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import hashlib


@dataclass
class GraphConfig:
    """Configuration for a specific Logseq graph."""
    path: str
    name: str = ""
    
    # Derived paths (per-graph indexes)
    chroma_dir: str = ""
    keyword_db: str = ""
    
    # Status
    page_count: int = 0
    chunk_count: int = 0
    vector_count: int = 0
    last_sync: Optional[str] = None
    is_synced: bool = False
    
    def __post_init__(self):
        self.path = os.path.expanduser(self.path)
        if not self.name:
            self.name = Path(self.path).name
        
        # Create graph-specific index paths using a hash of the path
        path_hash = hashlib.md5(self.path.encode()).hexdigest()[:8]
        base_dir = Path("./indexes") / f"{self.name}_{path_hash}"
        base_dir.mkdir(parents=True, exist_ok=True)
        
        self.chroma_dir = str(base_dir / "chroma")
        self.keyword_db = str(base_dir / "keyword_search.db")
    
    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "name": self.name,
            "chroma_dir": self.chroma_dir,
            "keyword_db": self.keyword_db,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "vector_count": self.vector_count,
            "last_sync": self.last_sync,
            "is_synced": self.is_synced,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GraphConfig":
        config = cls(path=data["path"], name=data.get("name", ""))
        config.chroma_dir = data.get("chroma_dir", config.chroma_dir)
        config.keyword_db = data.get("keyword_db", config.keyword_db)
        config.page_count = data.get("page_count", 0)
        config.chunk_count = data.get("chunk_count", 0)
        config.vector_count = data.get("vector_count", 0)
        config.last_sync = data.get("last_sync")
        config.is_synced = data.get("is_synced", False)
        return config


class GraphManager:
    """Manages the active Logseq graph configuration."""
    
    def __init__(self, default_path: str):
        self.default_path = os.path.expanduser(default_path)
        self._current_graph: Optional[GraphConfig] = None
        self._graphs: dict[str, GraphConfig] = {}  # path -> GraphConfig
        
    def get_current(self) -> GraphConfig:
        """Get the current graph config, initializing with default if needed."""
        if self._current_graph is None:
            self._current_graph = GraphConfig(self.default_path)
        return self._current_graph
    
    def set_graph(self, path: str) -> GraphConfig:
        """Set the active graph by path."""
        expanded = os.path.expanduser(path)
        if expanded not in self._graphs:
            self._graphs[expanded] = GraphConfig(expanded)
        self._current_graph = self._graphs[expanded]
        return self._current_graph
    
    def validate_and_set(self, path: str) -> tuple[bool, str, Optional[GraphConfig]]:
        """Validate a path and set as current graph if valid."""
        from config import config
        is_valid, error = config.validate_logseq_path(path)
        if not is_valid:
            return False, error, None
        
        graph = self.set_graph(path)
        return True, "", graph
    
    def update_status(self, page_count: int = None, chunk_count: int = None, 
                      vector_count: int = None, last_sync: str = None, 
                      is_synced: bool = None):
        """Update the current graph's status."""
        if self._current_graph:
            if page_count is not None:
                self._current_graph.page_count = page_count
            if chunk_count is not None:
                self._current_graph.chunk_count = chunk_count
            if vector_count is not None:
                self._current_graph.vector_count = vector_count
            if last_sync is not None:
                self._current_graph.last_sync = last_sync
            if is_synced is not None:
                self._current_graph.is_synced = is_synced
    
    def get_chroma_dir(self) -> str:
        return self.get_current().chroma_dir
    
    def get_keyword_db(self) -> str:
        return self.get_current().keyword_db


# Global graph manager (initialized in main/app)
graph_manager: Optional[GraphManager] = None


def init_graph_manager(default_path: str) -> GraphManager:
    """Initialize the global graph manager."""
    global graph_manager
    graph_manager = GraphManager(default_path)
    return graph_manager


def get_graph_manager() -> GraphManager:
    """Get the global graph manager."""
    global graph_manager
    if graph_manager is None:
        from config import config
        graph_manager = GraphManager(config.logseq_path)
    return graph_manager