"""
Configuration management for Logseq Lens.
Centralizes all settings from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Logseq Graph
    logseq_path: str = os.getenv("LOGSEQ_PATH", "~/Logseq")

    # LLM Providers
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openrouter/free")
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")

    # Embeddings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

    # Retrieval
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))

    # Research Agent
    max_research_iterations: int = int(os.getenv("MAX_RESEARCH_ITERATIONS", "3"))

    # Langfuse
    langfuse_secret_key: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_public_key: Optional[str] = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_base_url: Optional[str] = os.getenv("LANGFUSE_BASE_URL")

    # Paths
    chroma_persist_dir: str = "./chroma_logseq_db"
    keyword_db_path: str = "./keyword_search.db"

    def __post_init__(self):
        """Expand user paths."""
        self.logseq_path = os.path.expanduser(self.logseq_path)

    def get_logseq_path(self) -> Path:
        """Get the Logseq graph path as a Path object."""
        return Path(self.logseq_path)

    def validate_logseq_path(self, path: Optional[str] = None) -> tuple[bool, str]:
        """
        Validate that a path exists and appears to be a Logseq graph.
        Returns (is_valid, error_message).
        """
        check_path = Path(os.path.expanduser(path or self.logseq_path))

        if not check_path.exists():
            return False, f"Path does not exist: {check_path}"

        if not check_path.is_dir():
            return False, f"Path is not a directory: {check_path}"

        # Check for Logseq indicators
        has_pages = (check_path / "pages").exists()
        has_journals = (check_path / "journals").exists()
        has_config = (check_path / "logseq" / "config.edn").exists() or (check_path / "config.edn").exists()

        if not (has_pages or has_journals or has_config):
            return False, f"Directory does not appear to be a Logseq graph (missing pages/, journals/, or config.edn): {check_path}"

        return True, ""


# Global config instance
config = Config()