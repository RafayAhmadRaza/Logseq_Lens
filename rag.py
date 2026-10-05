"""
Legacy RAG module - maintained for backward compatibility.
New code should use services/ modules directly.
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openrouter import ChatOpenRouter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import hashlib
import re
import sqlite3

from dotenv import load_dotenv
from config import config
from services import (
    keyword_search, vector_search, hybrid_search, get_document,
    get_chroma_db, get_keyword_db, setup_keyword_db,
    keyword_add, keyword_update, keyword_delete,
    content_hash, sync_graph, load_logseq_graph, chunk_parsed_pages,
    parse_documents, get_index_stats
)
from services.logseq_parser import parse_logseq_page_enhanced

load_dotenv()


# ---- LLM Provider Abstraction ----

def get_llm(provider: str):
    """Get LLM instance for the given provider."""
    if provider == "Ollama":
        return ChatOllama(
            model=config.ollama_model,
            temperature=0
        )
    elif provider == "OpenRouter":
        if not config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
        return ChatOpenRouter(
            model=config.openrouter_model,
            temperature=0,
            api_key=config.openrouter_api_key
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---- Structured Output ----

class RAGResponse(BaseModel):
    answer: str = Field(
        description="Answer to the user's question using only the provided context."
    )
    sufficient_context: bool = Field(
        description="Whether the provided context contains enough information to answer the question."
    )
    sources: List[str] = Field(
        description="Sources used to answer the question."
    )


# ---- Prompts ----

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


def get_rag_chain(provider: str):
    """Get the RAG chain with structured output for a provider."""
    llm = get_llm(provider)
    structured_llm = llm.with_structured_output(RAGResponse)
    return prompt | structured_llm


# ---- Context Building ----

def build_context(docs) -> str:
    """Build context string from documents (Document or SearchResult)."""
    context = []
    for i, doc in enumerate(docs, start=1):
        content = getattr(doc, 'page_content', getattr(doc, 'content', ''))
        context.append(
            f"[Source {i}]\n"
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"{content}"
        )
    return "\n\n".join(context)


# ---- Legacy Functions (kept for compatibility) ----

def parse_documents_legacy(docs: List[Dict]) -> List[Dict]:
    """Legacy parse function - returns dict format."""
    documents = []
    for doc in docs:
        page = parse_logseq_page_enhanced(doc)
        parsed = {
            "page_properties": page.page_properties,
            "page_content": page.page_content,
            "blocks": page.blocks,
            "source": page.source,
        }
        documents.append(parsed)
    return documents


def chunk_documents(docs: List[Dict]) -> List[Document]:
    """Legacy chunk function."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=25,
    )
    chunked_documents = []
    for doc in docs:
        metadata = {
            "source": doc['source'],
        }
        texts = text_splitter.create_documents(
            [doc['page_content']],
            metadatas=[metadata]
        )
        for i, chunk in enumerate(texts):
            chunk.metadata = {
                "stable_id": f"{doc['source']}:{i}",
                "source": doc["source"],
                "chunk_index": i,
                "content_hash": content_hash(chunk.page_content)
            }
            chunked_documents.append(chunk)
    return chunked_documents


def store_embedding_documents(docs: List[Document]):
    """Legacy store function."""
    vector_store = Chroma(
        collection_name="logseq_docs",
        embedding_function=HuggingFaceEmbeddings(model_name=config.embedding_model),
        persist_directory="./chroma_logseq_db"
    )
    vector_store.add_documents(docs, ids=[chunk.metadata['stable_id'] for chunk in docs])
    return vector_store


def get_db():
    """Legacy get_db - uses old fixed path."""
    vector_store = Chroma(
        collection_name="logseq_docs",
        embedding_function=HuggingFaceEmbeddings(model_name=config.embedding_model),
        persist_directory="./chroma_logseq_db"
    )
    return vector_store


def semantic_search(query: str, k: int = 5):
    """Legacy semantic search."""
    return vector_search(query, k)


def semantic_search_debug(query: str, k: int = 10):
    """Legacy debug search."""
    vector_store = get_db()
    results = vector_store.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- RESULT {i} ---")
        print(f"Score: {score}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Chunk: {doc.metadata['chunk_index']}")
        print(f"ID: {doc.metadata['stable_id']}")
        print(doc.page_content[:500])
    return [doc for doc, score in results]


def search(query: str, k: int = 5, rrf_k: int = 60):
    """Legacy hybrid search."""
    results = hybrid_search(query, k, rrf_k)
    return [r.to_document() for r in results]


def rag(query: str, k: int = 5, provider: str = "Ollama"):
    """Legacy RAG function."""
    documents = search(query, k)
    context = build_context(documents)
    rag_chain = get_rag_chain(provider)
    answer = rag_chain.invoke({"context": context, "question": query})
    return {"answer": answer, "documents": documents}


# ---- Main block for testing ----
if __name__ == "__main__":
    load_dotenv()
    path = config.logseq_path
    docs = load_logseq_graph(path)
    parsed = parse_documents(docs)
    chunked = chunk_parsed_pages(parsed)
    sync_graph()
    
    result = rag("What materials do I need to level Yelan?", k=5, provider="Ollama")
    print("\n===== STRUCTURED ANSWER =====")
    print("Answer:", result["answer"].answer)
    print("Enough context:", result["answer"].sufficient_context)
    print("Sources:")
    for source in result["answer"].sources:
        print("-", source)