
from langchain_core.embeddings import embeddings
import os
from pathlib import Path

import LogseqMarkdownParser
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import sqlite3

from langchain_core.documents import Document


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import hashlib
import re

class RAGResponse(BaseModel):
    answer: str = Field(
        description="Answer to the user's question using only the provided context."
    )
    sufficient_context: bool = Field(
        description="Whether the provided context contains enough information to answer the question."
    )
    sources: list[str] = Field(
        description="Sources used to answer the question."
    )

embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

llm = ChatOllama(
    model="gemma4:e4b",
    temperature=0
)

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

structured_llm = llm.with_structured_output(RAGResponse)
rag_chain = prompt | structured_llm



def build_context(docs):
    context = []

    for i, doc in enumerate(docs, start=1):
        context.append(
            f"[Source {i}]\n"
            f"Source: {doc.metadata['source']}\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(context)

def content_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()



def load_logseq_graph(graph_path:str):

    """Loads the logseq graph from given path"""


    graph = Path(graph_path)

    if graph.exists():
        documents = []

        for file in graph.rglob("*.md"):

            if "bak" in file.parts:
                continue

            content = file.read_text(encoding="utf-8")

            documents.append({
            "content":content,
            "source":str(file)
        })
        return documents
    else:
        return "Graph Does Not Exist"

def parse_documents(docs):
    """Parses Logseq Documents to Make the Documents more structured while maintaing content"""
    documents = []
   
    for doc in docs:
        page = LogseqMarkdownParser.parse_text(doc["content"])
        parsed = page.dict()
        parsed["source"] = doc["source"]
        documents.append(parsed)


    return documents
    
def chunk_documents(docs):
    """Chunks documents"""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=25,
    )
    chunked_documents = []
    for doc in docs:
        metadata = {
            "source":doc['source'],
            }
        texts = text_splitter.create_documents(
            [doc['page_content']],
            metadatas=[metadata])
        
        for i,chunk in enumerate(texts):
            chunk.metadata={
                "stable_id": f"{doc['source']}:{i}",
                "source":doc["source"],
                "chunk_index":i,
                "content_hash":content_hash(chunk.page_content)
            }
            chunked_documents.append(chunk)
    
    return chunked_documents

def store_embedding_documents(docs):
    """Embed and store documents in Chroma."""


    vector_store = Chroma(
        collection_name="logseq_docs",
        embedding_function=embeddings,
        persist_directory="./chroma_logseq_db"
    )

    vector_store.add_documents(docs,
    ids = [
        chunk.metadata['stable_id']
        for chunk in docs
    ])

    return vector_store

def get_db():
    path =Path("./chroma_logseq_db") 


    vector_store = Chroma(
        collection_name="logseq_docs",
        embedding_function=embeddings,
        persist_directory="./chroma_logseq_db"
    )

    return vector_store



def sync(docs):
    """Checks the db for any updates, removals and new entries"""
    chunked_documents = chunk_documents(docs)

    to_add = []
    to_update = []
    to_delete = []

    if Path("./chroma_logseq_db").exists():
        # print("Exists")

        vector_store = get_db()
        existing = vector_store._collection.get(
            include=['metadatas']
        )

        existing_docs = {
            metadata['stable_id']: metadata["content_hash"]
            for metadata in  existing['metadatas']
        }

        current_ids = {
            chunk.metadata["stable_id"]
            for chunk in chunked_documents
        }
        # print("Existing Chroma IDs:", existing["ids"][:5])
        for chunk in chunked_documents:
            stable_id = chunk.metadata["stable_id"]
            new_hash = chunk.metadata["content_hash"]

            if stable_id not in existing_docs:
                # print("ADD:",stable_id)
                to_add.append(chunk)
            elif existing_docs[stable_id] == new_hash:
                # print("SKIP",stable_id)
                continue
            else:
                # print("UPDATE: ",stable_id)
                to_update.append(chunk)
        
        for stable_id in existing_docs:
            if stable_id not in current_ids:
                print("DELETE:",stable_id)
                to_delete.append(stable_id)

        if to_add:
            vector_store.add_documents(
                documents=to_add,
                ids=[
                    chunk.metadata["stable_id"]
                    for chunk in to_add
                ]
            )

            keyword_add(to_add)


        if to_update:
            vector_store.update_documents(
                ids=[
                    chunk.metadata["stable_id"]
                    for chunk in to_update
                ],
                documents=to_update
            )

            keyword_update(to_update)


        if to_delete:
            vector_store.delete(ids=to_delete)

            keyword_delete(to_delete)
            
        return vector_store

    else:
        vector_store = store_embedding_documents(chunked_documents)
        setup_keyword_db()
        keyword_add(chunked_documents)

        return vector_store


def get_keyword_db():
    db = sqlite3.connect("./keyword_search.db")

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
    db = get_keyword_db()
    db.close()

def populate_keyword_db(docs):
    db = get_keyword_db()

    for doc in docs:
        db.execute(
            """
            INSERT INTO documents (
                stable_id,
                source,
                chunk_index,
                content
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                doc.metadata["stable_id"],
                doc.metadata["source"],
                doc.metadata["chunk_index"],
                doc.page_content
            )
        )

    db.commit()
    db.close()

def keyword_search(query, k=5):
    db = get_keyword_db()

    terms = re.findall(r"\b[\w]+\b", query.lower())

    stopwords = {
        "what",
        "is",
        "are",
        "do",
        "does",
        "did",
        "i",
        "me",
        "my",
        "the",
        "a",
        "an",
        "to",
        "of",
        "for",
        "how",
        "much",
        "many",
        "need",
    }

    terms = [
        term
        for term in terms
        if term not in stopwords
    ]

    if not terms:
        db.close()
        return []

    # Search each word rather than passing raw user input
    fts_query = " OR ".join(
        f'"{term}"'
        for term in terms
    )

    results = db.execute(
        """
        SELECT stable_id, source, chunk_index, content
        FROM documents
        WHERE documents MATCH ?
        LIMIT ?
        """,
        (fts_query, k)
    ).fetchall()

    db.close()

    return [
        Document(
            page_content=content,
            metadata={
                "stable_id": stable_id,
                "source": source,
                "chunk_index": chunk_index
            }
        )
        for stable_id, source, chunk_index, content in results
    ]
def semantic_search(query:str, k=5):
    """Search Logseq documents using semantic similarity"""
    vector_store = get_db()

    results = vector_store.similarity_search(
        query,
        k=k
    )

    return results

def semantic_search_debug(query: str, k=10):
    vector_store = get_db()

    results = vector_store.similarity_search_with_score(
        query,
        k=k
    )

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- RESULT {i} ---")
        print(f"Score: {score}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Chunk: {doc.metadata['chunk_index']}")
        print(f"ID: {doc.metadata['stable_id']}")
        print(doc.page_content[:500])

    return [doc for doc, score in results]

def search(query, k=5, rrf_k=60):
    """Hybrid search using Reciprocal Rank Fusion."""

    semantic_results = semantic_search(query, k)
    keyword_results = keyword_search(query, k)

    scores = {}
    documents = {}

    # Semantic search results
    for rank, doc in enumerate(semantic_results, start=1):
        stable_id = doc.metadata["stable_id"]

        scores[stable_id] = scores.get(stable_id, 0) + (
            1 / (rrf_k + rank)
        )

        documents[stable_id] = doc

    # Keyword search results
    for rank, doc in enumerate(keyword_results, start=1):
        stable_id = doc.metadata["stable_id"]

        scores[stable_id] = scores.get(stable_id, 0) + (
            1 / (rrf_k + rank)
        )

        documents[stable_id] = doc

    # Highest RRF score first
    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [
        documents[stable_id]
        for stable_id in ranked_ids[:k]
    ]

def keyword_add(docs):
    db = get_keyword_db()

    for doc in docs:
        stable_id = doc.metadata["stable_id"]

        db.execute(
            "DELETE FROM documents WHERE stable_id = ?",
            (stable_id,)
        )

        db.execute(
            """
            INSERT INTO documents
            (stable_id, source, chunk_index, content)
            VALUES (?, ?, ?, ?)
            """,
            (
                stable_id,
                doc.metadata["source"],
                doc.metadata["chunk_index"],
                doc.page_content
            )
        )

    db.commit()
    db.close()

def keyword_update(docs):
    """Updates existing documents in the keyword search database."""

    db = get_keyword_db()

    for doc in docs:
        stable_id = doc.metadata["stable_id"]

        db.execute(
            """
            DELETE FROM documents
            WHERE stable_id = ?
            """,
            (stable_id,)
        )

        db.execute(
            """
            INSERT INTO documents
            (stable_id, source, chunk_index, content)
            VALUES (?, ?, ?, ?)
            """,
            (
                stable_id,
                doc.metadata["source"],
                doc.metadata["chunk_index"],
                doc.page_content
            )
        )

    db.commit()
    db.close()


def keyword_delete(stable_ids):
    """Deletes documents from the keyword search database."""

    db = get_keyword_db()

    for stable_id in stable_ids:
        db.execute(
            """
            DELETE FROM documents
            WHERE stable_id = ?
            """,
            (stable_id,)
        )

    db.commit()
    db.close()


def rag(query, k=5):
    documents = search(query, k)

    context = build_context(documents)


    answer = rag_chain.invoke({
        "context": context,
        "question": query
    })

    return {
        "answer": answer,
        "documents": documents
    }

if __name__ == "__main__":
    load_dotenv()

    path = os.getenv("LOGSEQ_PATH")
    docs = load_logseq_graph(path)
    structured_docs = parse_documents(docs)
    sync(structured_docs)

    result = rag(
        "What materials do I need to level Yelan?",
        k=5
    )

    print("\n===== STRUCTURED ANSWER =====")

    print("Answer:")
    print(result["answer"].answer)

    print("\nEnough context:")
    print(result["answer"].sufficient_context)

    print("\nSources:")
    for source in result["answer"].sources:
        print("-", source)