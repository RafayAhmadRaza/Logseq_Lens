
import os
from pathlib import Path

import LogseqMarkdownParser
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter




def load_logseq_graph(graph_path:str):

    """Loads the logseq graph from given path"""


    graph = Path(graph_path)

    if graph.exists():
        documents = []

        for file in graph.rglob("*.md"):
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
        document = page.dict()
        document["source"] = str(doc)
        documents.append(document)


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
            "source":doc['source']
        }
        texts = text_splitter.create_documents(
            [doc['page_content']],
            metadatas=[metadata])
        chunked_documents.extend(texts)
    
    return chunked_documents



if __name__ == "__main__":

    load_dotenv()
    
    path = os.getenv("LOGSEQ_PATH")

    docs = load_logseq_graph(path)
    structured_docs = parse_documents(docs)
    chunked_docs=chunk_documents(structured_docs)
    print(chunked_docs)