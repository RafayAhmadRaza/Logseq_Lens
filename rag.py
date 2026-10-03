from pathlib import Path
from dotenv import load_dotenv





def load_logseq_graph(graph_path:str):

    """Loads the logseq graph from given path"""


    graph = Path(graph_path)

    documents = []

    for file in graph.rglob("*.md"):
        content = file.read_text(encoding="utf-8")

        documents.append({
            "content":content,
            "source":str(file)
        })
    return documents






if __name__ == "__main__":

    load_dotenv()
    
    path = logPath

    docs = load_logseq_graph(path)

    print(docs)

