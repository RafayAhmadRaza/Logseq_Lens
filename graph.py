import __main__
from rag import rag_chain
from rag import build_context
from rag import search
from rag import RAGResponse
from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage

class RAGState(TypedDict):
    question:str
    documents:list
    answer:RAGResponse
    messages: list[BaseMessage]


def retrieved_node(state:RAGState):
    documents = search(state['question'],k=5)

    return {
        "documents":documents
    }

def generate_node(state: RAGState):
    context = build_context(state["documents"])

    answer = rag_chain.invoke({
        "context": context,
        "question": state["question"]
    })

    return {
        "answer": answer,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer.answer)
        ]
    }


if __name__ == "__main__":
    graph = StateGraph(RAGState)

    graph.add_node("retrieve",retrieved_node)
    graph.add_node("generate",generate_node)
    graph.add_edge(START,'retrieve')
    graph.add_edge('retrieve','generate')
    graph.add_edge('generate',END)

    rag_graph = graph.compile()


    messages = []

    while True:
        question = input("\nYou: ")

        if question.lower() in {"exit", "quit"}:
            break

        result = rag_graph.invoke({
            "question": question,
            "documents": [],
            "answer": None,
            "messages": messages
        })

        messages = result["messages"]

        print("\nAssistant:", result["answer"].answer)

