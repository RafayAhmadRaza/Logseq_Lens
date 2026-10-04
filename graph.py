
from rag import rag_chain, llm
from rag import build_context
from rag import search
from rag import RAGResponse

from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse = get_client()
langfuse_handler = CallbackHandler()

query_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Rewrite the user's latest question into a standalone search query.

Use the conversation history to resolve references such as:
- it
- they
- that
- the previous one
- the haiku
- the project

If the question is already standalone, return it unchanged.

Return only the search query. Do not answer the question."""
    ),
    (
        "human",
        """Conversation history:
{history}

Latest question:
{question}"""
    )
])

query_rewriter = query_rewrite_prompt | llm | StrOutputParser()


class RAGState(TypedDict):
    question: str
    search_query: str
    documents: list
    answer: RAGResponse
    messages: Annotated[list[BaseMessage], add_messages]


def rewrite_query_node(state: RAGState):
    history = "\n".join(
        f"{message.type}: {message.content}"
        for message in state["messages"]
    )

    search_query = query_rewriter.invoke({
        "history": history,
        "question": state["question"]
    })

    return {
        "search_query": search_query
    }


def retrieved_node(state: RAGState):
    documents = search(
        state["search_query"],
        k=5
    )

    return {
        "documents": documents
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

graph = StateGraph(RAGState)

graph.add_node("rewrite_query", rewrite_query_node)
graph.add_node("retrieve", retrieved_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "rewrite_query")
graph.add_edge("rewrite_query", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

rag_graph = graph.compile()




if __name__ == "__main__":

    messages = []

    while True:

        question = input("\nYou: ")

        if question.lower() in {"exit", "quit"}:
            break

        result = rag_graph.invoke({
            "question": question,
            "search_query": "",
            "documents": [],
            "answer": None,
            "messages": messages
        },
        config={
            'callbacks':[langfuse_handler]
        }
        )


        messages = result["messages"]

        print("\nSearch query:", result["search_query"])
        print("\nAssistant:", result["answer"].answer)
