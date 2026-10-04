import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage

from graph import rag_graph


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Logseq Lens",
    page_icon="🔎",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("Logseq Lens")
st.caption("Ask questions about your Logseq notes.")


# --------------------------------------------------
# Conversation history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Chat input
# --------------------------------------------------

if prompt := st.chat_input("Ask your Logseq notes..."):

    # ----------------------------------------------
    # Display user message immediately
    # ----------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)


    # ----------------------------------------------
    # Convert Streamlit history to LangChain messages
    # ----------------------------------------------

    graph_messages = []

    for message in st.session_state.messages:

        if message["role"] == "user":

            graph_messages.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif message["role"] == "assistant":

            graph_messages.append(
                AIMessage(
                    content=message["content"]
                )
            )


    # ----------------------------------------------
    # Generate assistant response
    # ----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Logseq Lens is thinking..."):

            result = rag_graph.invoke({
                "question": prompt,
                "search_query": "",
                "documents": [],
                "answer": None,
                "messages": graph_messages
            })


        answer = result["answer"].answer

        st.markdown(answer)


        # ------------------------------------------
        # Display sources
        # ------------------------------------------

        documents = result["documents"]

        if documents:

            with st.expander(
                f"📚 Sources ({len(documents)})"
            ):

                for i, doc in enumerate(documents, 1):

                    source = doc.metadata.get(
                        "source",
                        "Unknown source"
                    )

                    st.markdown(
                        f"**{i}. 📄 {source}**"
                    )

                    st.caption(
                        doc.page_content[:500]
                    )


    # ----------------------------------------------
    # Save assistant response
    # ----------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })