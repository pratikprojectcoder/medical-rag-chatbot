import streamlit as st
import traceback

from langgraph_pipeline import (
    ask_medical_question_langgraph
)

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Medical RAG Chatbot",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .stChatMessage {
        border-radius: 12px;
        padding: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:

    st.title("Medical AI Assistant")

    st.markdown("---")

    st.subheader("System Information")

    st.info("LLM: Groq Llama 3")
    st.info("Workflow: LangGraph")
    st.info("Vector DB: FAISS")
    st.info("Embeddings: MiniLM")

    st.markdown("---")

    st.subheader("Capabilities")

    st.success("Conversational Memory")
    st.success("Semantic Search")
    st.success("Query Rewriting")
    st.success("Medical Safety Layer")
    st.success("Emergency Detection")

    st.markdown("---")

    st.subheader("Project Architecture")

    st.code(
        """
User Query
    ↓
LangGraph Workflow
    ↓
Query Rewrite
    ↓
Semantic Retrieval
    ↓
Validation
    ↓
Safety Layer
    ↓
LLM Generation
        """
    )

    st.markdown("---")

    # CLEAR CHAT BUTTON

    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()

# ============================================
# TITLE
# ============================================

st.title("Medical RAG Chatbot")

st.caption(
    "AI-powered medical assistant using LangGraph, "
    "FAISS semantic retrieval, and Groq LLM."
)

# ============================================
# SESSION STATE
# ============================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# ============================================
# DISPLAY OLD CHAT MESSAGES
# ============================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # ============================================
        # DISPLAY SOURCES
        # ============================================

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            with st.expander(
                " View Retrieved Sources"
            ):

                for i, doc in enumerate(message["sources"]):

                    st.markdown(
                        f"##  Source {i+1}"
                    )

                    col1, col2 = st.columns(2)

                    # PAGE NUMBER
                    with col1:

                        if "page" in doc.metadata:

                            st.caption(
                                f" Page: "
                                f"{doc.metadata['page']}"
                            )

                    # DOCUMENT NAME
                    with col2:

                        if "source" in doc.metadata:

                            source_name = (
                                doc.metadata["source"]
                                .split("\\")[-1]
                            )

                            st.caption(
                                f" File: {source_name}"
                            )

                    # DOCUMENT SNIPPET

                    st.info(
                        doc.page_content[:300] + "..."
                    )

                    st.divider()

# ============================================
# CHAT INPUT
# ============================================

query = st.chat_input(
    "Ask your medical question..."
)

# ============================================
# PROCESS USER QUERY
# ============================================

if query:

    # ============================================
    # SHOW USER MESSAGE
    # ============================================

    with st.chat_message("user"):

        st.markdown(query)

    # ============================================
    # SAVE USER MESSAGE
    # ============================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # ============================================
    # BUILD CHAT HISTORY
    # ============================================

    chat_history = []

    messages = st.session_state.messages

    for i in range(len(messages) - 1):

        current_msg = messages[i]
        next_msg = messages[i + 1]

        if (
            current_msg["role"] == "user"
            and next_msg["role"] == "assistant"
        ):

            chat_history.append(
                {
                    "question": current_msg["content"],
                    "answer": next_msg["content"]
                }
            )

    # ============================================
    # GENERATE ANSWER
    # ============================================

    try:

        with st.chat_message("assistant"):

            with st.spinner(
                "Generating medical answer..."
            ):

                answer, results = (
                    ask_medical_question_langgraph(
                        query,
                        chat_history
                    )
                )

            # ============================================
            # DISPLAY ANSWER
            # ============================================

            st.markdown(answer)

            # ============================================
            # DISPLAY SOURCES
            # ============================================

            with st.expander(
                " View Retrieved Sources"
            ):

                for i, doc in enumerate(results):

                    st.markdown(
                        f"##  Source {i+1}"
                    )

                    col1, col2 = st.columns(2)

                    # PAGE NUMBER
                    with col1:

                        if "page" in doc.metadata:

                            st.caption(
                                f" Page: "
                                f"{doc.metadata['page']}"
                            )

                    # DOCUMENT NAME
                    with col2:

                        if "source" in doc.metadata:

                            source_name = (
                                doc.metadata["source"]
                                .split("\\")[-1]
                            )

                            st.caption(
                                f" File: {source_name}"
                            )

                    # SOURCE CONTENT

                    st.info(
                        doc.page_content[:300] + "..."
                    )

                    st.divider()

        # ============================================
        # SAVE ASSISTANT MESSAGE
        # ============================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": results
            }
        )

    # ============================================
    # ERROR HANDLING
    # ============================================

    except Exception as e:

        st.error(f" Error: {str(e)}")

        with st.expander(
            "View Technical Error"
        ):

            st.code(
                traceback.format_exc()
            )