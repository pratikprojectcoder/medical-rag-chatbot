import os

import streamlit as st

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq

# ============================================
# LOAD ENV VARIABLES
# ============================================

load_dotenv()
print(os.getenv("GROQ_API_KEY"))

# ============================================
# GROQ CLIENT
# ============================================

api_key = os.getenv("GROQ_API_KEY")

if api_key:
    api_key = api_key.strip()

client = Groq(
    api_key=api_key
)

# ============================================
# LOAD EMBEDDING MODEL
# ============================================

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

print("Embedding model loaded successfully!")

# ============================================
# LOAD VECTOR DATABASE
# ============================================

@st.cache_resource
def load_vectorstore():

    return FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

print("FAISS vector database loaded successfully!")

# ============================================
# QUERY REWRITING FUNCTION
# ============================================

def rewrite_query(query, chat_history):

    # ============================================
    # BUILD CONVERSATION HISTORY
    # ============================================

    history_text = ""

    for chat in chat_history[-3:]:

        history_text += f"""
        User: {chat['question']}
        Assistant: {chat['answer']}
        """

    # ============================================
    # QUERY REWRITE PROMPT
    # ============================================

    rewrite_prompt = f"""
    You are a medical query rewriting assistant.

    Convert conversational medical questions
    into complete standalone medical queries.

    Use conversation history carefully.

    If the question already contains enough context,
    return it unchanged.

    Conversation History:
    {history_text}

    Current User Question:
    {query}

    ONLY return the rewritten standalone query.
    """

    # ============================================
    # LLM CALL
    # ============================================

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=80,
        messages=[
            {
                "role": "user",
                "content": rewrite_prompt
            }
        ]
    )

    rewritten_query = (
        response.choices[0]
        .message.content
        .strip()
    )

    print("\n==============================")
    print("REWRITTEN QUERY")
    print("==============================")
    print(rewritten_query)

    return rewritten_query

# ============================================
# MAIN RAG FUNCTION
# ============================================

def ask_medical_question(query, chat_history):

    # ============================================
    # QUERY REWRITING
    # ============================================

    rewritten_query = rewrite_query(
        query,
        chat_history
    )

    # ============================================
    # SEMANTIC SEARCH
    # ============================================

    results = vectorstore.similarity_search_with_score(
        rewritten_query,
        k=8
    )

    # ============================================
    # CONTEXT VALIDATION + RERANKING
    # ============================================

    filtered_docs = []

    seen_content = set()

    for doc, score in results:

        content = doc.page_content.strip()

        # SKIP SMALL CHUNKS

        if len(content) < 120:
            continue

        # SKIP DUPLICATES

        content_key = content[:200]

        if content_key in seen_content:
            continue

        seen_content.add(content_key)

        # KEEP RELEVANT DOC

        filtered_docs.append(doc)

    # LIMIT DOCS

    filtered_docs = filtered_docs[:5]

    # ============================================
    # BUILD CONTEXT
    # ============================================

    context = "\n\n".join(
        [
            doc.page_content
            for doc in filtered_docs
        ]
    )

    # ============================================
    # BASIC EMERGENCY DETECTION
    # ============================================

    emergency_keywords = [
        "chest pain",
        "heart attack",
        "stroke",
        "suicidal",
        "can't breathe",
        "difficulty breathing",
        "severe bleeding",
        "unconscious",
        "seizure"
    ]

    emergency_flag = any(
        keyword in query.lower()
        for keyword in emergency_keywords
    )

    # ============================================
    # FINAL PROMPT
    # ============================================

    prompt = f"""
    You are a professional and responsible
    medical assistant.

    Answer the user's medical question clearly,
    naturally, and concisely using ONLY the
    provided medical context.

    IMPORTANT RULES:

    - Answer ONLY using the provided context.
    - Do NOT invent medical facts.
    - Do NOT hallucinate diagnoses or treatments.
    - Never claim certainty for diagnoses.
    - Encourage professional consultation
      for serious conditions.
    - Recommend emergency care for
      dangerous symptoms.
    - Keep answers concise and medically responsible.
    - Avoid unnecessary repetition.
    - Do NOT mention embeddings,
      vector databases, retrieval systems,
      or internal AI operations.
    - If information is insufficient, say:
      "I could not find sufficient medical information."

    Medical Context:
    {context}

    User Question:
    {query}

    Emergency Warning:
    {"Potential medical emergency detected." if emergency_flag else "No emergency warning needed."}

    Answer:
    """

    # ============================================
    # GENERATE ANSWER
    # ============================================

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=350,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # ============================================
    # FINAL ANSWER
    # ============================================

    answer = (
        response.choices[0]
        .message.content
        .strip()
    )

    # ============================================
    # MEDICAL DISCLAIMER
    # ============================================

    answer += """

⚠️ This information is for educational purposes
and is not a substitute for professional
medical advice.
"""

    return answer, filtered_docs