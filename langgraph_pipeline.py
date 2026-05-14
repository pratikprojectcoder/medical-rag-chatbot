from typing import TypedDict

from langgraph.graph import StateGraph, END

from rag_pipeline import (
    rewrite_query,
    vectorstore,
    client
)

# ============================================
# GRAPH STATE
# ============================================

class MedicalState(TypedDict):

    query: str

    chat_history: list

    rewritten_query: str

    retrieved_docs: list

    filtered_docs: list

    context: str

    emergency_flag: bool

    answer: str

# ============================================
# QUERY REWRITE NODE
# ============================================

def rewrite_node(state):

    rewritten_query = rewrite_query(
        state["query"],
        state["chat_history"]
    )

    state["rewritten_query"] = rewritten_query

    return state

# ============================================
# RETRIEVAL NODE
# ============================================

def retrieval_node(state):

    results = vectorstore.similarity_search_with_score(
        state["rewritten_query"],
        k=8
    )

    state["retrieved_docs"] = results

    return state

# ============================================
# VALIDATION NODE
# ============================================

def validation_node(state):

    filtered_docs = []

    seen_content = set()

    for doc, score in state["retrieved_docs"]:

        content = doc.page_content.strip()

        if len(content) < 120:
            continue

        content_key = content[:200]

        if content_key in seen_content:
            continue

        seen_content.add(content_key)

        filtered_docs.append(doc)

    filtered_docs = filtered_docs[:5]

    state["filtered_docs"] = filtered_docs

    return state

# ============================================
# SAFETY NODE
# ============================================

def safety_node(state):

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
        keyword in state["query"].lower()
        for keyword in emergency_keywords
    )

    state["emergency_flag"] = emergency_flag

    return state

# ============================================
# CONTEXT NODE
# ============================================

def context_node(state):

    context = "\n\n".join(
        [doc.page_content for doc in state["filtered_docs"]]
    )

    state["context"] = context

    return state

# ============================================
# GENERATION NODE
# ============================================

def generation_node(state):

    prompt = f"""
    You are a professional and responsible medical assistant.

    Answer the user's medical question clearly,
    naturally, and concisely using ONLY the
    provided medical context.

    IMPORTANT RULES:

    - Answer ONLY using the provided medical context.
    - Do NOT invent medical facts.
    - Do NOT hallucinate treatments or diagnoses.
    - If the context is insufficient, say:
      "I could not find sufficient medical information."
    - Never claim certainty for a diagnosis.
    - Do NOT pretend to be a licensed doctor.
    - Encourage professional medical consultation
      for serious conditions.
    - If symptoms appear dangerous or life-threatening,
      recommend immediate medical attention.
    - Keep answers concise, clear, and medically responsible.
    - Avoid unnecessary repetition.

    Medical Context:
    {state["context"]}

    User Question:
    {state["query"]}

    Emergency Warning:
    {"Potential medical emergency detected." if state["emergency_flag"] else "No emergency warning needed."}

    Answer:
    """

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

    answer = response.choices[0].message.content.strip()

    answer += "\n\n⚠️ This information is for educational purposes and is not a substitute for professional medical advice."

    state["answer"] = answer

    return state

# ============================================
# BUILD GRAPH
# ============================================

graph = StateGraph(MedicalState)

graph.add_node("rewrite", rewrite_node)

graph.add_node("retrieve", retrieval_node)

graph.add_node("validate", validation_node)

graph.add_node("safety", safety_node)

graph.add_node("context", context_node)

graph.add_node("generate", generation_node)

# ============================================
# EDGES
# ============================================

graph.set_entry_point("rewrite")

graph.add_edge("rewrite", "retrieve")

graph.add_edge("retrieve", "validate")

graph.add_edge("validate", "safety")

graph.add_edge("safety", "context")

graph.add_edge("context", "generate")

graph.add_edge("generate", END)

# ============================================
# COMPILE GRAPH
# ============================================

medical_graph = graph.compile()

# ============================================
# MAIN FUNCTION
# ============================================

def ask_medical_question_langgraph(query, chat_history):

    initial_state = {
        "query": query,
        "chat_history": chat_history,
        "rewritten_query": "",
        "retrieved_docs": [],
        "filtered_docs": [],
        "context": "",
        "emergency_flag": False,
        "answer": ""
    }

    final_state = medical_graph.invoke(initial_state)

    return (
        final_state["answer"],
        final_state["filtered_docs"]
    )