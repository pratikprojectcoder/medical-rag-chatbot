# Medical RAG Chatbot

An advanced conversational Medical AI Assistant built using LangGraph, FAISS, Groq LLM, Streamlit, and Retrieval-Augmented Generation (RAG).

The system performs semantic retrieval over medical textbooks and generates context-grounded medical responses with conversational memory, query rewriting, safety-aware workflows, and source citations.

# Features

* Conversational medical AI
* Multi-turn conversation support
* Context-aware follow-up questions
* Retrieval-Augmented Generation (RAG)
* Semantic search over medical textbooks
* Source-grounded responses
* Reduced hallucinations
* LangGraph workflow orchestration
* Query rewriting
* Safety validation
* Emergency symptom detection
* Streamlit chat interface
* Expandable retrieved sources
* Conversation history
* Cached FAISS loading
* Cached embedding model loading
* Environment-variable-based API security

# System Architecture

```text
Medical PDFs
      ↓
Document Loader
      ↓
Text Chunking
      ↓
Embedding Model
      ↓
FAISS Vector Database
      ↓
LangGraph Workflow
      ↓
Query Rewriting
      ↓
Semantic Retrieval
      ↓
Validation + Safety
      ↓
LLM Generation
      ↓
Final Medical Answer
```

# Tech Stack

| Component              | Technology            |
| ---------------------- | --------------------- |
| Frontend               | Streamlit             |
| Workflow Orchestration | LangGraph             |
| RAG Framework          | LangChain             |
| Vector Database        | FAISS                 |
| Embeddings             | Sentence Transformers |
| LLM                    | Groq Llama 3          |
| PDF Parsing            | PyMuPDF               |
| Language               | Python                |

# Project Structure

```text
medical-rag-chatbot/
│
├── app.py
├── rag_pipeline.py
├── langgraph_pipeline.py
├── ingest.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── faiss_index/
│
├── data/
│   └── medical_pdfs/
│
└── venv/
```

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/medical-rag-chatbot.git
```

```bash
cd medical-rag-chatbot
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

# Environment Variables

Create a `.env` file:

```text
GROQ_API_KEY=your_groq_api_key
```

# Add Medical PDFs

Place medical PDFs inside:

```text
data/medical_pdfs/
```

# Build Vector Database

Run:

```bash
python ingest.py
```

This will:

* load PDFs
* split text into chunks
* generate embeddings
* create FAISS vector database

# Run Application

```bash
streamlit run app.py
```

# Example Queries

```text
What is asthma?
```

```text
What causes it?
```

```text
How is it treated?
```

```text
I have chest pain and difficulty breathing
```

# LangGraph Workflow

```text
User Query
      ↓
Rewrite Node
      ↓
Retrieval Node
      ↓
Validation Node
      ↓
Safety Node
      ↓
Context Node
      ↓
Generation Node
      ↓
Final Medical Answer
```

# Safety Features

* Emergency symptom detection
* Medical disclaimers
* Retrieval-grounded answers
* Reduced hallucinations
* Encourages professional consultation

# Resume Description

Medical RAG Chatbot System | Python, LangGraph, FAISS, Groq LLM, Streamlit

* Built an advanced conversational Medical AI Assistant using Retrieval-Augmented Generation (RAG) with semantic search over 600+ pages of medical textbooks and reference documents.

* Implemented a modular LangGraph workflow architecture with dedicated nodes for query rewriting, semantic retrieval, context validation, safety checks, and response generation.

* Developed a FAISS-based vector database pipeline using Sentence-Transformer embeddings (`all-MiniLM-L6-v2`) for efficient semantic similarity search and source-grounded medical responses.

* Designed conversational memory and query rewriting mechanisms to support multi-turn interactions and context-aware follow-up questions.

* Added a medical safety layer with emergency symptom detection and responsible-response guardrails to reduce hallucinations and encourage professional medical consultation for critical conditions.

* Built an interactive Streamlit interface featuring conversational chat UI, retrieved-source citations, expandable context display, session-based chat history, and real-time response generation.

* Optimized application performance using cached embedding models and vectorstore loading, environment-variable-based API security, and production-style project structuring for deployment readiness.

# Future Improvements

* Hybrid search (BM25 + Vector Search)
* Voice-based interaction
* Multi-agent workflows
* Medical image analysis
* FastAPI backend
* Docker deployment
* Authentication system
* Database-based chat storage

# License

This project is for educational and research purposes only.

Medical information generated by the system should not be considered professional medical advice.
