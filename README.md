# 🧠 Second Brain

> A personal knowledge assistant that ingests everything you learn and becomes an AI that "thinks like you."

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agents-orange.svg)](https://langchain-ai.github.io/langgraph/)

---

## 📖 Overview

Second Brain is an AI-powered knowledge management system that:

- **Ingests** your notes, articles, PDFs, and web content
- **Understands** relationships between concepts using embeddings and vector search
- **Answers** questions using RAG (Retrieval-Augmented Generation)
- **Acts** as an intelligent agent that can search, browse, and take notes for you

This project is designed as a learning journey through AI engineering, progressing from beginner concepts to advanced production systems.

---

## 🎯 Features

- 💬 Conversational AI with memory
- 📄 Multi-format document ingestion (PDF, Markdown, Web)
- 🔍 Semantic search over your knowledge base
- 🤖 Autonomous agent with tool use
- 📊 Retrieval evaluation and optimization
- 🚀 Production-ready API

---

## 🗺️ Project Roadmap

This project is structured in 8 phases, each building on the previous:

| Phase | Focus | Status |
|-------|-------|--------|
| [Phase 1](#phase-1-basic-chat-api) | Basic Chat API | 🔲 Not Started |
| [Phase 2](#phase-2-prompt-engineering--memory) | Prompt Engineering & Memory | 🔲 Not Started |
| [Phase 3](#phase-3-document-ingestion--chunking) | Document Ingestion & Chunking | 🔲 Not Started |
| [Phase 4](#phase-4-embeddings--vector-database) | Embeddings & Vector Database | 🔲 Not Started |
| [Phase 5](#phase-5-rag-pipeline) | RAG Pipeline | 🔲 Not Started |
| [Phase 6](#phase-6-agents--tools-with-langgraph) | Agents & Tools with LangGraph | 🔲 Not Started |
| [Phase 7](#phase-7-advanced-retrieval--evaluation) | Advanced Retrieval & Evaluation | 🔲 Not Started |
| [Phase 8](#phase-8-custom-models--production) | Custom Models & Production | 🔲 Not Started |

---

## 🏗️ Architecture

```
Phase 1-2:  User → LLM API → Response

Phase 3-5:  User → Retriever → Vector DB
                      ↓
                  LLM API → Response

Phase 6+:   User → Agent (LangGraph)
                      ↓
            ┌─────────┼─────────┐
            ↓         ↓         ↓
         Search    Tools     RAG
            ↓         ↓         ↓
            └─────────┼─────────┘
                      ↓
                  Response
```

---

## 📚 Phase Details

### Phase 1: Basic Chat API
**Timeline:** Week 1 | **Difficulty:** Beginner

Build a simple CLI chatbot that calls an LLM API.

**Learning Objectives:**
- API authentication and requests
- Prompt structure (system/user/assistant)
- Streaming responses
- Basic error handling

**Key Files:**
```
src/
└── chat.py          # Basic chat implementation
```

---

### Phase 2: Prompt Engineering & Memory
**Timeline:** Week 2 | **Difficulty:** Beginner

Add conversation memory and experiment with different prompting techniques.

**Learning Objectives:**
- Context window management
- Few-shot prompting
- System prompt design
- Token counting and truncation strategies

**Key Files:**
```
src/
├── chat.py
├── memory.py        # Conversation history management
└── prompts/
    └── templates.py # Prompt templates
```

---

### Phase 3: Document Ingestion & Chunking
**Timeline:** Weeks 3-4 | **Difficulty:** Intermediate

Ingest markdown notes, PDFs, and web articles into the system.

**Learning Objectives:**
- Text extraction (PyMuPDF, BeautifulSoup)
- Chunking strategies (fixed, recursive, semantic)
- Metadata extraction and preservation
- Understanding chunk size and overlap tradeoffs

**Key Files:**
```
src/
├── ingestion/
│   ├── pdf.py       # PDF extraction
│   ├── markdown.py  # Markdown processing
│   └── web.py       # Web scraping
└── chunking/
    ├── fixed.py     # Fixed-size chunking
    ├── recursive.py # Recursive text splitter
    └── semantic.py  # Semantic chunking
```

---

### Phase 4: Embeddings & Vector Database
**Timeline:** Weeks 5-6 | **Difficulty:** Intermediate

Embed document chunks and store them in a vector database for retrieval.

**Learning Objectives:**
- How embeddings represent semantic meaning
- Embedding models (OpenAI, sentence-transformers)
- Vector database setup (Chroma → Pinecone)
- Similarity search and distance metrics (cosine, euclidean)

**Key Files:**
```
src/
├── embeddings/
│   ├── openai.py    # OpenAI embeddings
│   └── local.py     # Sentence-transformers
└── vectorstore/
    ├── chroma.py    # Chroma implementation
    └── pinecone.py  # Pinecone implementation
```

---

### Phase 5: RAG Pipeline
**Timeline:** Weeks 7-8 | **Difficulty:** Intermediate

Answer questions using retrieved context from your knowledge base.

**Learning Objectives:**
- Retrieval + generation flow
- Context injection into prompts
- Citation and source tracking
- Hallucination reduction techniques

**Key Files:**
```
src/
├── rag/
│   ├── retriever.py # Document retrieval
│   ├── generator.py # Response generation
│   └── chain.py     # RAG chain orchestration
└── prompts/
    └── rag.py       # RAG-specific prompts
```

---

### Phase 6: Agents & Tools with LangGraph
**Timeline:** Weeks 9-11 | **Difficulty:** Advanced

Build an agent that can search your knowledge, browse the web, run code, and take notes.

**Learning Objectives:**
- Agent architectures (ReAct pattern)
- LangGraph for stateful workflows
- Tool definition and function calling
- Conditional routing and control flow
- State management across interactions

**Key Files:**
```
src/
├── agents/
│   ├── graph.py     # LangGraph definition
│   ├── state.py     # Agent state schema
│   └── nodes.py     # Graph nodes
└── tools/
    ├── search.py    # Knowledge base search
    ├── web.py       # Web browsing
    ├── code.py      # Code execution
    └── notes.py     # Note taking
```

---

### Phase 7: Advanced Retrieval & Evaluation
**Timeline:** Weeks 12-14 | **Difficulty:** Advanced

Improve retrieval quality and build a framework to measure it.

**Learning Objectives:**
- Hybrid search (keyword + semantic)
- Reranking with cross-encoders
- Query transformation (HyDE, multi-query)
- Evaluation metrics (RAGAS, faithfulness, relevance)
- Building evaluation datasets

**Key Files:**
```
src/
├── retrieval/
│   ├── hybrid.py    # Hybrid search
│   ├── rerank.py    # Reranking models
│   └── transform.py # Query transformation
└── evaluation/
    ├── metrics.py   # Evaluation metrics
    ├── datasets.py  # Test dataset management
    └── runner.py    # Evaluation runner
```

---

### Phase 8: Custom Models & Production
**Timeline:** Weeks 15+ | **Difficulty:** Advanced

Fine-tune embeddings for your domain, add observability, and deploy to production.

**Learning Objectives:**
- TensorFlow/PyTorch for embedding fine-tuning
- Contrastive learning (triplet loss, InfoNCE)
- Observability and tracing (Langfuse/LangSmith)
- Caching strategies for performance
- API design with FastAPI
- Containerization and deployment

**Key Files:**
```
src/
├── training/
│   ├── dataset.py   # Training data preparation
│   ├── model.py     # Model architecture
│   └── train.py     # Training loop
├── api/
│   ├── main.py      # FastAPI application
│   └── routes/      # API endpoints
└── observability/
    ├── tracing.py   # Request tracing
    └── metrics.py   # Performance metrics
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.11+ |
| **LLM** | Anthropic Claude / OpenAI GPT |
| **Embeddings** | sentence-transformers → fine-tuned |
| **Vector DB** | Chroma (local) → Pinecone (production) |
| **Agents** | LangGraph |
| **Training** | TensorFlow / PyTorch |
| **API** | FastAPI |
| **Observability** | Langfuse |

---

## 📁 Project Structure

```
second-brain/
├── src/
│   ├── chat.py
│   ├── memory.py
│   ├── ingestion/
│   ├── chunking/
│   ├── embeddings/
│   ├── vectorstore/
│   ├── rag/
│   ├── agents/
│   ├── tools/
│   ├── retrieval/
│   ├── evaluation/
│   ├── training/
│   ├── api/
│   └── observability/
├── tests/
├── notebooks/           # Experimentation notebooks
├── data/
│   ├── raw/            # Original documents
│   └── processed/      # Chunked documents
├── models/             # Trained models
├── configs/            # Configuration files
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- An API key for Anthropic or OpenAI

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/second-brain.git
cd second-brain

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Quick Start

```bash
# Run the basic chat (Phase 1)
python src/chat.py

# Run the RAG pipeline (Phase 5+)
python src/rag/chain.py

# Run the agent (Phase 6+)
python src/agents/graph.py

# Start the API server (Phase 8)
uvicorn src.api.main:app --reload
```

---

## 📊 Evaluation Results

*Results will be added as phases are completed.*

| Metric | Baseline | Current |
|--------|----------|---------|
| Retrieval Recall@5 | - | - |
| Answer Faithfulness | - | - |
| Answer Relevance | - | - |
| Latency (p95) | - | - |

---

## 📝 What I Learned

*This section documents key learnings from each phase.*

### Phase 1
- *Coming soon...*

### Phase 2
- *Coming soon...*


<p align="center">
  Built with ❤️ as a journey through AI Engineering
</p>
