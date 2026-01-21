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

- 💬 Conversational AI with memory and streaming
- 📄 Multi-format document ingestion (PDF, Markdown, Web, Images)
- 🔍 Semantic search with caching and hybrid retrieval
- 🤖 Autonomous agent with tool use and intent routing
- 🛡️ Production guardrails and safety measures
- 📊 Comprehensive evaluation and observability
- 💰 Cost tracking and model routing
- 🚀 Production-ready API with local model support

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

Phase 3-5:  User → Guardrails → Retriever → Vector DB
                                    ↓
                                LLM API → Response

Phase 6+:   User → Guardrails → Intent Classifier
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
                Simple Q&A    RAG Pipeline    Agent (LangGraph)
                    ↓               ↓               ↓
                    │               │       ┌───────┼───────┐
                    │               │       ↓       ↓       ↓
                    │               │    Search   Tools   RAG
                    ↓               ↓       ↓       ↓       ↓
                    └───────────────┼───────────────────────┘
                                    ↓
                            Model Router
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
                Haiku           Sonnet          Local
              (Simple)        (Complex)       (Private)
                    ↓               ↓               ↓
                    └───────────────┼───────────────┘
                                    ↓
                    Cache → Response → Feedback Loop
```

---

## 🧠 AI Engineering Concepts Covered

| Category | Concepts |
|----------|----------|
| **Core LLM** | API calls, streaming, prompting, structured outputs, token management |
| **RAG** | Chunking, embeddings, vector DBs, retrieval, reranking, hybrid search |
| **Agents** | LangGraph, tool use, ReAct pattern, state management, intent routing |
| **Safety** | Guardrails, input validation, PII detection, prompt injection defense |
| **Optimization** | Caching (semantic + exact), model routing, cost tracking |
| **Evaluation** | RAGAS, LLM-as-judge, synthetic data generation, A/B testing |
| **ML Training** | Embedding fine-tuning, contrastive learning, distillation basics |
| **Production** | FastAPI, observability, local models, feedback loops, batch processing |

---

## 📚 Phase Details

### Phase 1: Basic Chat API
**Timeline:** Week 1 | **Difficulty:** Beginner

Build a CLI chatbot with streaming responses and proper error handling.

**Learning Objectives:**
- API authentication and requests
- Prompt structure (system/user/assistant)
- Streaming responses (real-time token output)
- Error handling with exponential backoff
- Retry logic and fallback strategies
- Graceful degradation patterns

**Key Files:**
```
src/
├── chat.py              # Basic chat implementation
└── utils/
    ├── streaming.py     # Stream handling utilities
    └── retry.py         # Retry logic with backoff
```
---

### Phase 2: Prompt Engineering & Memory
**Timeline:** Week 2-3 | **Difficulty:** Beginner

Add conversation memory, structured outputs, and cost awareness.

**Learning Objectives:**
- Context window management
- Few-shot prompting techniques
- System prompt design
- Structured outputs with Pydantic
- Token counting and truncation strategies
- Cost tracking per conversation

**Key Files:**
```
src/
├── chat.py
├── memory.py            # Conversation history management
├── prompts/
│   └── templates.py     # Prompt templates
├── schemas/
│   └── outputs.py       # Pydantic output models
└── utils/
    ├── tokens.py        # Token counting utilities
    └── costs.py         # Cost tracking
```
---

### Phase 3: Document Ingestion & Chunking
**Timeline:** Weeks 4-5 | **Difficulty:** Intermediate

Ingest documents including multi-modal content with batch processing.

**Learning Objectives:**
- Text extraction (PyMuPDF, BeautifulSoup)
- Image extraction and processing
- Chunking strategies (fixed, recursive, semantic)
- Metadata extraction and preservation
- Understanding chunk size and overlap tradeoffs
- Batch processing for large document sets
- Async patterns for parallel ingestion

**Key Files:**
```
src/
├── ingestion/
│   ├── pdf.py           # PDF extraction (text + images)
│   ├── markdown.py      # Markdown processing
│   ├── web.py           # Web scraping
│   ├── images.py        # Image handling for multi-modal
│   └── batch.py         # Batch processing orchestration
└── chunking/
    ├── fixed.py         # Fixed-size chunking
    ├── recursive.py     # Recursive text splitter
    └── semantic.py      # Semantic chunking
```
---

### Phase 4: Embeddings & Vector Database
**Timeline:** Weeks 6-7 | **Difficulty:** Intermediate

Embed document chunks and store them in a vector database for retrieval.

**Learning Objectives:**
- How embeddings represent semantic meaning
- Embedding models (OpenAI, sentence-transformers)
- Vector database setup (Chroma → Pinecone)
- Similarity search and distance metrics (cosine, euclidean, dot product)
- Index optimization and performance tuning
- Metadata filtering strategies

**Key Files:**
```
src/
├── embeddings/
│   ├── base.py          # Embedding interface
│   ├── openai.py        # OpenAI embeddings
│   └── local.py         # Sentence-transformers
└── vectorstore/
    ├── base.py          # Vector store interface
    ├── chroma.py        # Chroma implementation
    └── pinecone.py      # Pinecone implementation
```
---

### Phase 5: RAG Pipeline
**Timeline:** Weeks 8-10 | **Difficulty:** Intermediate

Build a secure RAG pipeline with guardrails and safety measures.

**Learning Objectives:**
- Retrieval + generation flow
- Context injection into prompts
- Citation and source tracking
- Hallucination reduction techniques
- Input validation and sanitization
- PII detection and handling
- Prompt injection defense
- Output guardrails

**Key Files:**
```
src/
├── rag/
│   ├── retriever.py     # Document retrieval
│   ├── generator.py     # Response generation
│   └── chain.py         # RAG chain orchestration
├── guardrails/
│   ├── input.py         # Input validation
│   ├── pii.py           # PII detection
│   ├── injection.py     # Prompt injection defense
│   └── output.py        # Output validation
└── prompts/
    └── rag.py           # RAG-specific prompts
```
---

### Phase 6: Agents & Tools with LangGraph
**Timeline:** Weeks 11-14 | **Difficulty:** Advanced

Build an agent with intent classification and intelligent routing.

**Learning Objectives:**
- Agent architectures (ReAct pattern)
- LangGraph for stateful workflows
- Tool definition and function calling
- Conditional routing and control flow
- State management across interactions
- Intent classification for routing
- When to use agent vs simple RAG

**Key Files:**
```
src/
├── agents/
│   ├── graph.py         # LangGraph definition
│   ├── state.py         # Agent state schema
│   └── nodes.py         # Graph nodes
├── routing/
│   ├── classifier.py    # Intent classification
│   └── router.py        # Query routing logic
└── tools/
    ├── base.py          # Tool interface
    ├── search.py        # Knowledge base search
    ├── web.py           # Web browsing
    ├── code.py          # Code execution
    └── notes.py         # Note taking
```
---

### Phase 7: Advanced Retrieval & Evaluation
**Timeline:** Weeks 15-18 | **Difficulty:** Advanced

Optimize retrieval with caching, model routing, and comprehensive evaluation.

**Learning Objectives:**
- Hybrid search (keyword + semantic)
- Reranking with cross-encoders
- Query transformation (HyDE, multi-query)
- Semantic caching for cost reduction
- Model routing based on complexity
- Evaluation metrics (RAGAS, faithfulness, relevance)
- LLM-as-judge evaluation patterns
- Synthetic data generation for testing
- A/B testing retrieval strategies

**Key Files:**
```
src/
├── retrieval/
│   ├── hybrid.py        # Hybrid search
│   ├── rerank.py        # Reranking models
│   └── transform.py     # Query transformation (HyDE)
├── caching/
│   ├── exact.py         # Exact match cache
│   └── semantic.py      # Semantic similarity cache
├── routing/
│   └── model_router.py  # Model selection logic
├── evaluation/
│   ├── metrics.py       # Evaluation metrics
│   ├── llm_judge.py     # LLM-as-judge evaluator
│   ├── synthetic.py     # Synthetic data generation
│   ├── datasets.py      # Test dataset management
│   └── runner.py        # Evaluation runner
└── experiments/
    └── ab_testing.py    # A/B testing framework
```

---

### Phase 8: Custom Models & Production
**Timeline:** Weeks 19+ | **Difficulty:** Advanced

Fine-tune embeddings, add local model support, deploy with full observability.

**Learning Objectives:**
- TensorFlow/PyTorch for embedding fine-tuning
- Contrastive learning (triplet loss, InfoNCE)
- Knowledge distillation basics
- Local model deployment (Ollama, vLLM)
- Observability and tracing (Langfuse/LangSmith)
- Caching strategies for performance
- API design with FastAPI
- Containerization and deployment
- Feedback collection and improvement loops
- Cost dashboards and monitoring

**Key Files:**
```
src/
├── training/
│   ├── dataset.py       # Training data preparation
│   ├── model.py         # Model architecture
│   ├── losses.py        # Contrastive losses
│   ├── train.py         # Training loop
│   └── distill.py       # Knowledge distillation
├── inference/
│   ├── local.py         # Ollama/vLLM integration
│   └── quantization.py  # Model quantization
├── api/
│   ├── main.py          # FastAPI application
│   ├── routes/          # API endpoints
│   └── middleware/      # Auth, rate limiting
├── observability/
│   ├── tracing.py       # Request tracing
│   ├── metrics.py       # Performance metrics
│   └── dashboard.py     # Cost dashboard
└── feedback/
    ├── collector.py     # Feedback collection
    └── improver.py      # Feedback-based improvements
```

**Key Concepts:**
```python
# Contrastive learning for embeddings
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature
    
    def forward(self, anchor, positive, negatives):
        # Compute similarities
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sims = F.cosine_similarity(
            anchor.unsqueeze(1), 
            negatives, 
            dim=2
        )
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
        logits = logits / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        
        return F.cross_entropy(logits, labels)

# Feedback loop
class FeedbackCollector:
    def record(self, query: str, response: str, feedback: Feedback):
        self.db.insert({
            "query": query,
            "response": response,
            "rating": feedback.rating,  # thumbs up/down
            "timestamp": now()
        })
    
    def get_improvement_candidates(self) -> list[dict]:
        # Find queries with negative feedback
        return self.db.query(
            "SELECT * FROM feedback WHERE rating = 'negative'"
        )

# Local model integration
class LocalLLM:
    def __init__(self, model: str = "llama3.2"):
        self.client = ollama.Client()
        self.model = model
    
    async def generate(self, prompt: str) -> str:
        response = await self.client.generate(
            model=self.model,
            prompt=prompt
        )
        return response["response"]
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.11+ |
| **LLM (Cloud)** | Anthropic Claude / OpenAI GPT |
| **LLM (Local)** | Ollama, vLLM |
| **Embeddings** | sentence-transformers → fine-tuned |
| **Vector DB** | Chroma (local) → Pinecone (production) |
| **Agents** | LangGraph |
| **Training** | TensorFlow / PyTorch |
| **API** | FastAPI |
| **Observability** | Langfuse, Prometheus, Grafana |
| **Caching** | Redis |

---

## 📁 Project Structure

```
second-brain/
├── src/
│   ├── chat.py
│   ├── memory.py
│   ├── ingestion/
│   │   ├── pdf.py
│   │   ├── markdown.py
│   │   ├── web.py
│   │   ├── images.py
│   │   └── batch.py
│   ├── chunking/
│   │   ├── fixed.py
│   │   ├── recursive.py
│   │   └── semantic.py
│   ├── embeddings/
│   │   ├── base.py
│   │   ├── openai.py
│   │   └── local.py
│   ├── vectorstore/
│   │   ├── base.py
│   │   ├── chroma.py
│   │   └── pinecone.py
│   ├── rag/
│   │   ├── retriever.py
│   │   ├── generator.py
│   │   └── chain.py
│   ├── guardrails/
│   │   ├── input.py
│   │   ├── pii.py
│   │   ├── injection.py
│   │   └── output.py
│   ├── agents/
│   │   ├── graph.py
│   │   ├── state.py
│   │   └── nodes.py
│   ├── routing/
│   │   ├── classifier.py
│   │   ├── router.py
│   │   └── model_router.py
│   ├── tools/
│   │   ├── base.py
│   │   ├── search.py
│   │   ├── web.py
│   │   ├── code.py
│   │   └── notes.py
│   ├── retrieval/
│   │   ├── hybrid.py
│   │   ├── rerank.py
│   │   └── transform.py
│   ├── caching/
│   │   ├── exact.py
│   │   └── semantic.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── llm_judge.py
│   │   ├── synthetic.py
│   │   ├── datasets.py
│   │   └── runner.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── losses.py
│   │   ├── train.py
│   │   └── distill.py
│   ├── inference/
│   │   ├── local.py
│   │   └── quantization.py
│   ├── api/
│   │   ├── main.py
│   │   └── routes/
│   ├── observability/
│   │   ├── tracing.py
│   │   ├── metrics.py
│   │   └── dashboard.py
│   ├── feedback/
│   │   ├── collector.py
│   │   └── improver.py
│   ├── schemas/
│   │   └── outputs.py
│   ├── prompts/
│   │   ├── templates.py
│   │   └── rag.py
│   └── utils/
│       ├── streaming.py
│       ├── retry.py
│       ├── tokens.py
│       └── costs.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
├── notebooks/               # Experimentation notebooks
├── data/
│   ├── raw/                # Original documents
│   ├── processed/          # Chunked documents
│   └── eval/               # Evaluation datasets
├── models/                 # Trained models
├── configs/                # Configuration files
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- An API key for Anthropic or OpenAI
- Docker (optional, for local models)

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
python -m src.chat

# Run the RAG pipeline (Phase 5+)
python -m src.rag.chain

# Run the agent (Phase 6+)
python -m src.agents.graph

# Start the API server (Phase 8)
uvicorn src.api.main:app --reload

# Run with local models (Phase 8)
docker-compose up ollama
python -m src.inference.local
```

---

## 📊 Evaluation Results

*Results will be added as phases are completed.*

| Metric | Baseline | Current | Target |
|--------|----------|---------|--------|
| Retrieval Recall@5 | - | - | >0.85 |
| Answer Faithfulness | - | - | >0.90 |
| Answer Relevance | - | - | >0.85 |
| Latency (p95) | - | - | <2s |
| Cache Hit Rate | - | - | >40% |
| Cost per Query | - | - | <$0.01 |

---

## 💰 Cost Tracking

*Will be populated as the project progresses.*

| Model | Queries | Tokens | Cost |
|-------|---------|--------|------|
| Claude Haiku | - | - | - |
| Claude Sonnet | - | - | - |
| Local (Ollama) | - | - | $0 |
| **Total** | - | - | - |

---

## 📝 What I Learned

*This section documents key learnings from each phase.*

### Phase 1: Basic Chat API
- *Coming soon...*

### Phase 2: Prompt Engineering & Memory
- *Coming soon...*

### Phase 3: Document Ingestion & Chunking
- *Coming soon...*

### Phase 4: Embeddings & Vector Database
- *Coming soon...*

### Phase 5: RAG Pipeline
- *Coming soon...*

### Phase 6: Agents & Tools
- *Coming soon...*

### Phase 7: Advanced Retrieval & Evaluation
- *Coming soon...*

### Phase 8: Custom Models & Production
- *Coming soon...*

---

## 🧪 Running Tests

```bash
# Unit tests
pytest tests/unit

# Integration tests
pytest tests/integration

# Evaluation tests
pytest tests/evaluation

# All tests with coverage
pytest --cov=src tests/
```
---

<p align="center">
  Built with ❤️ as a comprehensive journey through AI Engineering
</p>
