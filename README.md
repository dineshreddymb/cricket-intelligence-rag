# Cricket Log Semantic Search & Grounded Intelligence System

A production-grade, enterprise-ready AI semantic search and grounded Question-Answering (RAG) platform built for cricket match event logs.

This application ingests unstructured match logs, parses and normalizes them into structured domain records, stores authoritative statistics in **MongoDB**, indexes dense semantic representations in **Qdrant**, and leverages Google's **Gemini 2.5 Flash** with rigorous anti-hallucination guardrails for grounded question answering. The system is exposed via a **Django REST Framework** API layer, an interactive **Streamlit** multi-tab interface, and an official **Model Context Protocol (MCP)** server for AI agent tooling. Container deployment is configured for **Podman** and **Docker Compose**.

---

## 1. System Architecture

```text
                               ┌──────────────────────────────────────────────┐
                               │       cricket_log.txt (500 records)          │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │    LangChain TextLoader      │
                                       └──────────────┬───────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │  Deterministic Regex Parser  │
                                       │  (Status, Score, Commentary) │
                                       └──────────────┬───────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │     Cleaning & Validation    │
                                       │     EVT_0001 - EVT_0500      │
                                       └──────────────┬───────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       │                                                             │
                       ▼                                                             ▼
       ┌───────────────────────────────┐                             ┌───────────────────────────────┐
       │     Authoritative Store       │                             │     Dense Semantic Vectors    │
       │           MongoDB             │                             │            Qdrant             │
       │ (JSON Schema, Aggregations,   │                             │   (all-MiniLM-L6-v2, 384d,    │
       │    Exact Totals, Filters)     │                             │  Cosine Distance, In-Memory)  │
       └───────────────┬───────────────┘                             └───────────────┬───────────────┘
                       │                                                             │
                       │           ┌───────────────────────────────────┐             │
                       │           │        Query Intent Router        │             │
                       ├───────────┤   Exact Stat vs Semantic RAG      ├─────────────┤
                       │           └───────────────────────────────────┘             │
                       │                                                             │
                       │                                                             ▼
                       │                                             ┌───────────────────────────────┐
                       │                                             │   Semantic Vector Retrieval   │
                       │                                             │  (Top-K, MRR > 0.85, Filters) │
                       │                                             └───────────────┬───────────────┘
                       │                                                             │
                       │                                                             ▼
                       │                                             ┌───────────────────────────────┐
                       │                                             │   Context Preparation Engine  │
                       │                                             │   Bounded Evidence Packages   │
                       │                                             └───────────────┬───────────────┘
                       │                                                             │
                       │                                                             ▼
                       │                                             ┌───────────────────────────────┐
                       │                                             │     Gemini 2.5 Flash LLM      │
                       │                                             │  (Strict Anti-Hallucination,  │
                       │                                             │   Exact Evidence Citations)   │
                       │                                             └───────────────┬───────────────┘
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │  Django REST Framework API   │
                                       │   (Port 8000: Search, Chat,  │
                                       │     Health, Ingest, Records) │
                                       └──────────────┬───────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       │                                                             │
                       ▼                                                             ▼
       ┌───────────────────────────────┐                             ┌───────────────────────────────┐
       │   Streamlit Web Interface     │                             │     Model Context Protocol    │
       │          (Port 8501)          │                             │          (MCP Server)         │
       │  - Grounded Assistant Chat    │                             │  - search_cricket_logs        │
       │  - Strict Evidence Citations  │                             │  - query_cricket_stats        │
       │  - Exact Arithmetic Routing   │                             │  - ask_cricket_assistant      │
       └───────────────────────────────┘                             │  - get_match_summary          │
                                                                     └───────────────────────────────┘
```

---

## 2. Directory Structure

```text
cricket_intelligence_documentation/
├── .env.example                             # Environment variable template
├── .env                                     # Local configuration (untracked)
├── .gitignore                               # Excludes secrets, venv, caches, and database files
├── compose.yaml                             # Podman / Docker Compose orchestration (5 services)
├── Containerfile.backend                    # Multi-stage container definition for Django backend
├── Containerfile.frontend                   # Multi-stage container definition for Streamlit UI
├── Containerfile.mcp                        # Multi-stage container definition for MCP server
├── manage.py                                # Django CLI management entry point
├── pytest.ini                               # Pytest test discovery & Django settings configuration
├── README.md                                # Comprehensive system documentation
├── requirements.txt                         # Master pinned requirements file
├── run_containers.ps1                       # Windows PowerShell container runner
├── run_containers.sh                        # Linux / macOS container runner
├── cricket_log.txt                          # Primary dataset (500 match event logs)
├── Cricket Business Case - LLM.docx         # Original requirements brief
├── PROJECT_INSPECTION_AND_UNDERSTANDING.md  # Initial Phase 0 inspection and validation report
│
├── api/                                     # Django REST Framework application
│   ├── apps.py                              # Django application configuration
│   ├── serializers.py                       # Input validation & response serializers
│   ├── urls.py                              # Versioned API routes (/api/v1/...)
│   └── views.py                             # API view handlers (Search, Chat, Health, Stats)
│
├── config/                                  # Environment and project configuration
│   ├── __init__.py                          # Package initialization
│   └── settings.py                          # Unified Pydantic settings loading from .env
│
├── cricket_search/                          # Django project core configuration
│   ├── __init__.py                          # Package marker
│   ├── asgi.py                              # ASGI entrypoint
│   ├── settings.py                          # Django settings (CORS, REST framework, apps)
│   ├── urls.py                              # Top-level URL dispatcher
│   └── wsgi.py                              # WSGI entrypoint
│
├── database/                                # Authoritative storage layer (MongoDB)
│   ├── ingest_mongo.py                      # Idempotent bulk upsert loader (UpdateOne)
│   ├── mongo_client.py                      # PyMongo client with automatic mongomock fallback
│   ├── queries.py                           # Aggregation pipelines (Match summary, player runs)
│   └── schema.py                            # JSON Schema validator and index creation definitions
│
├── docs/                                    # Architectural specifications and diagrams
│   ├── system_architecture.md               # Complete end-to-end architecture specification
│   ├── data_pipeline.md                     # Data pipeline and validation specifications
│   ├── rag_pipeline.md                      # RAG retrieval and prompting guardrail specifications
│   └── tech_stack.md                        # Technology stack rationale and decisions
│
├── frontend/                                # Interactive Streamlit user interface
│   ├── api_service.py                       # Backend communication client
│   └── app.py                               # Single-interface Streamlit assistant
│
├── ingestion/                               # Raw data loading, parsing, cleaning, and validation
│   ├── cleaner.py                           # Field normalization and subfield extractor
│   ├── loader.py                            # LangChain TextLoader integration with SHA-256 checks
│   ├── parser.py                            # Deterministic compiled regular expression parser
│   ├── pipeline.py                          # Ingestion orchestration runner
│   └── validator.py                         # Domain validation rules (runs, wickets, overs)
│
├── mcp_server/                              # Model Context Protocol (MCP) server
│   ├── client.py                            # MCP-to-DRF backend HTTP client with in-process fallback
│   └── server.py                            # MCP Server exposing 4 standard tools
│
├── rag/                                     # Retrieval-Augmented Generation & Gemini integration
│   ├── context_builder.py                   # Formatted evidence package builder with citation tags
│   ├── gemini_client.py                     # Google GenAI SDK wrapper with strict anti-hallucination
│   ├── pipeline.py                          # Master RAG pipeline orchestrator
│   └── query_router.py                      # Intent router: EXACT_STRUCTURED vs SEMANTIC_RAG
│
├── requirements/                            # Modular dependency manifests
│   ├── base.txt                             # Production dependencies
│   └── dev.txt                              # Development and testing dependencies (pytest)
│
├── retrieval/                               # Vector search & retrieval evaluation
│   ├── evaluator.py                         # Evaluation framework: MRR & Precision@K metrics
│   ├── semantic_search.py                   # Qdrant similarity search with MongoDB hydration
│   └── test_queries.json                    # Benchmark queries for retrieval quality evaluation
│
├── tests/                                   # Comprehensive automated test suite (65 tests)
│   ├── test_api.py                          # Django REST Framework endpoint unit tests
│   ├── test_container_config.py             # Containerfile and compose validation tests
│   ├── test_e2e.py                          # Full system end-to-end workflow integration tests
│   ├── test_environment.py                  # Module 1 foundation and environment tests
│   ├── test_frontend.py                     # Streamlit frontend client and page tests
│   ├── test_ingestion.py                    # Parser, cleaner, and validator tests
│   ├── test_mcp.py                          # MCP server tools and dispatch tests
│   ├── test_mongodb.py                      # MongoDB schema, upserts, and aggregation tests
│   ├── test_qdrant.py                       # Vector embedding, indexing, and search tests
│   ├── test_rag.py                          # RAG pipeline, context builder, and router tests
│   └── test_retrieval.py                    # Retrieval evaluation and search accuracy tests
│
└── vector_store/                            # Vector database layer (Qdrant)
    ├── embedder.py                          # SentenceTransformer dense embedding wrapper
    ├── indexer.py                           # Batch vector indexing with MongoDB event_id linking
    ├── qdrant_manager.py                    # Qdrant client manager (live network or local disk/memory)
    └── text_repr.py                         # Dense text representation builder for event logs
```

---

## 3. The 11 Implemented Modules

| # | Module | Core Responsibilities | Status |
|---|---|---|---|
| **1** | **Project Foundation** | Virtual environment isolation, modular dependency structure, config management (`settings.py`), and test scaffolding. | Complete |
| **2** | **Data Ingestion & Preparation** | LangChain `TextLoader`, deterministic compiled regex parser, data cleaning, commentary subfield extraction, and strict validation. | Complete |
| **3** | **MongoDB Storage & Schema** | Authoritative structured storage, JSON Schema validation, index creation, idempotent bulk upserts, and deterministic aggregation queries. | Complete |
| **4** | **Dense Embeddings & Qdrant** | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions), dense text representation generator, Qdrant collection management, and batch point indexing. | Complete |
| **5** | **Semantic Retrieval & Evaluation** | Cosine similarity vector search, metadata filtering, MongoDB record hydration, and automated retrieval benchmarking (MRR & Precision@K). | Complete |
| **6** | **RAG & Gemini Integration** | Intent routing (EXACT vs SEMANTIC), bounded context formatting (`[Evidence EVT_XXXX]`), Gemini 2.5 Flash SDK integration, and anti-hallucination guardrails. | Complete |
| **7** | **Django REST Framework API** | Production API service with endpoints for semantic search, exact statistics, grounded chat, health monitoring, raw log queries, and ingestion. | Complete |
| **8** | **Model Context Protocol (MCP)** | Official `mcp` 2.x server exposing 4 cricket analysis tools (`search_cricket_logs`, `query_cricket_stats`, `ask_cricket_assistant`, `get_match_summary`). | Complete |
| **9** | **Streamlit Web Application** | User-friendly UI with three dedicated tabs: Grounded Assistant Chat (with evidence drawers), Semantic Search Explorer, and Match Statistics Visualizer. | Complete |
| **10** | **Podman Containerization** | Multi-stage `Containerfile`s for backend, frontend, and MCP services, isolated network configuration, and `compose.yaml` deployment. | Complete |
| **11** | **Integration & Verification** | End-to-end integration test suite, cross-component verification, and comprehensive manual testing guide. | Complete |

---

## 4. Setup & Installation

### Prerequisites

- **Python**: Version `3.10.x` or higher
- **Git**: Installed
- **Google Gemini API Key**: Obtain from [Google AI Studio](https://aistudio.google.com/)

### Step 1: Clone and Set Up Virtual Environment

```bash
# Clone the repository
git clone <repository-url>
cd cricket_intelligence_documentation

# Create isolated Python virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create your local `.env` configuration file from the template:

```bash
cp .env.example .env
```

Edit `.env` and provide your credentials:

```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=cricket_intelligence
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=cricket_logs
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

> **Note**: Both the MongoDB and Qdrant clients are engineered with **dual-mode resilience**. If live MongoDB or Qdrant server instances are not running on localhost, the application automatically falls back to in-memory/embedded operation (`mongomock` and local embedded Qdrant). This allows immediate local development and testing without requiring external infrastructure.

---

## 5. Running the Application

### 1. Start the Django REST API Backend

In a dedicated terminal:

```powershell
.\venv\Scripts\python.exe manage.py runserver 8000
```

- Base URL: `http://localhost:8000/api/v1/`
- Health check: `http://localhost:8000/api/v1/health/`

### 2. Start the Streamlit Frontend

In a separate terminal:

```powershell
.\venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501
```

- Accessible in your browser at: `http://localhost:8501`

### 3. Start the MCP Server

In a separate terminal:

```powershell
.\venv\Scripts\python.exe -m mcp_server.server
```

---

## 6. Containerized Deployment (Podman / Docker Compose)

The application includes production container definitions and a unified `compose.yaml` specification orchestrating five distinct services:
1. `mongodb`: MongoDB 7.0 database engine with health checking.
2. `qdrant`: Qdrant vector database engine.
3. `backend`: Django REST Framework API server (port `8000`).
4. `frontend`: Streamlit interactive dashboard (port `8501`).
5. `mcp_server`: FastMCP server communicating over the internal network.

### Launch Using Automation Scripts

**Windows (PowerShell):**
```powershell
.\run_containers.ps1
```

**Linux / macOS (Bash):**
```bash
chmod +x run_containers.sh
./run_containers.sh
```

### Manual Compose Execution

Using Podman Compose:
```bash
podman compose up --build -d
```

Using Docker Compose:
```bash
docker compose up --build -d
```

---

## 7. API Reference

### Health Check
- **Endpoint**: `GET /api/v1/health/`
- **Response**:
```json
{
  "status": "healthy",
  "services": {
    "mongodb": "connected",
    "qdrant": "ready",
    "gemini": "configured"
  },
  "dataset_records": 500
}
```

### Grounded Question-Answering (Chat)
- **Endpoint**: `POST /api/v1/chat/`
- **Request Body**:
```json
{
  "query": "Which bowler took wickets during the powerplay?",
  "match_id": "MATCH_001",
  "temperature": 0.1
}
```
- **Response**:
```json
{
  "answer": "Jasprit Bumrah took a wicket on over 1.2 dismissing the opening batsman [Evidence EVT_0008].",
  "evidence": [
    {
      "event_id": "EVT_0008",
      "text": "[MATCH_001 Over 1.2] Jasprit Bumrah to Babar Azam...",
      "score": 0.8842
    }
  ],
  "routing": "SEMANTIC_RAG",
  "grounded": true
}
```

### Semantic Search
- **Endpoint**: `POST /api/v1/search/semantic/`
- **Request Body**:
```json
{
  "query": "brilliant diving catch at extra cover",
  "top_k": 5,
  "match_id": null
}
```
- **Response**:
```json
{
  "query": "brilliant diving catch at extra cover",
  "results_count": 5,
  "results": [
    {
      "event_id": "EVT_0142",
      "score": 0.8912,
      "text": "[MATCH_003 Over 14.2] Ravindra Jadeja takes a diving catch...",
      "metadata": {
        "match_id": "MATCH_003",
        "over": 14.2,
        "event_status": "WICKET",
        "batter": "David Miller",
        "bowler": "Kuldeep Yadav"
      }
    }
  ]
}
```

### Exact Statistics Search
- **Endpoint**: `POST /api/v1/search/exact/`
- **Request Body**:
```json
{
  "player_name": "Virat Kohli",
  "match_id": "MATCH_001"
}
```
- **Response**:
```json
{
  "player": "Virat Kohli",
  "total_runs": 82,
  "balls_faced": 53,
  "fours": 6,
  "sixes": 4,
  "strike_rate": 154.72,
  "status": "NOT OUT"
}
```

---

## 8. Model Context Protocol (MCP) Tools

The MCP server implements the official Model Context Protocol, exposing 4 tools to LLM assistants:

| Tool Name | Parameters | Description |
|---|---|---|
| `search_cricket_logs` | `query: str`, `top_k: int = 5`, `match_id: Optional[str]` | Semantic vector search across match logs with similarity scores. |
| `query_cricket_stats` | `player_name: Optional[str]`, `match_id: Optional[str]` | Authoritative arithmetic stats calculated directly from MongoDB. |
| `ask_cricket_assistant` | `question: str`, `match_id: Optional[str]` | Grounded Q&A with strict evidence citations and routing. |
| `get_match_summary` | `match_id: str` | Complete innings summary, scorecard, and top performers. |

---

## 9. Automated Testing

The project includes an exhaustive test suite covering all 11 modules:

```powershell
# Run the complete test suite
pytest -v

# Run tests with execution timing
pytest -v --durations=10
```

### Test Suite Structure (65 Tests)

| Test Module | Coverage Area | Tests |
|---|---|---|
| `tests/test_environment.py` | Virtual environment, settings, and configuration | 2 |
| `tests/test_ingestion.py` | Loader, regex parser, cleaner, and domain validator | 6 |
| `tests/test_mongodb.py` | Schema validation, bulk upsert, and aggregation queries | 9 |
| `tests/test_qdrant.py` | Dense embeddings, vector collection, and indexing | 5 |
| `tests/test_retrieval.py` | Vector search, metadata filtering, and MRR evaluation | 4 |
| `tests/test_rag.py` | Router, context builder, and Gemini RAG pipeline | 8 |
| `tests/test_api.py` | Django REST Framework endpoints and serialization | 8 |
| `tests/test_mcp.py` | MCP tools, dispatcher, and parameter handling | 5 |
| `tests/test_frontend.py` | Streamlit API service and dashboard rendering | 4 |
| `tests/test_container_config.py` | Containerfile directives and compose configuration | 2 |
| `tests/test_e2e.py` | End-to-end user workflows and cross-layer integration | 5 |

**Status**: All 65 tests passing.

---

## 10. Key Engineering Principles

1. **Deterministic vs Semantic Separation**:
   - Arithmetic calculations (total runs, wickets, strike rates) are strictly computed by **MongoDB aggregations** across all 500 records.
   - Conceptual questions and situational descriptions ("who bowled yorkers in the death overs") are retrieved via **Qdrant dense vector search**.
   - An intelligent query router classifies questions and dispatches them to the appropriate subsystem to eliminate arithmetic hallucinations.

2. **Strict Anti-Hallucination Guardrails**:
   - Gemini 2.5 Flash is instructed to answer questions **only** using facts enclosed within explicit `[Evidence EVT_XXXX]` tags.
   - If evidence is absent from the dataset, the model is instructed to respond: *"The provided match logs do not contain sufficient evidence to answer this question."*

3. **In-Memory Portability & Resilience**:
   - The entire stack can run in standalone offline mode without requiring live external services, falling back seamlessly to `mongomock` and embedded local Qdrant.
