# Cricket Log Semantic Search — Project Inspection and Understanding Report

**Phase 0 | Date: 2026-11-04 | Status: Inspection complete — awaiting implementation approval**

---

## 1. Workspace Inventory

### 1.1 Top-Level Files and Folders

| Relative Path | Type | Purpose | Status |
|---|---|---|---|
| `cricket_log.txt` | Plain text (.txt) | Primary dataset — 500 cricket match event records | Existing, complete |
| `.env.example` | ENV template | Configuration variable template for Gemini API key and model | Existing, complete |
| `Cricket Business Case - LLM.docx` | Word document | Original assignment / business case brief | Existing, complete |
| `docs/` | Directory | Approved project specification documents | Existing |
| `docs/01_Cricket_Log_System_Architecture.docx` | Word document | Approved system architecture specification | Existing, complete |
| `docs/02_Cricket_Log_Technology_Stack.docx` | Word document | Approved technology stack specification | Existing, complete |
| `docs/03_Cricket_Log_Pipeline_and_Inference_Flow.docx` | Word document | Approved pipeline and inference flow specification | Existing, complete |

### 1.2 Missing Items (Nothing Implemented Yet)

| Expected Item | Status |
|---|---|
| Python source files (any) | **Missing — none exist** |
| Django project / apps | **Missing** |
| Streamlit app file | **Missing** |
| Ingestion / parsing modules | **Missing** |
| MongoDB connection code | **Missing** |
| Qdrant client / indexing code | **Missing** |
| Embedding generation module | **Missing** |
| RAG / LLM integration | **Missing** |
| MCP server implementation | **Missing** |
| `requirements.txt` / `pyproject.toml` | **Missing** |
| `Dockerfile` / `Containerfile` / `compose.yaml` | **Missing** |
| Tests and test reports | **Missing** |
| `README.md` | **Missing** |
| `.env` (actual, not example) | **Missing** (correct — should not be committed) |
| `pytest.ini` / `setup.cfg` | **Missing** |

> **Assessment: The workspace is a clean slate.** Only the dataset, documentation, and `.env.example` are present. No prior implementation exists to audit, reuse, or correct.

---

## 2. Dataset Inspection Findings

### 2.1 File Metadata

| Property | Value |
|---|---|
| File path | `cricket_log.txt` |
| File size | **100,794 bytes (~98.4 KB)** |
| SHA-256 checksum | `7AF2B2A6A56532892D09FBA916168125C2FD77B70BFD46A96DA63BEF781E21F8` |
| Total lines | **500** |
| Empty lines | **0** |
| Non-empty records | **500** |
| Date range of records | 2025-09-21 to 2025-10-05 |

### 2.2 Record Structure

Every record follows a **single consistent line format**:

```
[TIMESTAMP] Match:MATCH_ID Team:TEAM Player:PLAYER Event:EVENT Runs:N Over:N Status:STATUS Commentary:FREE_TEXT
```

**All 500 records match this pattern exactly.** No malformed, truncated, or structurally deviant lines were found.

### 2.3 Field Analysis

| Field | Format | Values / Range | Notes |
|---|---|---|---|
| `timestamp` | `[YYYY-MM-DD HH:MM:SS]` | 2025-09-21 to 2025-10-05 | Always present, bracketed |
| `match_id` | `Match:MATCHNNN` | MATCH001 – MATCH010 | 10 unique matches |
| `team` | `Team:Team X` | Team A, Team B, Team C, Team D | 4 unique teams |
| `player` | `Player:Player N` | Player 1 – Player 11 | 11 unique player identifiers |
| `event` | `Event:EVENT_TEXT` | 10 distinct event types | Multi-word values (e.g. `LBW given`, `wicket taken`) |
| `runs` | `Runs:N` | 0–6 (integer) | Always numeric, always present |
| `over` | `Over:N` | 1–50 (integer) | Always numeric, always present |
| `status` | `Status:STATUS_TEXT` | 4 values | May contain spaces (`pending review`) |
| `commentary` | `Commentary:FREE_TEXT` | Variable length free text | Always last field |

### 2.4 Event Types (10 distinct)

| Event | Count |
|---|---|
| LBW given | 63 |
| run scored | 57 |
| wicket taken | 54 |
| stumping | 53 |
| over completed | 49 |
| boundary hit | 49 |
| no ball | 48 |
| six hit | 48 |
| catch taken | 40 |
| wide | 39 |

### 2.5 Status Values (4 distinct)

| Status | Count |
|---|---|
| pending review | 137 |
| disputed | 127 |
| confirmed | 123 |
| verified | 113 |

### 2.6 Match Distribution

Each of the 10 matches (MATCH001–MATCH010) has **exactly 50 records**. Distribution is perfectly uniform.

### 2.7 Commentary Patterns (6 identified templates)

| Pattern | Count | Notes |
|---|---|---|
| `Multiple events observed: Player X performed Y, while another Z occurred in over N.` | 100 | Most common; references two events |
| `Unexpected Y by Player X (ROLE) altered the flow of the game.` | 83 | Role may be present or empty |
| `Player X engaged in Y and scored N runs; match status: STATUS.` | 75 | Repeats structured fields |
| `Spectators reacted as Player X executed Y, contributing N runs.` | 74 | Readable narrative |
| `Player X (ROLE) performed Y, adding N runs in over N.` | 43 | Role may be empty `()` |
| `In over N at VENUE (CONDITION), Player X was involved in Y.` | 46 | Venue and condition optional |

### 2.8 Optional / Implicit Sub-fields in Commentary

Commentary text embeds additional **optional** information not present as explicit top-level fields:

| Sub-field | Source | Values |
|---|---|---|
| `player_role` | Inside `()` in commentary | `batsman`, `bowler`, `all-rounder`, `fielder`, or **empty `()`** (133 occurrences) |
| `venue` | In commentary `at Stadium N` | Stadium 1–4, appears in ~46 records; absent in 454 |
| `weather_condition` | In commentary `(condition)` | `windy` (15), `overcast` (10), `humid` (7), `rainy` (6), `sunny` (5); absent in majority |

> **Important:** `player_role`, `venue`, and `weather_condition` are **optional and inconsistently present**. They must be extracted with nullable/optional handling and must not be assumed present for every record.

### 2.9 Parsing Edge Cases

| Edge Case | Detail | Handling Required |
|---|---|---|
| Multi-word `Event` values | `LBW given`, `wicket taken`, `run scored`, `over completed`, `boundary hit`, `six hit`, `catch taken`, `no ball` | Cannot split on whitespace; must parse up to `Runs:` token |
| Multi-word `Status` value | `pending review` | Cannot split on whitespace; must parse up to `Commentary:` token |
| Empty role in commentary | `Player X ()` — 133 occurrences | Must yield `null`/`None`, not empty string |
| `Commentary:` appears last | Commentary text may contain colons, commas, parentheses | Must extract as the remainder after `Commentary:` prefix |
| `at ()` in commentary | No venue and no condition | Yields `null` for both venue and weather |
| `at Stadium N ()` | Venue present, condition absent | Venue extracted, condition `null` |
| `at (overcast)` | Condition present, no explicit venue | Condition extracted, venue `null` |

### 2.10 Duplicate Records

**Zero duplicate lines.** All 500 records are unique.

### 2.11 Malformed Records

**Zero malformed records.** All 500 lines conform to the expected pattern.

### 2.12 Aggregate Statistics

- **Total runs** across all 500 records: **1,422**
- **Total matches**: 10 (MATCH001–MATCH010)
- **Total teams**: 4 (Team A–D)
- **Total player identifiers**: 11 (Player 1–11)

---

## 3. Existing Implementation Assessment

| Component | Status | Notes |
|---|---|---|
| Dataset loader | Not implemented | |
| Deterministic parser | Not implemented | |
| Cleaning and validation module | Not implemented | |
| MongoDB ingestion | Not implemented | |
| Qdrant indexing | Not implemented | |
| Embedding generation | Not implemented | |
| Django project | Not implemented | |
| Django REST API views | Not implemented | |
| MCP server | Not implemented | |
| Streamlit frontend | Not implemented | |
| Podman / container config | Not implemented | |
| Tests | Not implemented | |
| README | Not implemented | |

**Conclusion:** This is a greenfield implementation. There is no existing code to audit, correct, or reuse.

---

## 4. Architecture Understanding

### 4.1 What Problem Are We Solving?

Cricket match event logs are stored as unstructured plain text. They cannot be queried by meaning ("find moments of dramatic wicket events in disputed matches") using standard database queries, and they are too voluminous for a user to read manually. We need to:

1. **Structure** the unstructured text into validated records.
2. **Store** those records in a database optimized for structured querying.
3. **Index** semantic representations of each record so natural-language questions can retrieve relevant events.
4. **Answer** questions in natural language, grounded exclusively in the actual dataset.

### 4.2 What Is the Expected Final Application?

A web-accessible chatbot and search interface where a user can:
- Ask natural-language questions: *"Which players scored boundaries in disputed matches?"*
- Ask exact structured questions: *"How many runs did Team A score in MATCH003?"*
- Receive answers with cited evidence (event IDs) and disclosed limitations.
- Interact via a Streamlit frontend or via MCP-compatible tools.

### 4.3 How Will the TXT Dataset Be Loaded?

A **LangChain `TextLoader`** (or equivalent simple file reader) opens `cricket_log.txt` and reads the full text. Each physical line is one log record. The loader's only job is to deliver the raw text — it does not understand cricket fields.

### 4.4 What Is the Responsibility of the Loader?

The loader:
- Opens the file with the correct encoding (UTF-8).
- Reads all 500 lines.
- Passes the raw text lines to the parser.

The loader does **not** parse fields, validate records, or apply any cricket-domain knowledge.

### 4.5 Why Do We Need a Deterministic Parser After Loading?

The loader gives us raw strings. A deterministic parser (using Python `re` regex) extracts the structured fields from the fixed-format log line. This is deterministic — the same input always produces the same parsed output — which is essential for reproducibility, debugging, and testing. LangChain's AI extractors are explicitly not used here because the format is fully regular and does not require AI interpretation.

### 4.6 How Will Validation and Cleaning Work?

After parsing, each record is validated:
- `timestamp`: valid ISO-8601 datetime.
- `match_id`: matches pattern `MATCH\d{3}`.
- `runs`: integer, 0–6.
- `over`: integer, 1–50.
- `status`: one of the four known values.
- `commentary`: non-empty string, preserved verbatim.
- `player_role`, `venue`, `weather`: extracted opportunistically from commentary; stored as `null` when absent.

Records failing validation are logged and excluded; they are not silently dropped.

### 4.7 What Data Will Be Stored in MongoDB?

Each validated record becomes one MongoDB document:

```json
{
  "event_id":    "EVT_0001",
  "match_id":    "MATCH001",
  "timestamp":   "2025-09-22T00:23:24",
  "team":        "Team B",
  "player":      "Player 9",
  "event":       "LBW given",
  "runs":        5,
  "over":        37,
  "status":      "pending review",
  "commentary":  "Player 9 () performed LBW given, adding 5 runs in over 37.",
  "player_role": null,
  "venue":       null,
  "weather":     null
}
```

MongoDB is the **authoritative structured store**. All exact calculations (totals, counts, filters) are performed against MongoDB, not against vector search results.

### 4.8 Which Data Will Be Converted Into Embeddings?

A **text representation** is built from each validated record using selected fields:

```
Match: MATCH001 | Team: Team B | Player: Player 9 | Event: LBW given | Runs: 5 | Over: 37 | Status: pending review | Commentary: Player 9 () performed LBW given, adding 5 runs in over 37.
```

This text representation combines structured fields **and** the free-text commentary into a single string that the embedding model can encode.

### 4.9 What Will Be Stored in Qdrant?

One Qdrant **point** per record:
- **Vector**: dense embedding (e.g., 384 dimensions from `all-MiniLM-L6-v2`).
- **Payload**: `event_id`, `match_id`, `team`, `player`, `event`, `status` — structured metadata for filtering.
- The full record is **not** stored in Qdrant; only what is needed for search and filtering.

The `event_id` payload field is the bridge back to MongoDB.

### 4.10 How Will Semantic Search Work?

1. User question is embedded by the same model used at ingestion time.
2. Qdrant performs approximate nearest-neighbour search (cosine similarity) to return the top-K most similar points.
3. The returned `event_id` values are used to fetch the full authoritative records from MongoDB.
4. Hydrated records are assembled as evidence.

### 4.11 How Will Exact Structured Questions Be Answered?

For questions like *"How many runs did Team A score in MATCH001?"*, no vector search is used. The Django backend:
1. Identifies the question as a structured/aggregation query.
2. Runs a MongoDB aggregation query (`$match` + `$group` + `$sum`).
3. Returns the calculated result directly.
4. May optionally send a brief evidence summary to Gemini for natural-language formatting — but the **number comes from MongoDB, not from an LLM**.

### 4.12 What Role Will Gemini Play?

Gemini (`gemini-2.5-flash`, via the `google-genai` SDK) is used **only for natural-language synthesis**. It:
- Receives: the user's question + a bounded set of hydrated evidence records + a grounding instruction.
- Returns: a fluent answer grounded exclusively in the provided evidence.
- Does **not** receive the full dataset, all MongoDB records, or complete embeddings.
- Does **not** perform arithmetic — exact calculations are done by MongoDB/Python before Gemini is called.
- Does **not** invent player roles, venues, weather, or scores absent from the evidence.

### 4.13 Why Are Django REST Framework and MCP Needed?

**Django REST Framework (DRF):**
- Acts as the backend coordinator for all pipeline logic.
- Exposes REST endpoints for: semantic search, exact queries, and RAG chat.
- Coordinates MongoDB lookup, Qdrant retrieval, evidence preparation, and Gemini calls.
- Streamlit and MCP both talk to DRF — neither talks directly to databases.

**MCP Server:**
- A lightweight wrapper that exposes selected DRF API endpoints as **MCP tools**.
- Allows MCP-compatible clients (e.g., AI assistants, IDE agents) to call the same search and chat capabilities programmatically.
- Contains **no business logic** — it calls DRF APIs and passes results through.

### 4.14 What Role Does Streamlit Play?

Streamlit is the **user-facing frontend**. It:
- Renders a chat interface and search results.
- Calls DRF REST APIs via HTTP (never directly to MongoDB or Qdrant).
- Displays: answer text, cited event IDs, evidence scope, and limitations.

### 4.15 How Will Podman Be Used?

Podman containerizes each service:
- **MongoDB container**: database service with a named volume.
- **Qdrant container**: vector store service with a named volume.
- **Django/ingestion container**: backend API and offline ingestion scripts.
- **Streamlit container**: frontend.
- **MCP container**: MCP server (co-located with Django or as a fifth lightweight service).

Services are connected via a shared Podman network. A `compose.yaml` manages startup order and volume mounts.

---

## 5. End-to-End Pipeline Example

### Input (raw log line):
```
[2025-09-22 00:23:24] Match:MATCH001 Team:Team B Player:Player 9 Event:LBW given Runs:5 Over:37 Status:pending review Commentary:Player 9 () performed LBW given, adding 5 runs in over 37.
```

### Step 1 — Loader output:
Raw string passed to parser unchanged.

### Step 2 — Parser output:
```python
{
  "timestamp": "2025-09-22 00:23:24",
  "match_id":  "MATCH001",
  "team":      "Team B",
  "player":    "Player 9",
  "event":     "LBW given",
  "runs":      5,
  "over":      37,
  "status":    "pending review",
  "commentary": "Player 9 () performed LBW given, adding 5 runs in over 37."
}
```

### Step 3 — Cleaner/Validator output:
All fields pass. Optional sub-fields: `player_role=None`, `venue=None`, `weather=None` (empty role `()` detected).

### Step 4 — MongoDB document stored:
```json
{
  "_id":        "<ObjectId>",
  "event_id":   "EVT_0001",
  "match_id":   "MATCH001",
  "timestamp":  "2025-09-22T00:23:24",
  "team":       "Team B",
  "player":     "Player 9",
  "event":      "LBW given",
  "runs":       5,
  "over":       37,
  "status":     "pending review",
  "commentary": "Player 9 () performed LBW given, adding 5 runs in over 37.",
  "player_role": null,
  "venue":       null,
  "weather":     null
}
```

### Step 5 — Text representation for embedding:
```
Match: MATCH001 | Team: Team B | Player: Player 9 | Event: LBW given | Runs: 5 | Over: 37 | Status: pending review | Commentary: Player 9 () performed LBW given, adding 5 runs in over 37.
```

### Step 6 — Qdrant point stored:
```json
{
  "id":      1,
  "vector":  [0.023, -0.117, 0.344, "...384 dims..."],
  "payload": {
    "event_id":  "EVT_0001",
    "match_id":  "MATCH001",
    "team":      "Team B",
    "player":    "Player 9",
    "event":     "LBW given",
    "status":    "pending review"
  }
}
```

### Step 7 — Inference:
**User query:** *"Show me LBW decisions that are still under review"*

- Query embedded → Qdrant cosine search → returns `EVT_0001` and similar records.
- MongoDB lookup using `event_id` values → full authoritative records returned.
- Evidence assembled.
- Gemini receives question + evidence → generates grounded answer.

### Step 8 — API response:
```json
{
  "answer": "There are records of LBW given events with 'pending review' status. For example, in Match MATCH001, Player 9 was involved in an LBW decision in over 37, contributing 5 runs. The status remains pending review.",
  "evidence": ["EVT_0001", "EVT_0057", "EVT_0142"],
  "scope": "Top-5 semantically similar records retrieved; not guaranteed to be exhaustive.",
  "limitation": "Semantic search retrieves candidate records; for a complete count of all pending LBW decisions, use the structured query endpoint."
}
```

---

## 6. Technology Stack Assessment

### 6.1 Confirmed Stack (from approved documents)

| Layer | Technology | Status |
|---|---|---|
| Language | Python 3.10 | Confirmed (installed) |
| Backend framework | Django REST Framework | Confirmed |
| Structured database | MongoDB + PyMongo | Confirmed |
| Vector database | Qdrant | Confirmed |
| LLM | Gemini 2.5 Flash via `google-genai` | Confirmed — API key in `.env.example` |
| MCP | Python MCP SDK | Confirmed |
| Frontend | Streamlit | Confirmed |
| Containerization | Podman | Confirmed |
| Testing | pytest | Confirmed |
| Version control | Git / GitHub | Confirmed |

### 6.2 Libraries Requiring Validation

| Library | Purpose | Validation Needed |
|---|---|---|
| `langchain` / `langchain-community` | File loader | Whether `TextLoader` is explicitly required or `open()` suffices |
| `sentence-transformers` | Dense embedding model | Benchmark on representative questions |
| `qdrant-client` | Qdrant access | Native vs LangChain integration decision |
| `langchain-qdrant` | LangChain Qdrant wrapper | Compare with native client |

### 6.3 Explicitly Not Needed

| Technology | Reason to Exclude |
|---|---|
| LangGraph | No graph orchestration, state machine, or conditional paths required |
| CrewAI / multi-agent frameworks | Single-agent pipeline; not in requirements |
| Redis | No measured caching need identified |
| Second vector database | Qdrant alone is sufficient |

---

## 7. Gaps and Risks

### 7.1 Gaps (Items That Must Be Built)

| Gap | Priority |
|---|---|
| Complete Python source implementation | Critical |
| `requirements.txt` with pinned versions | Critical |
| `README.md` with setup and run instructions | Critical |
| Podman `compose.yaml` / `Containerfile` | Critical |
| pytest suite (unit + integration) | Critical |
| `.env` (populated, not committed) | Required before running |

### 7.2 Technical Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Commentary regex too greedy | Low — format is consistent | Test parser against all 500 lines |
| Optional fields parsed incorrectly | Medium — multiple template patterns | Unit test each commentary template |
| Qdrant `event_id` mapping drift | Low | Verify count + spot-check mapping after ingestion |
| Gemini rate limits | Medium | Implement retry with backoff; cache in dev |
| Podman networking on Windows host | Medium | Test on target environment; document workarounds |
| `all-MiniLM-L6-v2` retrieval quality | Medium | Benchmark with 10+ representative questions |

### 7.3 Dataset-Specific Observations

- Records are **not sorted** by timestamp — timestamps within MATCH001 span multiple dates out of order.
- All timestamps end in `:24` seconds — characteristic of synthetic data generation. No functional impact.
- The `Multiple events observed` commentary template (100 records) references a second event in narrative text. That second event is **not a separate record** — do not attempt to parse it as one.
- Dataset is fully synthetic: no real player names, real teams, or real match IDs.

---

## 8. Recommended Implementation Sequence

| Module | Objective | Depends on |
|---|---|---|
| **1** — Environment & Scaffold | Project structure, dependencies, Django skeleton | Nothing |
| **2** — Loader, Parser, Cleaner, Validator | 500 tested structured records from TXT | Module 1 |
| **3** — MongoDB Ingestion | 500 documents in MongoDB, indexed | Modules 1, 2 |
| **4** — Embedding & Qdrant Indexing | 500 Qdrant points, event_id mapped | Modules 1, 2, 3 |
| **5** — Retrieval Evaluation | Validate dense retrieval quality | Module 4 |
| **6** — RAG Pipeline & Gemini | Grounded answers from evidence | Modules 4, 5 |
| **7** — Django REST API | REST endpoints: semantic, exact, chat, health | Modules 1–6 |
| **8** — MCP Server | MCP tools calling DRF endpoints | Module 7 |
| **9** — Streamlit Frontend | User-facing chat and search UI | Module 7 |
| **10** — Podman Containerization | All services in containers | Modules 1–9 |
| **11** — E2E Testing & GitHub Delivery | Final integration tests, README, delivery | Modules 1–10 |

---

## 9. Decisions Requiring Confirmation

### Decision 1 — LangChain Loader vs. Plain Python `open()`

**Question:** Is demonstrating `TextLoader` specifically required, or is any loader that reads the file correctly acceptable?

| Option | Pros | Cons |
|---|---|---|
| Plain `open()` **(Recommended)** | Zero extra dependency; full control; cleaner | No LangChain `Document` wrapper |
| `TextLoader` (LangChain) | Satisfies explicit LangChain grading criteria | Adds dependency; does nothing `open()` cannot do |

**→ Please confirm: is `TextLoader` specifically required by the assignment?**

---

### Decision 2 — Whether LangGraph Is Needed

**Assessment:** Pipeline is linear. No conditional branching, retries, or state machines identified.

**Recommendation:** **Do not use LangGraph.**

**→ Please confirm if LangGraph must be demonstrated.**

---

### Decision 3 — Native Qdrant Client vs. LangChain Qdrant Integration

| Option | Pros | Cons |
|---|---|---|
| Native `qdrant-client` **(Recommended)** | Full payload control; easier event_id mapping verification | Slightly more verbose |
| `langchain-qdrant` | LangChain retriever integration | Abstracts payload control |

**→ Please confirm: is `langchain-qdrant` specifically required?**

---

### Decisions Already Resolved (No Confirmation Needed)

| Decision | Resolution |
|---|---|
| Parser strategy | Single compiled regex with named groups + secondary optional extraction from commentary |
| Record vs. chunk granularity | One vector per record |
| Dense vs. hybrid retrieval | Dense first; hybrid only if Module 5 evaluation reveals gaps |
| MongoDB access | PyMongo directly (no ORM) |
| Gemini integration | Official `google-genai` SDK, model configurable via env var |
| MCP-to-DRF communication | `httpx` HTTP client calls DRF endpoints |
| Podman architecture | `compose.yaml` with named services and network |

---

## 10. Proposed First Coding Task

**Task: Module 1 + Module 2 — Environment Scaffold + Full Parser with Tests**

**Deliverables:**

1. Project directory structure created.
2. `requirements.txt` with pinned versions.
3. `ingestion/loader.py` — reads `cricket_log.txt`, returns list of raw line strings.
4. `ingestion/parser.py` — regex parser; returns list of raw parsed dicts.
5. `ingestion/cleaner.py` — type coercion, optional field extraction, `event_id` assignment.
6. `ingestion/validator.py` — field validation; logs/collects invalid records.
7. `tests/test_parser.py` — unit tests covering all 6 commentary templates, all edge cases, all 500 records.
8. `pytest.ini` — test discovery configuration.
9. Django project scaffold (`manage.py`, `settings.py`, app skeletons) — importable but not yet serving.
10. `README.md` skeleton — purpose, tech stack, setup instructions.

**Success condition:** `pytest tests/test_parser.py` passes with 500 records validated and 0 failures.

---

*Report generated: 2026-11-04 | Dataset SHA-256: `7AF2B2A6A56532892D09FBA916168125C2FD77B70BFD46A96DA63BEF781E21F8`*
