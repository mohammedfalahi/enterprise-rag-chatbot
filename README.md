# Enterprise Knowledge Intelligence Assistant (RAG Chatbot)

An enterprise-grade, multi-tenant Retrieval-Augmented Generation (RAG) platform engineered with strict database-level role-based access control (RBAC), hybrid search with Reciprocal Rank Fusion (RRF), cross-encoder reranking, and grounded token streaming.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16.4.4-black.svg)](https://nextjs.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Ollama](https://img.shields.io/badge/Embeddings-Qwen3--0.6B%20(Local)-orange.svg)](https://ollama.com/)
[![Tests](https://img.shields.io/badge/Tests-166%20Passed-brightgreen.svg)]()
[![Permission Leakage](https://img.shields.io/badge/Permission%20Leakage-0.0%25-success.svg)]()

---

## 1. Problem Statement

Most standard RAG prototypes fail in enterprise environments because they:
1. **Lack Tenant & Role Isolation:** They query a global vector index and hope the LLM prompt respects confidentiality, leading to severe permission leakage.
2. **Hallucinate on Missing Data:** Naive similarity search always returns top-$k$ results even when the corpus has no relevant information, causing the model to invent plausible falsehoods.
3. **Succumb to Prompt Injections:** Untrusted documents containing adversarial overrides hijack LLM context and leak system instructions.
4. **Depend on Slow, Costly External Embedding APIs:** Cloud embedding calls introduce latency, token costs, and privacy exposure.

This platform solves these problems through **defense-in-depth database enforcement**, **local on-premise embeddings**, **hybrid retrieval with cross-encoder refusal gating**, and **server-side identity validation**.

---

## 2. Key Features

- **Database-Level Tenant & Role Isolation:** Dense and sparse retrieval queries enforce `WHERE tenant_id = %s AND access_roles && %s` directly in PostgreSQL.
- **100% Local Embeddings:** Powered by **Qwen3-Embedding-0.6B** (`1024` dimensions) via local Ollama with zero truncation and strict dimension validation.
- **Two-Stage Hybrid Search & Reranking:**
  - Stage 1: Dense pgvector HNSW cosine search + Sparse PostgreSQL Full-Text Search fused via **Reciprocal Rank Fusion (RRF, $k=60$)**.
  - Stage 2: Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`) with a calibrated refusal gate (`min_rerank_score = -8.5`).
- **Grounded Token Streaming:** Server-Sent Events (SSE) streaming with **Google Gemini 3.5 Flash Lite** (with optional failover to Groq `openai/gpt-oss-120b`).
- **Verifiable Citations:** Returns interactive document titles, breadcrumb paths, deterministic chunk IDs, and cross-encoder relevance scores.
- **Multi-Turn Memory & Query Rewriting:** Conversational context window of 9 turns with LLM-powered anaphora resolution for follow-up questions.
- **Fast-Path Intent Routing:** Pattern classifier that instantly answers greetings and chit-chat with zero database or vector overhead.
- **Enterprise Engineering:** In-memory sliding-window rate limiting, asynchronous connection pooling (`psycopg_pool`), and Langfuse OpenTelemetry tracing with automatic PII scrubbing.

---

## 3. Technology Stack

| Layer | Component | Version / Identifier | Purpose |
|---|---|---|---|
| **Frontend** | Next.js / React / Tailwind CSS | `16.4.4` / `19.3.0` / `^4` | Multi-turn chat UI, role dashboard, upload modal |
| **Backend API** | FastAPI / Python | `>=0.162.0` / `3.12.14` | High-performance asynchronous REST & SSE server |
| **Database** | Supabase PostgreSQL + pgvector | `15.8` / pgvector `0.8.0` | Relational store, HNSW vector search, full-text search |
| **Connection Pool** | Psycopg 3 Pool (`psycopg[binary,pool]`) | `>=3.3.7` | Asynchronous connection pooling (min 2, max 10) |
| **Auth & Security** | Supabase Auth / PyJWT | `>=2.8.0` | Asymmetric ES256 JWT signature verification via JWKS |
| **Embedding Engine** | Ollama (`qwen3-embedding:0.6b`) | Ollama v0.5+ | Local 1024-dimensional dense vector embeddings |
| **Reranker** | Sentence-Transformers | `>=6.2.0` | `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker |
| **Generation LLM** | Google GenAI SDK | `>=2.34.0` | Gemini 3.5 Flash Lite streaming answer synthesis |
| **Alternative LLM** | Groq (`openai` SDK) | `>=1.0.0` | Fallback inference: `openai/gpt-oss-120b` |
| **Observability** | Langfuse | `>=5.1.0` | OpenTelemetry-compliant execution tracing & scrubbing |

---

## 4. End-to-End System Architecture

```mermaid
flowchart TD
    User([Corporate User]) --> FE[Next.js 16 Client]
    FE -->|ES256 Bearer JWT| API[FastAPI Backend]

    subgraph Security & Access Control
        API --> RATE[Sliding-Window Rate Limiter]
        RATE --> JWKS[ES256 JWKS Signature Cache]
        JWKS --> PROF[Server-Side Profile: tenant_id & roles]
    end

    subgraph RAG Orchestration
        PROF --> INTENT{Intent Router}
        INTENT -- Greeting --> FAST[Direct Streaming Reply]
        INTENT -- Domain Query --> REWRITE{History > 0?}
        REWRITE -- Yes --> REWRITER[Gemini Query Rewriter]
        REWRITE -- No --> CLEAN[Raw Query]
        REWRITER --> CLEAN
    end

    subgraph Hybrid Retrieval Pipeline
        CLEAN --> OLLAMA[Local Ollama: Qwen3-0.6B Embedder]
        OLLAMA --> DENSE[Dense Vector Search: HNSW Cosine]
        CLEAN --> SPARSE[Sparse Lexical Search: Postgres FTS]
        DENSE --> RRF[Reciprocal Rank Fusion: k=60]
        SPARSE --> RRF
        RRF --> RERANK[Cross-Encoder Reranker: MiniLM-L6]
        RERANK --> GATE{Top Score >= -8.5?}
        GATE -- No --> REFUSE[Safe Refusal: Insufficient Context]
        GATE -- Yes --> CTX[Grounded Context + Citations]
    end

    subgraph Generation & Persistence
        CTX --> GEMINI[Google Gemini 3.5 Flash Lite]
        GEMINI --> SSE[Server-Sent Events Stream]
        SSE --> FE
        API --> DB[(Supabase PostgreSQL)]
        API --> OBS[Langfuse Observability Tracing]
    end
```

---

## 5. Input Documents & Dataset Profile

The system includes pre-ingested enterprise data from the **Redwood Inference Enterprise Benchmark Corpus**:
- **Total Documents:** `5,144`
- **Total Chunks:** `20,211`
- **Chunk Size Budget:** 350–380 tokens with monotonic identifiers (`{doc_id}#{idx:04d}`).
- **Supported Formats:** PDF, DOCX, TXT, Markdown, Slack threads, email chains, Fireflies meeting notes.
- **Enterprise Roles:** `employee`, `sales_rep`, `manager`, `executive`.

---

## 6. Empirical Evaluation Highlights

Evaluated against the **EnterpriseRAG-Bench** benchmark suite (`/Users/falahi/datasets-test`):

| Metric | Hybrid RRF + Reranker (Baseline) | Dense Alone | Sparse Alone (Postgres FTS) |
|---|:---:|:---:|:---:|
| **Permission Leakage Rate** | **0.0% (Zero Leakage)** | **0.0%** | **0.0%** |
| **Recall@5** | **55.61%** | 55.61% | 0.0% |
| **Precision@5** | **18.58%** | 18.58% | 0.0% |
| **MRR** | **0.6524** | 0.6578 | 0.0000 |
| **nDCG@5** | **0.5774** | 0.5751 | 0.0000 |
| **Prompt Injection Refusal** | **100.0%** | 0.0% | 100.0% |

> **Key Takeaway:** While dense retrieval alone achieves raw semantic recall, it exhibits a **0.0% Injection Refusal rate** (vulnerable to prompt attacks). Introducing the **Cross-Encoder reranker with the `-8.5` refusal gate raised Injection Refusal to 100%**, acting as an automated semantic firewall against adversarial queries.

---

## 7. Setup & Installation Guide

### Prerequisites
- **Python:** 3.12+ (managed with `uv`)
- **Node.js:** v20+ & npm
- **Supabase CLI & Docker:** Docker Desktop running
- **Ollama:** Installed locally (`brew install ollama` on macOS)

---

### Step 1: Clone & Configure Environment

```bash
# Clone the repository
git clone https://github.com/mohammed-falahi/enterprise-chatbot.git
cd enterprise-chatbot

# Configure Backend Environment
cp backend/.env.example backend/.env
```

Ensure `backend/.env` contains your Supabase credentials and Gemini API key:
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:54322/postgres
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_JWT_AUDIENCE=authenticated
GEMINI_API_KEY=your_gemini_api_key_here
```

---

### Step 2: Start Local Ollama & Pull Embedding Model

```bash
# Start Ollama daemon
ollama serve

# Pull the active Qwen3 embedding model (in a separate terminal)
ollama pull qwen3-embedding:0.6b
```

Verify that the model outputs 1024 dimensions:
```bash
curl -s http://127.0.0.1:11434/api/embed -d '{"model":"qwen3-embedding:0.6b","input":"test"}' | jq '.embeddings[0] | length'
# Output should be: 1024
```

---

### Step 3: Start Supabase & Run Database Migrations

```bash
# Start local Supabase containers (PostgreSQL, Auth, Storage)
supabase start

# Migrations in supabase/migrations/ apply automatically on start.
# To verify migration status:
supabase migration list
```

---

### Step 4: Run Backend Automated Tests

```bash
cd backend
PYTHONPATH=. uv run pytest
# Expected result: 166 passed in ~23 seconds
```

---

### Step 5: Start the Application

**Terminal 1 — Backend API:**
```bash
cd backend
PYTHONPATH=. uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*API documentation available at `http://localhost:8000/docs`.*

**Terminal 2 — Next.js Frontend:**
```bash
cd frontend
npm install
npm run dev
```
*Application available at `http://localhost:3000`.*

---

## 8. Detailed Technical Documentation Links

For comprehensive architectural design, database schemas, evaluation methodologies, and demo runbooks, refer to:

- [**docs/PROJECT_TECHNICAL_REFERENCE.md**](docs/PROJECT_TECHNICAL_REFERENCE.md) — The authoritative technical reference document covering all five official evaluation requirements (Architecture, HLD, Technical Solution, Demo Script, and Input Documents).
- [**docs/LIVE_DEMO_CHECKLIST.md**](docs/LIVE_DEMO_CHECKLIST.md) — Practical pre-flight checklist, test user credential matrix, rehearsed demo script, and contingency runbook.
- [**architecture.md**](architecture.md) — System architectural baseline and technical constraints.

---

## 9. License

This project is proprietary and confidential, prepared for technical evaluation purposes.
