---
title: EndeeAgent Memory
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🧠 AgentMemory> **Long-term episodic memory for AI agents, powered by [Endee](https://github.com/endee-io/endee) vector database.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Endee](https://img.shields.io/badge/vector--db-Endee-brightgreen)](https://github.com/endee-io/endee)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-required-blue)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-pytest-informational)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview & Problem Statement

Modern AI assistants are **stateless by default**. Every new conversation starts from a blank slate — the agent has no memory of who you are, what you've built together, or what preferences you've expressed before.

This means:
- You re-introduce yourself every session ("I'm Alex, I work in fintech…")
- The agent forgets your tech stack, coding style, and ongoing projects
- No personalisation across sessions

**AgentMemory solves this** by giving any AI agent a persistent, searchable long-term memory backed by **Endee**, a high-performance vector database capable of handling **up to 1 billion vectors on a single node**.

When you ask the agent "How do I set up async DB access?", it silently recalls from Endee that you are a senior engineer building a FastAPI + PostgreSQL backend, that you prefer concise code examples, and that your deadline is end of March — and responds accordingly, without you having to repeat any of that.

---

## 🏗️ System Design & Technical Approach

```
┌────────────────────────────────────────────────────────────────────────┐
│                        AgentMemory Architecture                        │
│                                                                        │
│  USER MESSAGE                                                          │
│       │                                                                │
│       ▼                                                                │
│  ┌─────────────┐  embed()  ┌──────────────────┐                       │
│  │  Embedder   │──────────▶│   Endee Index    │  index.query()        │
│  │(MiniLM-L6)  │           │  (cosine, INT8)  │──────────┐            │
│  └─────────────┘           └──────────────────┘          │            │
│                                      ▲                    ▼            │
│                               upsert │         ┌─────────────────┐    │
│                                      │         │ Top-K Memories  │    │
│  ┌─────────────┐  summarise  ┌───────┴──────┐  └────────┬────────┘    │
│  │  LLM Buffer │────────────▶│ MemoryStore  │           │            │
│  │(session buf)│             │ (Endee wrap) │           ▼            │
│  └─────────────┘             └──────────────┘  ┌─────────────────┐   │
│         ▲                                       │  LLM (Mistral)    │   │
│         │                                       │  + context      │   │
│         └───────────────────────────────────────└────────┬────────┘   │
│                                                          │            │
│                                                    ANSWER + SOURCES   │
└────────────────────────────────────────────────────────────────────────┘
```

### Memory Lifecycle

| Phase | What Happens | Endee Operation |
|-------|-------------|-----------------|
| **Recall** | User sends a message → embed it → find similar past memories | `index.query(vector, top_k=5)` |
| **Generate** | Inject memories into LLM system prompt → produce grounded answer | — |
| **Buffer** | Append (user, assistant) turns to session buffer | — |
| **Checkpoint** | Every N turns → LLM summarises buffer → save to Endee | `index.upsert([{id, vector, meta}])` |
| **Persist** | On SIGINT / session end → flush remaining buffer | `index.upsert(...)` |

### Why this architecture works

- **Semantic, not keyword**: Endee's ANN search finds memories by meaning, not exact words. "How do I handle async DB?" will recall "User is building FastAPI + PostgreSQL" even though neither "async" nor "DB" appears verbatim.
- **Scales indefinitely**: Endee handles 1B+ vectors; your agent can accumulate years of memory.
- **Summarisation before storage**: The LLM distills conversations into compact, information-dense memories before embedding — reducing noise and storage costs.
- **Cross-session recall**: Memories from any past session are searchable, enabling true long-term personalisation.

---

## 🧩 How Endee Is Used

AgentMemory uses the **official Endee Python SDK** (`pip install endee`) as its sole vector store. All vector operations go through Endee.

### Index Configuration

```python
from endee import Endee, Precision

client = Endee()  # connects to http://localhost:8080 by default
client.create_index(
    name="agent_memory",
    dimension=384,           # all-MiniLM-L6-v2 embedding size
    space_type="cosine",     # cosine similarity (normalised dot-product)
    precision=Precision.INT8 # INT8 quantisation: 4× smaller, ~2× faster
)
```

### Storing a Memory (Upsert)

```python
index = client.get_index("agent_memory")
index.upsert([
    {
        "id": "sess_abc_1706345600000_d3f1a2",
        "vector": [0.12, -0.34, ...],   # 384-dim embedding of the summary
        "meta": {
            "session_id":  "sess_abc",
            "summary":     "User prefers dark mode; asked about React hooks.",
            "role":        "mixed",
            "turn":        12,
            "tags":        ["preference", "react"],
            "timestamp":   "2026-02-27T10:30:00Z"
        }
    }
])
```

### Recalling Memories (Query)

```python
query_vector = embed("What UI preferences does the user have?")  # 384 floats

results = index.query(vector=query_vector, top_k=5)
# results[i].id          → memory ID
# results[i].similarity  → cosine similarity score (0–1)
# results[i].meta        → full metadata dict including summary text
```

The `summary` text from each result is assembled into a context block and injected into the LLM's system prompt before answering.

### Why Endee?

| Requirement | How Endee Meets It |
|-------------|-------------------|
| Fast ANN search (< 5 ms at 1M vectors) | SIMD-optimised HNSW (AVX2/AVX512/NEON/SVE2) |
| Single-node scalability | Up to 1B vectors per node |
| Simple integration | Official Python SDK + REST API |
| Persistent storage across restarts | Docker volume (`endee-data`) |
| Low operational overhead | Single Docker container |
| Open source | Apache-2.0 licence |

---

## 📁 Project Structure

```
agentmemory/
├── docker-compose.yml       ← Starts Endee (endeeio/endee-server:latest)
├── .env.example             ← Configuration template
├── requirements.txt
├── README.md
├── LICENSE
│
├── src/
│   ├── __init__.py
│   ├── config.py            ← Centralised settings (reads .env)
│   ├── embedder.py          ← sentence-transformers wrapper (384-dim vectors)
│   ├── memory_store.py      ← MemoryEntry + MemoryStore (Endee SDK wrapper) ⭐
│   ├── summariser.py        ← LLM summarisation & answer generation
│   ├── agent.py             ← MemoryAgent orchestrator
│   ├── cli.py               ← Rich interactive REPL
│   ├── api.py               ← FastAPI HTTP server
│   └── __main__.py
│
├── scripts/
│   └── demo.py              ← Standalone demo (no LLM key needed)
│
└── tests/
    ├── test_embedder.py
    ├── test_memory_store.py
    ├── test_agent.py
    └── test_api.py
```

---

## 🚀 Setup & Execution Instructions

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Docker + Docker Compose | v2+ | Run Endee vector DB |
| Python | 3.10+ | Run AgentMemory |
| Mistral API key | — | LLM for answers & summarisation |

---

### Step 1 — Start Endee

```bash
docker compose up -d
```

Endee will start on port `8080`. Verify it's running:

```bash
curl http://localhost:8080/api/v1/index/list
# → {"indexes":[]}
```

Or open [http://localhost:8080](http://localhost:8080) in your browser to access the Endee dashboard.

---

### Step 2 — Install Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

### Step 3 — Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required for LLM
LLM_PROVIDER=mistral
MISTRAL_API_KEY=YOUR_MISTRAL_API_KEY
MISTRAL_MODEL=mistral-large-latest
```

Everything else works with defaults for local development.

---

### Step 4 — Run the Demo (no LLM key needed)

The demo seeds pre-written memories into Endee and demonstrates semantic recall without an LLM call:

```bash
python scripts/demo.py
```

Expected output:

```
AgentMemory – Demo

① Seeding past memories into Endee...
✓ Seeded 6 memories into Endee.

② Demonstrating semantic recall from Endee...

Query: What UI preferences does the user have?
┌─────┬──────────────────────────────────────────────────────┬─────────────────────┐
│ Rank│ Retrieved Memory                                      │ Tags                │
├─────┼──────────────────────────────────────────────────────┼─────────────────────┤
│ #1  │ The user prefers dark mode across all applications... │ preference, ui, ... │
│ #2  │ Alex prefers concise code examples over lengthy...    │ preference, coding  │
└─────┴──────────────────────────────────────────────────────┴─────────────────────┘
...

③ Simulating a conversation with memory injection...
```

---

### Step 5 — Interactive Chat (requires LLM key)

```bash
python -m src.cli
```

**Resume a named session** (memories persist across restarts):

```bash
python -m src.cli --session my_project_sess
```

**In-chat commands:**

| Command | Description |
|---------|-------------|
| `/recall <query>` | Search Endee memories semantically |
| `/history` | Show all memories from this session |
| `/save` | Force checkpoint to Endee |
| `/stats` | Show Endee index statistics |
| `exit` | Quit and auto-save session |

**Example session:**

```
AgentMemory — AI agent with long-term episodic memory
Vector DB: Endee  │  Session: sess_f3a1b2c4

You: My name is Alex and I'm building a FastAPI app with PostgreSQL
Agent: Nice to meet you, Alex! FastAPI + PostgreSQL is a great combination...

You: I prefer dark mode and concise code examples
Agent: Got it — I'll keep examples tight and minimal going forward.

[... 20 turns later, memories auto-saved to Endee ...]

# ── New process, same session ──
$ python -m src.cli --session sess_f3a1b2c4

You: Help me with my database setup
Agent: Sure Alex! Since you're working on FastAPI + PostgreSQL and prefer
       concise examples, here's what you need...
       ^ Remembers your name, stack, and preferences from the previous session
```

---

### Step 6 — REST API Server

```bash
uvicorn src.api:app --reload --port 7860
```

Open [http://localhost:7860/docs](http://localhost:7860/docs) for the interactive Swagger UI.

**Key endpoints:**

```bash
# Create / resume a session
curl -X POST http://localhost:7860/sessions \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my_session"}'

# Chat
curl -X POST http://localhost:7860/sessions/my_session/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What frameworks should I use for my project?"}'

# Semantic memory search
curl -X POST http://localhost:7860/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "user programming preferences", "top_k": 5}'

# View session memories
curl http://localhost:7860/sessions/my_session/memories
```

---

### Step 7 — Run Tests

```bash
pytest tests/ -v
```

All tests run offline (Endee and LLM calls are mocked).

---

## 🔧 Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ENDEE_BASE_URL` | `http://localhost:8080` | Endee server address |
| `ENDEE_AUTH_TOKEN` | *(empty)* | Auth token (optional for local dev) |
| `ENDEE_INDEX_NAME` | `agent_memory` | Name of the Endee index |
| `LLM_PROVIDER` | `mistral` | `mistral` |
| `MISTRAL_API_KEY` | — | Your Mistral key |
| `MISTRAL_MODEL` | `mistral-large-latest` | Mistral model name |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `MEMORY_TOP_K` | `5` | Memories retrieved per query |
| `SUMMARY_MAX_TOKENS` | `150` | Max tokens per memory summary |
| `SESSION_WINDOW` | `20` | Turns before auto-checkpoint |

---

## 📈 Performance Characteristics

| Metric | Value |
|--------|-------|
| Embedding latency (CPU) | ~20 ms per text |
| Endee upsert (1 memory) | < 5 ms |
| Endee query (1M vectors) | < 5 ms (SIMD-optimised HNSW) |
| Memory per vector (INT8, 384-dim) | ~400 bytes |
| Max vectors (single node) | 1,000,000,000 |
| End-to-end turn latency | ~1–3 s (LLM-dominated) |

---

## ☁️ Free Deployment (Hugging Face Spaces)

You can host AgentMemory 100% for free on **Hugging Face Spaces**, which automatically spins up both the Endee Vector Database and the FastAPI server within a single Docker container.

1. Create a new [Hugging Face Space](https://huggingface.co/new-space).
2. Choose **Docker** as the Space SDK and select the **Blank** template.
3. Link your GitHub repository to your Hugging Face Space (or push the code manually).
4. In your Hugging Face Space settings, add your `MISTRAL_API_KEY` under **Variables and secrets** (as a Secret).
5. The included `Dockerfile` and `start.sh` script will automatically:
   - Start the Endee Vector Database in the background (`localhost:8080`).
   - Start the FastAPI web server on Hugging Face's required port (`7860`).
6. Once the build finishes, your API will be live!

---

## 🛠️ Production Notes

- Set `NDD_AUTH_TOKEN` in both `docker-compose.yml` and `.env` for authenticated access
- For high-throughput: increase `NDD_NUM_THREADS` in `docker-compose.yml`
- Back up Endee data with: `docker run --rm -v endee-data:/data -v $(pwd):/backup alpine tar czf /backup/endee-backup.tar.gz /data`
- Use AVX512 build on server CPUs (Intel Xeon / AMD EPYC) for maximum speed

---

## 🤝 Contributing

Pull requests are welcome. Please open an issue first for major changes.

---

## 📄 License

MIT © 2026
