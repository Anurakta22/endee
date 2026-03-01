# 🧠 AgentMemory

> Long-term episodic memory for AI agents, powered by [Endee](https://github.com/endee-io/endee) vector database.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face-yellow)](https://anurakta22-endeeagent-memory.hf.space)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Problem Statement

AI assistants are **stateless by default** — every conversation starts fresh. You re-introduce yourself, repeat your preferences, and re-explain your context every single time.

**AgentMemory fixes this.** It gives an AI agent persistent long-term memory using Endee as the vector database. The agent remembers who you are and what you've discussed, across sessions.

---

## 🏗️ How It Works

```
User message
    │
    ▼
Embedder (MiniLM-L6, 384-dim)
    │
    ├──► Endee: query top-5 similar past memories
    │
    ▼
LLM (Mistral) + memories as context → Answer
    │
    ▼
Every N turns → LLM summarises → Endee: store new memory
```

- **Recall**: User message is embedded → Endee returns semantically similar past memories
- **Generate**: Memories are injected into the LLM prompt → personalised, grounded answer
- **Store**: After every N turns, the conversation is summarised → saved back to Endee

---

## 🧩 How Endee Is Used

All vector operations go through the official **Endee Python SDK** (`pip install endee`).

```python
from endee import Endee, Precision

client = Endee()  # connects to http://localhost:8080
client.create_index("agent_memory", dimension=384, space_type="cosine", precision=Precision.INT8)

# Store a memory
index.upsert([{"id": "mem_001", "vector": embed("User prefers dark mode"), "meta": {...}}])

# Recall relevant memories
results = index.query(vector=embed("What UI does the user prefer?"), top_k=5)
```

Endee is used for:
- Storing conversation summaries as 384-dim vectors (INT8 quantised)
- Semantic recall — finds memories **by meaning**, not keywords
- Cross-session memory persistence

---

## 📁 Project Structure

```
agentmemory/
├── src/
│   ├── memory_store.py   ← Endee SDK wrapper (core of the project)
│   ├── agent.py          ← Orchestrates recall → generate → store
│   ├── embedder.py       ← sentence-transformers (MiniLM-L6)
│   ├── summariser.py     ← LLM summarisation & answer generation
│   ├── api.py            ← FastAPI REST server
│   └── cli.py            ← Interactive terminal chat
├── scripts/demo.py       ← Demo (no LLM key needed)
├── tests/                ← pytest unit tests
├── Dockerfile            ← For Hugging Face Spaces
├── start.sh              ← Starts Endee + FastAPI together
├── docker-compose.yml    ← Local Endee setup
└── .env.example          ← Config template
```

---

## 🚀 Setup & Running Locally

### 1. Start Endee

```bash
docker compose up -d
```

### 2. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY
```

### 4. Run the demo (no API key needed)

```bash
python scripts/demo.py
```

### 5. Interactive chat

```bash
python -m src.cli
```

### 6. REST API

```bash
uvicorn src.api:app --reload --port 7860
# Open http://localhost:7860/docs
```

### 7. Tests

```bash
pytest tests/ -v
```

---

## ☁️ Live Demo

The project is deployed on Hugging Face Spaces (free tier), running **Endee + FastAPI in a single Docker container**.

👉 **[anurakta22-endeeagent-memory.hf.space](https://anurakta22-endeeagent-memory.hf.space)**

Use the chat interface — or `/docs` for the raw API.

> **⚠️ Free Tier Limitation:** Hugging Face Spaces (free) puts containers to sleep after inactivity and resets them on restart. Since Endee runs inside the container, its stored memories are wiped on each restart. This is a **hosting constraint, not a limitation of Endee** — on a real server with a persistent Docker volume (`-v endee-data:/data`), memories would survive indefinitely. Locally, data persists across restarts as long as your Docker volume exists.

---

## ⚙️ Key Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | — | Required for LLM |
| `ENDEE_BASE_URL` | `http://localhost:8080` | Endee server |
| `MEMORY_TOP_K` | `5` | Memories recalled per query |
| `SESSION_WINDOW` | `20` | Turns before auto-checkpoint |

---

## 📄 License

MIT © 2026
