"""
AgentMemory Demo Script
=======================
Demonstrates the full memory lifecycle without needing a live LLM:
  1. Creates an agent
  2. Seeds several pre-written memories into Endee
  3. Queries Endee to show semantic recall
  4. Simulates a multi-turn conversation with memory injection

Run:
    python scripts/demo.py

Requirements:
    - Endee running on localhost:8080 (docker compose up -d)
    - ENDEE_BASE_URL set in .env (default: http://localhost:8080)
    - No LLM key needed (LLM calls are bypassed in demo mode)
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.memory_store import MemoryEntry, MemoryStore
from src.embedder import embed

console = Console()


def seed_memories(store: MemoryStore) -> None:
    console.print("\n[bold yellow]① Seeding past memories into Endee...[/bold yellow]")

    seed_data = [
        {
            "summary": "The user prefers dark mode across all applications and finds light mode strains their eyes.",
            "session_id": "sess_past_001",
            "tags": ["preference", "ui", "dark-mode"],
            "turn": 3,
        },
        {
            "summary": "The user is building a FastAPI backend with PostgreSQL. They asked about async SQLAlchemy patterns.",
            "session_id": "sess_past_001",
            "tags": ["coding", "python", "fastapi", "database"],
            "turn": 12,
        },
        {
            "summary": "The user mentioned their name is Alex and they work as a senior software engineer at a fintech startup.",
            "session_id": "sess_past_002",
            "tags": ["personal", "career"],
            "turn": 1,
        },
        {
            "summary": "Alex prefers concise code examples over lengthy explanations. They know Python well but are learning Rust.",
            "session_id": "sess_past_002",
            "tags": ["preference", "coding", "python", "rust"],
            "turn": 8,
        },
        {
            "summary": "The user asked about vector databases and was specifically interested in how HNSW indexing works.",
            "session_id": "sess_past_003",
            "tags": ["ml", "vector-db", "hnsw"],
            "turn": 5,
        },
        {
            "summary": "The user's project deadline is end of March 2026. They are working on an AI-powered search feature.",
            "session_id": "sess_past_003",
            "tags": ["deadline", "project", "ai-search"],
            "turn": 15,
        },
    ]

    entries = [MemoryEntry(**d) for d in seed_data]
    store.save_batch(entries)
    console.print(f"[green]✓ Seeded {len(entries)} memories into Endee.[/green]")


def demo_recall(store: MemoryStore) -> None:
    console.print("\n[bold yellow]② Demonstrating semantic recall from Endee...[/bold yellow]")

    queries = [
        "What UI preferences does the user have?",
        "What programming languages is the user learning?",
        "Tell me about the user's current project",
        "How does vector indexing work?",
    ]

    for q in queries:
        console.print(f"\n[cyan]Query:[/cyan] {q}")
        memories = store.recall(q, top_k=2)

        table = Table(show_lines=True, min_width=80)
        table.add_column("Rank", style="dim", width=5)
        table.add_column("Retrieved Memory", style="white")
        table.add_column("Tags", style="green", width=25)

        for i, m in enumerate(memories, 1):
            table.add_row(f"#{i}", m.summary, ", ".join(m.tags))

        console.print(table)


def demo_conversation(store: MemoryStore) -> None:
    console.print("\n[bold yellow]③ Simulating a conversation with memory injection...[/bold yellow]")

    user_message = "Can you help me set up async database access for my project?"

    console.print(f"\n[bold blue]User:[/bold blue] {user_message}")

    # Retrieve relevant memories
    memories = store.recall(user_message, top_k=3)

    # Build context (simulates what the LLM would receive)
    memory_context = "\n".join(
        f"  [{i+1}] {m.summary}" for i, m in enumerate(memories)
    )

    console.print(
        Panel(
            memory_context or "(no memories found)",
            title="[bold]Relevant memories retrieved from Endee[/bold]",
            border_style="yellow",
        )
    )

    # Simulate a grounded response (no real LLM call in demo)
    response = (
        "Sure Alex! Since you're already working on a FastAPI + PostgreSQL setup "
        "and prefer concise examples, here's the minimal async SQLAlchemy pattern:\n\n"
        "```python\n"
        "from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession\n"
        "from sqlalchemy.orm import sessionmaker\n\n"
        "engine = create_async_engine('postgresql+asyncpg://user:pass@localhost/db')\n"
        "AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)\n"
        "```\n\n"
        "Given your March deadline, this should get you unblocked quickly."
    )

    console.print(f"\n[bold green]Agent:[/bold green] {response}")
    console.print(
        "\n[dim italic]^ The agent personalised its response using memories "
        "from past sessions (name, tech stack, preferences, deadline) "
        "retrieved in real-time from Endee.[/dim italic]"
    )


def main():
    console.print(
        Panel(
            "[bold green]AgentMemory[/bold green] – Demo\n"
            "[dim]Long-term episodic memory for AI agents, powered by Endee vector DB[/dim]",
            expand=False,
        )
    )

    try:
        store = MemoryStore()
        console.print("[green]✓ Connected to Endee.[/green]")
    except Exception as e:
        console.print(f"[red]✗ Could not connect to Endee: {e}[/red]")
        console.print("[yellow]Make sure Endee is running: docker compose up -d[/yellow]")
        sys.exit(1)

    seed_memories(store)
    demo_recall(store)
    demo_conversation(store)

    console.print(
        "\n[bold green]Demo complete![/bold green] "
        "Start the interactive CLI with: [cyan]python -m src.cli[/cyan]\n"
    )


if __name__ == "__main__":
    main()
