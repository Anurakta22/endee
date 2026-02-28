"""
AgentMemory CLI
===============
A rich interactive REPL for chatting with the MemoryAgent.

Usage
-----
  python -m src.cli                          # new session each run
  python -m src.cli --session sess_abc123    # resume a named session
  python -m src.cli --recall "React hooks"  # show memories matching query
  python -m src.cli --history               # show all memories this session
  python -m src.cli --stats                 # show Endee index stats
"""

from __future__ import annotations

import signal
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import MemoryAgent

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
console = Console()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_memories(memories, title="Recalled Memories"):
    if not memories:
        console.print(f"[dim]No memories found.[/dim]")
        return
    table = Table(title=title, show_lines=True, highlight=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Summary", style="white")
    table.add_column("Session", style="cyan", width=16)
    table.add_column("Turn", style="yellow", width=5)
    table.add_column("Tags", style="green")
    for i, m in enumerate(memories, 1):
        table.add_row(
            str(i),
            m.summary,
            m.session_id,
            str(m.turn),
            ", ".join(m.tags),
        )
    console.print(table)


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def main(
    session: str = typer.Option(None, "--session", "-s", help="Session ID to resume"),
    recall: str = typer.Option(None, "--recall", "-r", help="Query memories and exit"),
    history: bool = typer.Option(False, "--history", "-H", help="Show session memories"),
    stats: bool = typer.Option(False, "--stats", help="Show Endee index stats"),
):
    """AgentMemory – AI agent with long-term memory powered by Endee."""

    agent = MemoryAgent(session_id=session)

    # ── Non-interactive modes ──────────────────────────────────────────────────

    if stats:
        info = agent.store.stats()
        console.print(Panel(str(info), title="[bold]Endee Index Stats[/bold]"))
        return

    if recall:
        console.print(f"\n[bold]Searching memories for:[/bold] [cyan]{recall}[/cyan]\n")
        memories = agent.recall(recall)
        _print_memories(memories, title=f"Top memories for: '{recall}'")
        return

    # ── Interactive REPL ───────────────────────────────────────────────────────

    console.print(
        Panel(
            Text.assemble(
                ("AgentMemory", "bold green"),
                " — AI agent with long-term episodic memory\n",
                ("Vector DB: ", "dim"),
                ("Endee  ", "bold cyan"),
                ("│  Session: ", "dim"),
                (agent.session_id, "bold yellow"),
            ),
            subtitle="[dim]Type 'exit' to quit  •  '/recall <query>' to inspect memories  •  '/history' for this session[/dim]",
        )
    )

    if history:
        mems = agent.session_history()
        _print_memories(mems, title=f"Memory history for session {agent.session_id}")
        if not typer.confirm("\nContinue chatting?", default=True):
            return

    # Graceful shutdown on Ctrl-C: flush remaining buffer to Endee
    def _sigint_handler(sig, frame):
        console.print("\n\n[yellow]Saving session to memory...[/yellow]")
        saved = agent.force_checkpoint()
        if saved:
            console.print("[green]✓ Session checkpoint saved to Endee.[/green]")
        console.print("[dim]Goodbye.[/dim]")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    # ── Chat loop ──────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            _sigint_handler(None, None)

        if not user_input:
            continue

        # Built-in slash commands
        if user_input.lower() in {"exit", "quit", "/exit", "/quit"}:
            _sigint_handler(None, None)

        if user_input.lower().startswith("/recall "):
            query = user_input[8:].strip()
            memories = agent.recall(query)
            _print_memories(memories, title=f"Memories matching: '{query}'")
            continue

        if user_input.lower() in {"/history", "/h"}:
            _print_memories(agent.session_history(), title="Session Memory History")
            continue

        if user_input.lower() in {"/stats"}:
            console.print(agent.store.stats())
            continue

        if user_input.lower() in {"/checkpoint", "/save"}:
            saved = agent.force_checkpoint()
            msg = "✓ Checkpoint saved." if saved else "Buffer was empty, nothing to save."
            console.print(f"[green]{msg}[/green]")
            continue

        # Normal chat turn
        with console.status("[dim]Thinking...[/dim]", spinner="dots"):
            answer = agent.chat(user_input)

        console.print(f"\n[bold green]Agent:[/bold green] {answer}\n")


if __name__ == "__main__":
    app()
