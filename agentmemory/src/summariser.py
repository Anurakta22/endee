"""
Summariser
==========
Converts a raw conversation window into a compact, searchable memory summary
using the configured LLM (OpenAI or Ollama).

Why summarise instead of storing raw messages?
  • Summaries are shorter → cheaper embeddings
  • LLM distills the *meaningful* information
  • One summary covers N messages → fewer Endee upserts
"""

from __future__ import annotations

from typing import List, Tuple

from .config import cfg

# ── Provider abstraction ──────────────────────────────────────────────────────

def _chat_openai(system: str, user: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=cfg.openai_api_key)
    response = client.chat.completions.create(
        model=cfg.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=cfg.summary_max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def _chat_ollama(system: str, user: str) -> str:
    import ollama

    response = ollama.chat(
        model=cfg.ollama_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response["message"]["content"].strip()


def _llm(system: str, user: str) -> str:
    if cfg.llm_provider == "ollama":
        return _chat_ollama(system, user)
    return _chat_openai(system, user)


# ── Public functions ──────────────────────────────────────────────────────────

_SUMMARY_SYSTEM = """You are a memory archivist for an AI assistant.
Your job is to distill conversation snippets into concise, factual memory entries.
Each memory entry should:
- Be 1–3 sentences max
- Capture user preferences, decisions, facts, or important context
- Use third-person references ("The user prefers...", "The assistant explained...")
- Omit pleasantries and filler
Output ONLY the memory text, no preamble."""


def summarise_window(messages: List[Tuple[str, str]]) -> str:
    """
    Produce a memory summary from a list of (role, content) message pairs.

    Parameters
    ----------
    messages : list of (role, content) tuples, e.g. [("user", "Hi"), ("assistant", "Hello")]

    Returns
    -------
    A short summary string suitable for embedding and storage in Endee.
    """
    if not messages:
        return ""

    conversation_text = "\n".join(
        f"{role.upper()}: {content}" for role, content in messages
    )
    return _llm(_SUMMARY_SYSTEM, conversation_text)


_TAG_SYSTEM = """You are a tagging assistant.
Given a memory summary, output a comma-separated list of 1–5 short topic tags (lowercase).
Examples: preference, coding, python, dark-mode, deadline
Output ONLY the comma-separated tags, nothing else."""


def extract_tags(summary: str) -> list[str]:
    """Return a list of topic tags for a memory summary."""
    try:
        raw = _llm(_TAG_SYSTEM, summary)
        return [t.strip().lower() for t in raw.split(",") if t.strip()][:5]
    except Exception:
        return []


_ANSWER_SYSTEM = """You are a helpful AI assistant with long-term memory.
You have access to relevant memories from past conversations.
Use them to give personalised, context-aware answers.
If memories are irrelevant to the current question, ignore them gracefully.
Always be concise and helpful."""


def generate_answer(
    user_message: str,
    memories: list,         # list[MemoryEntry]
    chat_history: List[Tuple[str, str]],
) -> str:
    """
    Generate an answer using retrieved memories as grounded context.

    Parameters
    ----------
    user_message  : The current user question.
    memories      : Retrieved MemoryEntry objects from Endee.
    chat_history  : Recent in-session messages (role, content) pairs.
    """
    # Build memory context block
    if memories:
        memory_block = "\n".join(
            f"[Memory {i+1}] {m.summary}  (from session {m.session_id}, turn {m.turn})"
            for i, m in enumerate(memories)
        )
        memory_context = f"RELEVANT MEMORIES FROM PAST SESSIONS:\n{memory_block}\n"
    else:
        memory_context = "No relevant past memories found.\n"

    # Build recent chat history block
    if chat_history:
        history_block = "\n".join(
            f"{role.upper()}: {content}" for role, content in chat_history[-6:]
        )
        history_context = f"\nRECENT CONVERSATION:\n{history_block}\n"
    else:
        history_context = ""

    system_prompt = f"{_ANSWER_SYSTEM}\n\n{memory_context}{history_context}"

    return _llm(system_prompt, user_message)
