"""MCP server exposing virtual-context as tools, resources, and prompts."""

from __future__ import annotations

import json
import logging
import os

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "virtual-context",
    instructions="Domain-aware virtual memory for persistent context across long sessions",
)

# Lazy engine singleton
_engine = None


def _get_engine():
    """Get or create the engine singleton."""
    global _engine
    if _engine is None:
        from ..engine import VirtualContextEngine
        config_path = os.environ.get("VIRTUAL_CONTEXT_CONFIG")
        _engine = VirtualContextEngine(config_path=config_path)
    return _engine


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def recall_context(
    message: str,
    active_tags: list[str] | None = None,
) -> str:
    """Retrieve relevant context summaries for an inbound message.

    Tags the message, fetches matching stored summaries by tag overlap,
    and returns the assembled context block ready to prepend to a prompt.

    Args:
        message: The user message to find relevant context for.
        active_tags: Tags already present in recent conversation (will be skipped).

    Returns:
        The assembled virtual-context block as XML, or empty string if no relevant context.
    """
    engine = _get_engine()
    return engine.transform(message, active_tags=active_tags)


@mcp.tool()
def compact_context(
    messages: list[dict],
) -> str:
    """Compact a conversation into stored domain summaries.

    Takes conversation messages, segments them by topic, summarizes each
    segment via LLM, and stores the summaries for future retrieval.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        JSON summary of compaction results.
    """
    from ..types import Message
    engine = _get_engine()

    parsed = [
        Message(role=m.get("role", "user"), content=m.get("content", ""))
        for m in messages
    ]

    report = engine.compact_manual(parsed)
    if report is None:
        return json.dumps({"status": "no_compaction", "reason": "No content to compact or no LLM provider"})

    return json.dumps({
        "status": "compacted",
        "segments_compacted": report.segments_compacted,
        "tokens_freed": report.tokens_freed,
        "tags": report.tags,
    })


@mcp.tool()
def domain_status() -> str:
    """Show statistics for all stored tags/domains.

    Returns tag names, usage counts, and token totals.

    Returns:
        JSON array of tag statistics.
    """
    engine = _get_engine()
    tags = engine._store.get_all_tags()
    return json.dumps([
        {
            "tag": t.tag,
            "usage_count": t.usage_count,
            "total_full_tokens": t.total_full_tokens,
            "total_summary_tokens": t.total_summary_tokens,
            "oldest_segment": t.oldest_segment.isoformat() if t.oldest_segment else None,
            "newest_segment": t.newest_segment.isoformat() if t.newest_segment else None,
        }
        for t in tags
    ], indent=2)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("virtualcontext://domains")
def list_domains() -> str:
    """List all stored tags/domains with their statistics."""
    engine = _get_engine()
    tags = engine._store.get_all_tags()
    lines = []
    for t in tags:
        lines.append(f"- **{t.tag}**: {t.usage_count} segments, {t.total_summary_tokens} summary tokens")
    return "\n".join(lines) if lines else "No domains stored yet."


@mcp.resource("virtualcontext://domains/{tag}")
def get_domain_summaries(tag: str) -> str:
    """Get all stored summaries for a specific tag/domain."""
    engine = _get_engine()
    summaries = engine._store.get_summaries_by_tags(tags=[tag], min_overlap=1, limit=50)
    if not summaries:
        return f"No summaries found for tag: {tag}"

    parts = []
    for s in summaries:
        parts.append(
            f"## {s.ref}\n"
            f"Tags: {', '.join(s.tags)}\n"
            f"Tokens: {s.summary_tokens}\n"
            f"Created: {s.created_at.isoformat()}\n\n"
            f"{s.summary}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

@mcp.prompt()
def recall(topic: str) -> str:
    """Recall stored context about a specific topic.

    Use this when you want to retrieve everything the system knows about a topic.

    Args:
        topic: The topic to recall context about.
    """
    return f"Please recall any stored context relevant to: {topic}"


@mcp.prompt()
def summarize_session() -> str:
    """Request a session summary for compaction.

    Use this to trigger compaction of the current conversation.
    """
    return (
        "Please compact the current conversation. "
        "Identify topic segments, summarize each independently, "
        "and store the summaries for future retrieval."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve():
    """Start the MCP server (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    serve()
