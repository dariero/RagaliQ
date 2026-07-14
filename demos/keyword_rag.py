"""Unstructured-retrieval RAG: keyword search over text with SQLite FTS5.

This is the *retrieval half* of a tiny RAG system under test (SUT). It has zero
external dependencies — only Python's built-in ``sqlite3`` with the FTS5
full-text-search engine.

Phase A, step 1 (A1): build the index and prove ``retrieve(query)`` returns the
right passages. No LLM calls happen here; this step costs nothing.

Run it directly to see retrieval in action::

    .venv/bin/python demos/keyword_rag.py
"""

from __future__ import annotations

import re
import sqlite3

# --- The knowledge base ------------------------------------------------------
# Deliberately FICTIONAL facts about a made-up city-state. Fiction matters: it
# forces any correct answer to come from THESE passages, not from the model's
# training data. That is what makes a faithfulness score meaningful later.
FACTS: list[str] = [
    "Zentra is a fictional island city-state founded in the year 1820.",
    "The official currency of Zentra is the zent, abbreviated ZT.",
    "Zentra's national dish is grilled saltfish served with roasted yam.",
    "The tallest building in Zentra is the Vox Tower, which stands 410 meters tall.",
    "Every October, Zentra celebrates a harvest festival called Lumen.",
    "The official language of Zentra is Calic, spoken by most residents.",
]

# Tiny stopword set so question words ("what", "is", "the") don't dominate the
# keyword query. A real system would use a proper tokenizer; this keeps the demo
# transparent — you can see exactly which words drive the search.
_STOPWORDS = frozenset(
    {"the", "is", "of", "a", "an", "what", "when", "where", "who", "how", "are", "does", "did"}
)


def build_index(conn: sqlite3.Connection) -> None:
    """Create an FTS5 virtual table and load the facts into it.

    Args:
        conn: An open SQLite connection (typically in-memory).
    """
    conn.execute("CREATE VIRTUAL TABLE facts USING fts5(body)")
    conn.executemany("INSERT INTO facts(body) VALUES (?)", [(fact,) for fact in FACTS])


def to_match_query(query: str) -> str:
    """Turn a natural-language question into an FTS5 keyword expression.

    FTS5's ``MATCH`` treats characters like ``?`` and quotes as query syntax and
    raises on them, so we extract bare alphanumeric keywords, drop stopwords, and
    join the rest with ``OR``.

    Args:
        query: The user's natural-language question.

    Returns:
        An ``OR``-joined keyword string (empty if the query has no usable terms).
    """
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    keywords = [tok for tok in tokens if tok not in _STOPWORDS]
    return " OR ".join(keywords)


def retrieve(conn: sqlite3.Connection, query: str, k: int = 2) -> list[str]:
    """Return the top-``k`` passages matching ``query``, ranked by relevance.

    Args:
        conn: An open SQLite connection with the FTS5 index built.
        query: The user's natural-language question.
        k: Maximum number of passages to return.

    Returns:
        The matching passage strings, best match first. Empty if nothing matches.
    """
    match = to_match_query(query)
    if not match:
        return []
    rows = conn.execute(
        "SELECT body FROM facts WHERE facts MATCH ? ORDER BY rank LIMIT ?",
        (match, k),
    ).fetchall()
    return [row[0] for row in rows]


def main() -> None:
    """Build the index and demonstrate retrieval for a few sample questions."""
    conn = sqlite3.connect(":memory:")
    build_index(conn)

    questions = [
        "What is the currency of Zentra?",
        "How tall is the Vox Tower?",
        "When is the Lumen festival held?",
    ]

    print(f"Indexed {len(FACTS)} facts into an FTS5 table.\n")
    for question in questions:
        match = to_match_query(question)
        passages = retrieve(conn, question, k=2)
        print(f"Q: {question}")
        print(f"   MATCH expression : {match}")
        print(f"   retrieved ({len(passages)}):")
        for passage in passages:
            print(f"     - {passage}")
        print()

    conn.close()


if __name__ == "__main__":
    main()
