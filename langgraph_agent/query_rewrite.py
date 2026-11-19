"""
Query rewriting, used as a healing action in the v2 loop.

A user asks "How much did Amazon sell?"; the filing says "Total net
sales". Rewriting the question into the vocabulary of a 10-K (statement
names, line items, fiscal year) helps both BM25 and the embedder find
the right chunk when the first retrieval missed.
"""

import logging

from langgraph_agent.retrieve_docs import get_groq_client

logger = logging.getLogger(__name__)

REWRITE_MODEL = "openai/gpt-oss-120b"

REWRITE_PROMPT = """Rewrite the question below into a search query for
finding the answer inside an SEC Form 10-K annual report.

- Keep the company name and fiscal year.
- Add the exact line-item or section names a 10-K would use
  (for example "Total net sales", "Net income", "Total assets",
  "Consolidated Statements of Income", "Human Capital", "Segment Information").
- Add one or two close synonyms.
- Return only the search query on one line, no explanation.

Question: {query}"""


def rewrite_query(query: str) -> str:
    """Return a 10-K-vocabulary search query, or the original on failure."""
    try:
        response = get_groq_client().chat.completions.create(
            model=REWRITE_MODEL,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
        )
        rewritten = (response.choices[0].message.content or "").strip()
    except Exception:  # the loop can still widen retrieval without a rewrite
        logger.warning("Query rewrite failed; keeping the original query", exc_info=True)
        return query
    return rewritten.splitlines()[0].strip() if rewritten else query
