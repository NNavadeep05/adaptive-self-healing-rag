"""Tests for the v2 retrieval building blocks (no Qdrant or model calls)."""

from langgraph_agent.bm25 import BM25Index, reciprocal_rank_fusion, tokenize
from langgraph_agent.metadata import (
    chunk_header,
    detect_companies,
    filing_metadata,
    point_id,
)
from langgraph_agent.table_text import html_to_text


# ---------------------------------------------------------
# Company metadata
# ---------------------------------------------------------

def test_filing_metadata_from_known_filename():
    assert filing_metadata("amazon_2025.html") == {"company": "Amazon", "fiscal_year": "2025"}
    assert filing_metadata("jpmorgan_2025.html")["company"] == "JPMorgan Chase"


def test_filing_metadata_for_unknown_upload():
    assert filing_metadata("my_notes.pdf") == {"company": None, "fiscal_year": None}


def test_detect_companies_by_name_and_alias():
    assert detect_companies("What was JPMorgan Chase's net income?") == ["JPMorgan Chase"]
    assert detect_companies("How did AWS grow?") == ["Amazon"]
    assert detect_companies("Compare Microsoft's FY2026 revenue with NVIDIA's") == [
        "Microsoft", "NVIDIA"]


def test_detect_companies_ignores_substrings_and_unknowns():
    assert detect_companies("What is the chaser effect?") == []
    assert detect_companies("What was revenue?") == []


def test_chunk_header_names_company_and_year():
    header = chunk_header({"company": "Amazon", "fiscal_year": "2025"}, "amazon_2025.html")
    assert header == "[Amazon | Form 10-K | fiscal year 2025]"
    assert chunk_header({"company": None, "fiscal_year": None}, "notes.pdf") == "[Source: notes.pdf]"


def test_point_id_is_deterministic_and_content_sensitive():
    a = point_id("amazon_2025.html", 3, "text")
    assert a == point_id("amazon_2025.html", 3, "text")
    assert a != point_id("amazon_2025.html", 3, "other text")
    assert a != point_id("amazon_2025.html", 4, "text")


# ---------------------------------------------------------
# BM25 + rank fusion
# ---------------------------------------------------------

def test_tokenize_keeps_numbers_whole():
    assert tokenize("Net sales were $716,924 million.") == ["net", "sales", "716924", "million"]


def test_bm25_ranks_exact_term_match_first():
    index = BM25Index(["total net sales were 716924",
                       "employees and culture",
                       "net income and total assets"])
    ranked = index.search("total net sales", limit=3)
    assert ranked[0] == 0
    assert 1 not in ranked  # no shared terms -> not returned


def test_bm25_prefers_rare_terms():
    docs = ["revenue revenue revenue", "revenue goodwill"]
    assert BM25Index(docs).search("goodwill revenue", limit=2)[0] == 1


def test_bm25_respects_allowed_subset():
    index = BM25Index(["net sales", "net sales again", "unrelated"])
    assert index.search("net sales", limit=3, allowed={1}) == [1]


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert fused[:2] in (["a", "b"], ["b", "a"])
    assert fused.index("b") < fused.index("c")
    assert set(fused) == {"a", "b", "c", "d"}


def test_reciprocal_rank_fusion_single_list_keeps_order():
    assert reciprocal_rank_fusion([["x", "y", "z"]]) == ["x", "y", "z"]


# ---------------------------------------------------------
# Table-aware HTML parsing
# ---------------------------------------------------------

TABLE_HTML = """
<html><body><p>CONSOLIDATED STATEMENTS OF INCOME</p>
<table>
  <tr><td>Year Ended</td><td>2025</td><td>2024</td></tr>
  <tr><td>Net income</td><td>$</td><td>57,048</td><td></td><td>$</td><td>58,471</td></tr>
  <tr><td>Net loss</td><td>(1,234</td><td>)</td></tr>
</table><script>ignored()</script></body></html>
"""


def test_table_rows_stay_on_one_line():
    text = html_to_text(TABLE_HTML)
    assert "Net income | 57,048 | 58,471" in text
    assert "Year Ended | 2025 | 2024" in text


def test_table_parsing_merges_split_negatives_and_drops_scripts():
    text = html_to_text(TABLE_HTML)
    assert "Net loss | (1,234)" in text
    assert "ignored" not in text
    assert "CONSOLIDATED STATEMENTS OF INCOME" in text


def test_table_parsing_tightens_spaced_parentheses():
    html = "<table><tr><td>Financing</td><td>( 15,879 )</td><td>9,661</td></tr></table>"
    assert "Financing | (15,879) | 9,661" in html_to_text(html)


# ---------------------------------------------------------
# Per-company decomposition
# ---------------------------------------------------------

from langgraph_agent.hybrid_retrieval import entity_precision, interleave  # noqa: E402


def test_interleave_alternates_and_skips_duplicates():
    assert interleave([["m1", "m2", "m3"], ["n1", "m2", "n2"]]) == ["m1", "n1", "m2", "m3", "n2"]


def test_interleave_handles_uneven_lists():
    assert interleave([["a"], ["b", "c", "d"]]) == ["a", "b", "c", "d"]


def test_entity_precision_counts_chunks_from_target_companies():
    companies = ["Amazon", "Microsoft", "Amazon", None]
    assert entity_precision(companies, ["Amazon"]) == 0.5
    assert entity_precision([], ["Amazon"]) is None
