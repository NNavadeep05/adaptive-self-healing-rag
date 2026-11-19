"""
Table-aware HTML to text conversion for SEC filings.

The naive ``soup.get_text("\\n")`` puts every table cell on its own line,
so "Net income", "$" and "57,048" end up on three separate lines and can
be split into different chunks. Here each table row becomes one line:

    Net income | 57,048 | 58,471

which keeps the label and its numbers together for the embedder, BM25
and the LLM.
"""

import re

from bs4 import BeautifulSoup

_SPACE_AFTER_OPEN = re.compile(r"\(\s+")
_SPACE_BEFORE_CLOSE = re.compile(r"\s+\)")
_NON_TEXT_TAGS = ["script", "style", "noscript", "ix:header", "ix:hidden"]
_DROP_CELLS = {"$", ""}
_ATTACH_TO_PREVIOUS = {")", "%", ")%"}


def _clean_cell(text: str) -> str:
    # "( 15,879 )" -> "(15,879)" so negatives read like the printed filing.
    return _SPACE_AFTER_OPEN.sub("(", _SPACE_BEFORE_CLOSE.sub(")", text))


def _row_cells(row) -> list[str]:
    cells: list[str] = []
    for cell in row.find_all(["td", "th"]):
        text = _clean_cell(cell.get_text(" ", strip=True))
        if text in _DROP_CELLS:
            continue
        if text in _ATTACH_TO_PREVIOUS and cells:
            # SEC tables put the closing ")" of "(1,234)" in its own cell.
            cells = [*cells[:-1], cells[-1] + text]
            continue
        cells.append(text)
    return cells


def _table_to_text(table) -> str:
    rows = (" | ".join(_row_cells(row)) for row in table.find_all("tr"))
    return "\n".join(row for row in rows if row)


def html_to_text(html: str) -> str:
    """Visible text of an HTML filing with one line per table row."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NON_TEXT_TAGS):
        tag.decompose()
    # Innermost tables first so a nested table is flattened before its parent.
    for table in reversed(soup.find_all("table")):
        table.replace_with("\n" + _table_to_text(table) + "\n")
    return soup.get_text(separator="\n", strip=True)
