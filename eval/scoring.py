"""
Answer matching for the evaluation scripts.

A ground truth such as "2,142,534" or "North America, International, AWS"
is matched against one or more texts (retrieved chunks or a generated
answer). The rules:

- Text is normalised: Unicode NFKC, unusual spaces and hyphens collapsed,
  markdown bold and "$" removed, lower-cased.
- Thousands separators inside numbers are removed *before* splitting,
  so "2,142,534" stays a single number instead of three fragments.
- Remaining commas / semicolons split a multi-part answer; every part
  must be present.
- Each part must match as a whole token, so "57,048" does not match
  "157,048" and "2.1 million" does not match "12.1 million".
- Numbers must also have the right sign and unit: "-57,048", "(57,048)",
  "57,048 thousand" and "5%" do not match "57,048" / "5".
- "40.31 %" is normalised to "40.31%", so a percentage ground truth
  matches either spacing (but not a different precision such as "40.3%").
- Citation markers such as "【Document 3】" are removed first.
- "N billion" is converted to millions when the conversion is exact,
  so "$331.839 billion" matches "331,839" (million) but "$331.8 billion"
  does not.
"""

import re
import unicodedata
from decimal import Decimal
from typing import Sequence

_HYPHENS = str.maketrans({
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
})
_WHITESPACE = re.compile(r"\s+")
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_PART_SEPARATOR = re.compile(r"[,;]")
_SPACE_BEFORE_PERCENT = re.compile(r"(?<=\d)\s+%")
_BILLIONS = re.compile(r"(?<![\w.])(\d+(?:\.\d{1,3})?) billion\b")
# LLM citation markers such as "【Document 3†L1-L4】" or "[Document 3]"
# contain numbers that must not count as answers.
_CITATIONS = re.compile(r"【[^】]*】|\[\s*document\s+\d+[^\]]*\]|\bdocument\s+\d+\b")


def _billions_to_millions(match: re.Match[str]) -> str:
    # Exact conversion only: "331.839 billion" -> "331839 million".
    # A rounded "331.8 billion" becomes "331800 million" and therefore
    # does not match an exact "331,839" million ground truth.
    millions = Decimal(match.group(1)) * 1000
    return f"{int(millions)} million"


def normalize(text: str) -> str:
    """Return a canonical, comparison-friendly form of ``text``."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_HYPHENS)
    text = text.replace("**", "").replace("$", "").lower()
    text = _CITATIONS.sub(" ", text)
    text = _THOUSANDS_SEPARATOR.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip()
    text = _SPACE_BEFORE_PERCENT.sub("%", text)
    return _BILLIONS.sub(_billions_to_millions, text)


def _answer_parts(ground_truth: str) -> list[str]:
    parts = _PART_SEPARATOR.split(normalize(ground_truth))
    return [part.strip() for part in parts if part.strip()]


def _part_pattern(part: str) -> re.Pattern[str]:
    escaped = re.escape(part)
    if not part[0].isdigit():
        # Whole-word match for names such as "graphics" or "aws".
        return re.compile(r"(?<![\w.])" + escaped + r"(?!\w)")
    # Numbers additionally reject a leading minus sign, a range or code
    # ("57048-12"), and a different unit or percentage right after them.
    return re.compile(
        r"(?<![\w.\-])" + escaped
        + r"(?!\w|\.\d|-\d|\s*%|\s*(?:thousand|percent)\b)"
    )


def _part_present(part: str, haystack: str) -> bool:
    is_number = part[0].isdigit()
    for match in _part_pattern(part).finditer(haystack):
        start, end = match.span()
        # "(57,048)" is the accounting notation for a negative number;
        # a parenthesised name such as "(aws)" is not.
        is_accounting_negative = (
            is_number
            and haystack[start - 1:start] == "("
            and haystack[end:end + 1] == ")"
        )
        if not is_accounting_negative:
            return True
    return False


def _all_parts_present(answer: str, haystack: str) -> bool:
    parts = _answer_parts(answer)
    if not parts:
        return False
    return all(_part_present(part, haystack) for part in parts)


def contains_answer(
    ground_truth: str | None,
    texts: Sequence[str],
    variants: Sequence[str] = (),
) -> bool | None:
    """
    Check whether the expected answer appears in ``texts``.

    Args:
        ground_truth: Expected answer. Multi-part answers are separated
            by commas or semicolons.
        texts: Retrieved chunks or a generated answer.
        variants: Other accepted phrasings (e.g. "713.2 billion" for
            "713,163" million). Each variant is matched with the same rules.

    Returns:
        True if the ground truth (or a variant) is fully present,
        False if not, None if no ground truth is defined.
    """
    if not ground_truth:
        return None

    haystack = normalize(" \n ".join(texts))
    if not haystack:
        return False

    candidates = [ground_truth, *variants]
    return any(_all_parts_present(candidate, haystack) for candidate in candidates)
