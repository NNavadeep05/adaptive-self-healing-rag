"""Tests for the evaluation answer-matching logic."""

from eval.scoring import contains_answer


# ---------------------------------------------------------
# Numeric ground truths
# ---------------------------------------------------------

def test_exact_number_with_thousands_separators_matches():
    assert contains_answer("331,839", ["Revenue was $331,839 million."])


def test_number_is_not_split_into_fragments_on_commas():
    # The old scorer split "2,142,534" into "2", "142", "534" and
    # accepted any text containing those three substrings.
    text = ["We opened 2 stores, closed 142 kiosks and hired 534 people."]
    assert not contains_answer("2,142,534", text)


def test_number_must_match_a_whole_token():
    assert not contains_answer("57,048", ["Net income was $157,048 million."])
    assert not contains_answer("57,048", ["Net income was $57,0489 million."])


def test_number_written_without_separators_matches():
    assert contains_answer("716,924", ["Net sales: 716924 (in millions)"])


def test_decimal_figure_matches_whole_token_only():
    assert contains_answer("2.1 million", ["approximately 2.1 million associates"])
    assert not contains_answer("2.1 million", ["approximately 12.1 million associates"])


def test_exactly_equivalent_billion_figure_matches_millions():
    # "$331.839 billion" is the same value as "$331,839 million".
    assert contains_answer("331,839", ["Total revenue was **$331.839 billion**."])


def test_rounded_billion_figure_does_not_match_exact_millions():
    assert not contains_answer("331,839", ["Total revenue was about $331.8 billion."])


def test_billion_variant_still_matches_itself():
    assert contains_answer("713,163", ["Revenue reached $713.2 billion."],
                           variants=["713.2 billion"])


def test_sign_and_accounting_negatives_do_not_match_positive_figure():
    assert not contains_answer("57,048", ["a net loss of -57,048 million"])
    assert not contains_answer("57,048", ["a net loss of (57,048) million"])


def test_parenthesised_names_still_match():
    # Regression: the accounting-negative rule must only apply to numbers.
    answer = "1. North America 2. International 3. Amazon Web Services (AWS)"
    assert contains_answer("North America, International, AWS", [answer])


def test_different_unit_or_percent_does_not_match():
    assert not contains_answer("57,048", ["57,048 thousand"])
    assert not contains_answer("5", ["growth of 5% year over year"])
    assert not contains_answer("5", ["growth of 5 percent"])


def test_number_in_a_range_or_code_does_not_match():
    assert not contains_answer("57,048", ["reference Q1 57,048-12"])


def test_matching_unit_still_matches():
    assert contains_answer("57,048", ["net income was $57,048 million"])
    assert contains_answer("2.1 million", ["2.1 million associates"])


def test_citation_markers_are_ignored():
    assert not contains_answer("12", ["Revenue grew strongly (see Document 12)."])
    assert not contains_answer("10", ["Revenue grew 【Document 10†L1-L4】."])
    assert not contains_answer("3", ["Revenue grew [Document 3]."])
    assert contains_answer("331,839", ["Revenue was $331,839 million【Document 3】."])


def test_wrong_number_does_not_match():
    assert not contains_answer("713,163", ["Total revenue was roughly $713,127 million."])


# ---------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------

def test_unicode_spaces_and_hyphens_are_normalised():
    # Real LLM output for Q07 used U+202F (narrow no-break space).
    answer = "two segments:\n- **Compute & Networking**\n- **Graphics**"
    assert contains_answer("Compute & Networking, Graphics", [answer])


def test_markdown_and_case_are_ignored():
    assert contains_answer("North America, International, AWS",
                           ["**north america**, **International** and **AWS**"])


def test_all_parts_of_a_multi_part_answer_are_required():
    assert not contains_answer("North America, International, AWS",
                               ["Segments: North America and AWS."])


def test_wrong_entity_answer_does_not_match():
    microsoft_segments = ("Productivity and Business Processes, Intelligent Cloud, "
                          "More Personal Computing")
    assert not contains_answer("North America, International, AWS", [microsoft_segments])


def test_matching_spans_multiple_texts():
    assert contains_answer("Compute & Networking, Graphics",
                           ["Compute & Networking segment", "Graphics segment"])


# ---------------------------------------------------------
# Accepted variants and missing ground truth
# ---------------------------------------------------------

def test_accepted_variant_matches_when_primary_form_absent():
    assert contains_answer("713,163", ["Walmart reported $713.2 billion."],
                           variants=["713.2 billion"])


def test_variants_do_not_loosen_the_primary_match():
    assert not contains_answer("713,163", ["$713,127 million"], variants=["713.2 billion"])


def test_empty_ground_truth_returns_none():
    assert contains_answer("", ["anything"]) is None
    assert contains_answer(None, ["anything"]) is None


def test_empty_texts_do_not_match():
    assert not contains_answer("331,839", [])


# ---------------------------------------------------------
# Percentages (hard-bucket computed metrics)
# ---------------------------------------------------------

def test_percent_ground_truth_matches_percent_answer():
    assert contains_answer("40.31%", ["Net margin is about 40.31 %."])
    assert contains_answer("40.31%", ["net margin ≈ **40.31%**"])


def test_percent_ground_truth_rejects_other_precision():
    assert not contains_answer("40.31%", ["Net margin is 40.3%."])
    assert not contains_answer("40.31%", ["Net margin is 140.31%."])
