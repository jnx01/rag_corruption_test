"""
evaluate.py — Measure answer quality and retrieval quality.

This is Module 6 from the spec. We compute exactly three metrics:

    1. ANSWER F1 — how much the words of the predicted answer overlap
       with the words of the reference answer (0.0 to 1.0, higher better).
       This is the standard F1 used by HotpotQA and SQuAD.

    2. EXACT MATCH — 1 if the normalized prediction is identical to the
       normalized reference, else 0. Very strict, very simple.

    3. SUPPORTING EVIDENCE RECALL — of the supporting paragraphs that
       EXIST in this condition, what fraction made it into the top-4
       retrieval? (0.0 to 1.0, higher better.)

Before comparing answers, we "normalize" them: lowercase, remove
punctuation, remove filler words like "the", and collapse extra spaces.
This stops trivial differences ("Honolulu" vs "honolulu.") from counting
as mistakes.
"""

import re
import string
from collections import Counter


# ---------------------------------------------------------------------------
# Text normalization (standard HotpotQA/SQuAD-style)
# ---------------------------------------------------------------------------

def normalize_answer(text):
    """Clean up an answer string so comparisons are fair.

    Steps: lowercase -> remove punctuation -> remove articles
    ("a", "an", "the") -> collapse whitespace.
    """
    # Lowercase everything.
    text = text.lower()

    # Remove all punctuation characters.
    text = "".join(ch for ch in text if ch not in string.punctuation)

    # Remove English articles, which usually don't carry meaning.
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Collapse any run of whitespace into a single space, and trim ends.
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Metric 1: Answer F1
# ---------------------------------------------------------------------------

def answer_f1(prediction, reference):
    """Compute word-overlap F1 between prediction and reference.

    F1 is the harmonic mean of:
        precision = (shared words) / (words in prediction)
        recall    = (shared words) / (words in reference)

    Returns a float between 0.0 (no overlap) and 1.0 (perfect match).
    """
    pred_words = normalize_answer(prediction).split()
    ref_words = normalize_answer(reference).split()

    # If both are empty, that's a perfect match. If only one is empty,
    # there is nothing to overlap, so F1 is 0.
    if len(pred_words) == 0 and len(ref_words) == 0:
        return 1.0
    if len(pred_words) == 0 or len(ref_words) == 0:
        return 0.0

    # Count shared words. Counter handles duplicates correctly:
    # "new york new" vs "new new york" shares min(2,2)=2 "new" etc.
    common = Counter(pred_words) & Counter(ref_words)
    num_shared = sum(common.values())

    if num_shared == 0:
        return 0.0

    precision = num_shared / len(pred_words)
    recall = num_shared / len(ref_words)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ---------------------------------------------------------------------------
# Metric 2: Exact Match
# ---------------------------------------------------------------------------

def exact_match(prediction, reference):
    """Return 1 if normalized answers are identical, else 0."""
    return int(normalize_answer(prediction) == normalize_answer(reference))


# ---------------------------------------------------------------------------
# Metric 3: Supporting evidence recall
# ---------------------------------------------------------------------------

def supporting_recall(retrieved_paragraphs, condition_paragraphs):
    """Fraction of available supporting paragraphs that were retrieved.

    Inputs:
        retrieved_paragraphs — the top-4 paragraphs from retrieval.
        condition_paragraphs — ALL paragraphs in this condition (so we can
                               see which supporting paragraphs exist).

    Output:
        A float between 0.0 and 1.0. If the condition contains NO
        supporting paragraphs (remove_all, contaminated), we return None
        because recall is not meaningful there — you can't find evidence
        that doesn't exist.
    """
    # Which paragraph IDs are supporting in this condition?
    supporting_ids = {p["paragraph_id"] for p in condition_paragraphs
                      if p["is_supporting"]}

    # No supporting evidence available -> recall is undefined.
    if not supporting_ids:
        return None

    # Which paragraph IDs were retrieved?
    retrieved_ids = {p["paragraph_id"] for p in retrieved_paragraphs}

    # Recall = (supporting paragraphs found) / (supporting paragraphs that exist)
    found = supporting_ids & retrieved_ids
    return len(found) / len(supporting_ids)


# ---------------------------------------------------------------------------
# Small test: run "python -m src.evaluate" to check this file works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Test normalization ---
    print("Normalization:")
    print("  'The Honolulu!' ->", repr(normalize_answer("The Honolulu!")))
    print()

    # --- Test F1 ---
    print("Answer F1:")
    print("  perfect match:", answer_f1("Arthur's Magazine", "Arthur's Magazine"))
    print("  partial match:", answer_f1("City and County of Honolulu", "Honolulu"))
    print("  no match:", answer_f1("Paris", "Honolulu"))
    print()

    # --- Test exact match ---
    print("Exact match:")
    print("  'Delhi' vs 'Delhi':", exact_match("Delhi", "Delhi"))
    print("  'the delhi' vs 'Delhi':", exact_match("the delhi", "Delhi"))
    print("  'New Delhi' vs 'Delhi':", exact_match("New Delhi", "Delhi"))
    print()

    # --- Test supporting recall ---
    print("Supporting recall:")
    fake_condition = [
        {"paragraph_id": "P0", "is_supporting": True},
        {"paragraph_id": "P1", "is_supporting": True},
        {"paragraph_id": "P2", "is_supporting": False},
    ]
    retrieved_both = [fake_condition[0], fake_condition[1], fake_condition[2]]
    retrieved_one = [fake_condition[0], fake_condition[2]]
    retrieved_none = [fake_condition[2]]
    print("  found 2 of 2:", supporting_recall(retrieved_both, fake_condition))
    print("  found 1 of 2:", supporting_recall(retrieved_one, fake_condition))
    print("  found 0 of 2:", supporting_recall(retrieved_none, fake_condition))
    no_support_condition = [{"paragraph_id": "P2", "is_supporting": False}]
    print("  no support exists:", supporting_recall(retrieved_none, no_support_condition))
