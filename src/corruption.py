"""
corruption.py — Create the 4 evidence conditions for each question.

This is Module 3 from the spec. For every question we start with its
original ~10 paragraphs (2 supporting + 8 distractors) and create four
versions:

    Condition 0 — CLEAN:        the original 10 paragraphs, untouched.
    Condition 1 — REMOVE_ONE:   remove one supporting paragraph
                                (1 supporting + 8 distractors remain).
    Condition 2 — REMOVE_ALL:   remove all supporting paragraphs
                                (0 supporting + 8 distractors remain).
    Condition 3 — CONTAMINATED: remove all supporting paragraphs AND add
                                2 extra paragraphs taken from OTHER questions.
                                The extra paragraphs are chosen with BM25 so
                                they are topically similar to the question
                                (plausible distractors, not random garbage).

The point of the experiment: as we go from condition 0 to 3, the evidence
available to the RAG system gets worse, and we measure how the answer
quality changes.
"""

import random

from rank_bm25 import BM25Okapi

# Fixed seed so "remove ONE supporting paragraph" always removes the same one.
# This keeps the experiment reproducible.
RANDOM_SEED = 42

# The four condition names, used everywhere (CSV columns, plots, report).
CONDITION_CLEAN = "clean"
CONDITION_REMOVE_ONE = "remove_one"
CONDITION_REMOVE_ALL = "remove_all"
CONDITION_CONTAMINATED = "contaminated"

ALL_CONDITIONS = [
    CONDITION_CLEAN,
    CONDITION_REMOVE_ONE,
    CONDITION_REMOVE_ALL,
    CONDITION_CONTAMINATED,
]


# ---------------------------------------------------------------------------
# Small helper: turn text into a list of lowercase words for BM25
# ---------------------------------------------------------------------------

def _tokenize(text):
    """Split text into lowercase words.

    This is a deliberately simple tokenizer: lowercase everything and split
    on whitespace. BM25 only needs word counts, so this is good enough.
    """
    return text.lower().split()


# ---------------------------------------------------------------------------
# The main function: build all 4 conditions for one question
# ---------------------------------------------------------------------------

def build_conditions(paragraphs, donor_pool, question_text):
    """Create the 4 evidence conditions for one question.

    Inputs:
        paragraphs    — list of paragraph records for THIS question
                        (from data.build_paragraphs).
        donor_pool    — paragraphs from OTHER questions, used to pick the
                        2 extra distractors for the contaminated condition.
        question_text — the question string, used to find similar donors
                        with BM25.

    Output:
        A dictionary mapping each condition name to its list of paragraphs.
        Example: conditions["remove_all"] -> list of 8 distractor paragraphs.
    """

    # Split the question's paragraphs into supporting and distractor groups.
    supporting = [p for p in paragraphs if p["is_supporting"]]
    distractors = [p for p in paragraphs if not p["is_supporting"]]

    # --- Condition 0: CLEAN -------------------------------------------------
    # Just the original paragraphs, exactly as they came from the dataset.
    clean = list(paragraphs)

    # --- Condition 1: REMOVE_ONE --------------------------------------------
    # Remove one supporting paragraph (chosen with a fixed seed so it is
    # always the same one). One supporting paragraph remains.
    rng = random.Random(RANDOM_SEED)
    removed_one = rng.choice(supporting)
    remaining_support = [p for p in supporting if p is not removed_one]
    remove_one = remaining_support + distractors

    # --- Condition 2: REMOVE_ALL --------------------------------------------
    # Remove every supporting paragraph. Only the 8 distractors remain.
    remove_all = list(distractors)

    # --- Condition 3: CONTAMINATED ------------------------------------------
    # Start from remove_all, then add 2 paragraphs from OTHER questions.
    # We use BM25 to pick donors that are topically similar to the question,
    # so they are plausible (misleading) rather than obviously irrelevant.
    #
    # We score donors against the question PLUS the question's own paragraph
    # texts. Using only the short question text gave weak matches in testing;
    # adding the paragraph text gives BM25 much more signal about the topic.
    similarity_query = question_text + " " + " ".join(p["text"] for p in paragraphs)
    extra_distractors = _pick_similar_donors(similarity_query, donor_pool, count=2)
    contaminated = distractors + extra_distractors

    return {
        CONDITION_CLEAN: clean,
        CONDITION_REMOVE_ONE: remove_one,
        CONDITION_REMOVE_ALL: remove_all,
        CONDITION_CONTAMINATED: contaminated,
    }


# ---------------------------------------------------------------------------
# Helper for the contaminated condition: find similar paragraphs elsewhere
# ---------------------------------------------------------------------------

def _pick_similar_donors(question_text, donor_pool, count=2):
    """Pick `count` paragraphs from the donor pool that are most similar
    to the question, according to BM25.

    Inputs:
        question_text — the question string.
        donor_pool    — list of paragraph records from OTHER questions.
        count         — how many donor paragraphs to return (spec says 2).

    Output:
        A list of `count` paragraph records (copies with fresh IDs so they
        can never be confused with the question's own paragraphs).
    """

    # If the donor pool is empty (should not happen), just return nothing.
    if not donor_pool:
        return []

    # Build a BM25 index over all donor paragraphs.
    tokenized_donors = [_tokenize(p["text"]) for p in donor_pool]
    bm25 = BM25Okapi(tokenized_donors)

    # Score every donor paragraph against the question.
    scores = bm25.get_scores(_tokenize(question_text))

    # Get the indices of the top-scoring donors (highest score first).
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i],
                         reverse=True)[:count]

    # Return COPIES of the donor paragraphs with new IDs (e.g. "D0", "D1").
    # Fresh IDs matter: a donor paragraph must never share an ID with one of
    # the question's own paragraphs, or our retrieval-recall metric could
    # get confused.
    donors = []
    for new_id, index in enumerate(top_indices):
        original = donor_pool[index]
        donors.append({
            "paragraph_id": f"D{new_id}",
            "title": original["title"],
            "text": original["text"],
            "is_supporting": False,  # donors are never supporting evidence
        })
    return donors


# ---------------------------------------------------------------------------
# Helper: build a donor pool from all questions EXCEPT the current one
# ---------------------------------------------------------------------------

def build_donor_pool(all_questions, exclude_question_id):
    """Collect paragraphs from every question except one.

    Inputs:
        all_questions       — the full list of selected HotpotQA examples.
        exclude_question_id — the ID of the question we are currently
                              building conditions for (its paragraphs must
                              NOT be in the pool).

    Output:
        A flat list of paragraph records from all other questions.
    """
    # Import here (instead of at the top) to avoid a circular import:
    # data.py does not import corruption.py, but corruption.py needs
    # build_paragraphs from data.py.
    from src.data import build_paragraphs

    pool = []
    for question in all_questions:
        if question["id"] == exclude_question_id:
            continue  # never donate a question's own paragraphs to itself
        pool.extend(build_paragraphs(question))
    return pool


# ---------------------------------------------------------------------------
# Small test: run "python -m src.corruption" to check this file works
# (This is Step 4 from the spec: test corruption manually on 5 questions.)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data import load_questions, build_paragraphs

    questions = load_questions()

    # Test on the first 5 questions, as the spec suggests.
    for question in questions[:5]:
        paragraphs = build_paragraphs(question)
        donor_pool = build_donor_pool(questions, question["id"])
        conditions = build_conditions(paragraphs, donor_pool, question["question"])

        supporting_ids = [p["paragraph_id"] for p in paragraphs if p["is_supporting"]]

        print("=" * 70)
        print("Question:", question["question"][:80])
        print()
        print("CLEAN")
        print("  supporting paragraphs:", ", ".join(supporting_ids))
        print()
        remove_one_support = [p["paragraph_id"] for p in conditions["remove_one"]
                              if p["is_supporting"]]
        print("REMOVE_ONE")
        print("  remaining support:", ", ".join(remove_one_support))
        print()
        print("REMOVE_ALL")
        print("  remaining support: none")
        print()
        contaminated = conditions["contaminated"]
        donor_ids = [p["paragraph_id"] for p in contaminated
                     if p["paragraph_id"].startswith("D")]
        donor_titles = [p["title"] for p in contaminated
                        if p["paragraph_id"].startswith("D")]
        print("CONTAMINATED")
        print("  remaining support: none")
        print(f"  additional similar distractors: {', '.join(donor_ids)}")
        print(f"  donor titles: {donor_titles}")
        print(f"  total paragraphs: {len(contaminated)}")
        print()
