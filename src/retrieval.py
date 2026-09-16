"""
retrieval.py — Rank paragraphs with BM25 and return the top 4.

This is Module 4 from the spec. The idea is simple:

    question + list of paragraphs  ->  BM25 ranking  ->  top 4 paragraphs

BM25 is a classic lexical (word-matching) ranking method. It scores each
paragraph by how well its words match the question's words, with two
useful properties:

    - rare words count for more than common words
    - very long paragraphs do not get an unfair advantage

No embeddings, no machine learning model — just word statistics.
"""

from rank_bm25 import BM25Okapi

# How many paragraphs we keep after ranking. The spec says top 4.
TOP_K = 4


def _tokenize(text):
    """Split text into lowercase words.

    Same deliberately simple tokenizer as in corruption.py: lowercase
    everything and split on whitespace. BM25 only needs word counts.
    """
    return text.lower().split()


def retrieve(question_text, paragraphs, top_k=TOP_K):
    """Rank the given paragraphs against the question and return the top k.

    Inputs:
        question_text — the question string.
        paragraphs    — list of paragraph records (from data.py, possibly
                        modified by corruption.py).
        top_k         — how many paragraphs to return (default 4).

    Output:
        A list of up to `top_k` paragraph records, best match first.
        (If a condition has fewer than 4 paragraphs, we return all of them.)
    """

    # Edge case: if there are no paragraphs at all, return an empty list.
    if not paragraphs:
        return []

    # Build a BM25 index over the paragraphs we were given.
    # Each paragraph is represented as a list of lowercase words.
    tokenized_paragraphs = [_tokenize(p["text"]) for p in paragraphs]
    bm25 = BM25Okapi(tokenized_paragraphs)

    # Score every paragraph against the question.
    scores = bm25.get_scores(_tokenize(question_text))

    # Sort paragraph indices by score, highest first, and keep the top k.
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i],
                            reverse=True)
    top_indices = ranked_indices[:top_k]

    # Return the actual paragraph records in ranked order.
    return [paragraphs[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Small test: run "python -m src.retrieval" to check this file works
# (This is Step 5 from the spec: test retrieval on 5-10 questions and
#  check whether BM25 finds the supporting paragraphs in the clean condition.)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data import load_questions, build_paragraphs
    from src.corruption import build_conditions, build_donor_pool

    questions = load_questions()

    # Test on the first 10 questions, across all 4 conditions.
    for question in questions[:10]:
        paragraphs = build_paragraphs(question)
        donor_pool = build_donor_pool(questions, question["id"])
        conditions = build_conditions(paragraphs, donor_pool, question["question"])

        # The IDs of the paragraphs that ARE supporting evidence.
        supporting_ids = {p["paragraph_id"] for p in paragraphs
                          if p["is_supporting"]}

        print("=" * 70)
        print("Question:", question["question"][:75])
        print("Supporting paragraphs:", ", ".join(sorted(supporting_ids)))

        for condition_name, condition_paragraphs in conditions.items():
            top4 = retrieve(question["question"], condition_paragraphs)
            top4_ids = [p["paragraph_id"] for p in top4]

            # Which of the retrieved paragraphs are supporting evidence?
            retrieved_support = [pid for pid in top4_ids
                                 if pid in supporting_ids]

            print(f"  {condition_name:<13} top4: {', '.join(top4_ids):<20} "
                  f"supporting retrieved: {retrieved_support if retrieved_support else 'none'}")
        print()
