"""
run_experiment.py — Run the full experiment and save results to CSV.

This is the pipeline that ties all the modules together. For every
question and every condition it does:

    question
      -> build the 4 evidence conditions   (corruption.py)
      -> retrieve top-4 paragraphs         (retrieval.py)
      -> ask the LLM for an answer         (generate.py, cached)
      -> compute the 3 metrics             (evaluate.py)
      -> save one row to results.csv

Usage:
    python -m experiments.run_experiment            # full run: 80 questions
    python -m experiments.run_experiment --limit 5  # pilot: 5 questions

The experiment is resumable: answers are cached in
results/answer_cache.json, so re-running never repeats an API call.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Make sure the project root is on the Python path, so "from src...."
# works no matter where we run this file from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_questions, build_paragraphs
from src.corruption import build_conditions, build_donor_pool, ALL_CONDITIONS
from src.retrieval import retrieve
from src.generate import generate_answer, make_client, _get_provider, MODELS
from src.evaluate import answer_f1, exact_match, supporting_recall

# Where the final results table is saved.
RESULTS_FILE = Path("results/results.csv")


def run_experiment(limit=None):
    """Run the experiment over all (or the first `limit`) questions.

    Input:
        limit — if given, only use the first `limit` questions
                (used for the 5-question pilot).

    Output:
        A pandas DataFrame with one row per question-condition pair,
        also saved to results/results.csv.
    """

    # Load the API key and create ONE client for the whole run (reusing it is
    # faster than creating a new one per call). The provider comes from .env.
    provider = _get_provider()
    client = make_client(provider)
    print(f"Using provider: {provider} | model: {MODELS[provider]}")

    # Load our 80 selected questions.
    questions = load_questions()
    if limit is not None:
        questions = questions[:limit]
        print(f"PILOT MODE: using only the first {limit} questions.\n")

    # Build the donor pool ONCE from all questions. It is used to pick the
    # 2 extra distractors for the contaminated condition. (A question's own
    # paragraphs are excluded inside build_donor_pool.)
    print("Building donor pool from all questions...")
    all_questions = load_questions()

    rows = []  # one dictionary per question-condition pair

    for q_index, question in enumerate(questions, start=1):
        paragraphs = build_paragraphs(question)
        donor_pool = build_donor_pool(all_questions, question["id"])
        conditions = build_conditions(paragraphs, donor_pool, question["question"])

        print(f"[{q_index}/{len(questions)}] {question['question'][:60]}...")

        for condition_name in ALL_CONDITIONS:
            condition_paragraphs = conditions[condition_name]

            # Retrieve the top-4 paragraphs for this condition.
            top4 = retrieve(question["question"], condition_paragraphs)

            # Ask the LLM for an answer (uses the cache if already done).
            answer = generate_answer(
                question["id"], condition_name, question["question"],
                top4, client=client, provider=provider,
            )

            # Compute the three metrics.
            f1 = answer_f1(answer, question["answer"])
            em = exact_match(answer, question["answer"])
            recall = supporting_recall(top4, condition_paragraphs)

            # Count how many words of context we sent to the model
            # (the optional "context size" metric from the spec).
            context_words = sum(len(p["text"].split()) for p in top4)

            rows.append({
                "question_id": question["id"],
                "question": question["question"],
                "question_type": question["type"],
                "level": question["level"],
                "condition": condition_name,
                "reference_answer": question["answer"],
                "generated_answer": answer,
                "f1": f1,
                "exact_match": em,
                "retrieval_recall": recall,
                "context_words": context_words,
                "retrieved_ids": ",".join(p["paragraph_id"] for p in top4),
            })

            print(f"    {condition_name:<13} f1={f1:.2f} em={em} "
                  f"recall={recall if recall is not None else 'n/a'}")

    # Turn all rows into a table and save it as CSV.
    results = pd.DataFrame(rows)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved {len(results)} rows to {RESULTS_FILE}")

    return results


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow "--limit N" on the command line to run a small pilot.
    parser = argparse.ArgumentParser(description="Run the RAG evidence "
                                     "corruption experiment.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only use the first N questions (for pilots).")
    args = parser.parse_args()

    run_experiment(limit=args.limit)
