"""
save_examples.py — Pick interesting failure examples and save them.

This is Step 10 from the spec. We look for questions where the model
answered correctly with clean evidence but incorrectly after the evidence
was damaged. These are the clearest demonstrations that evidence damage
caused the failure.

For each chosen question we save a plain-text file containing:
    question
    reference answer
    clean evidence + clean answer
    corrupted evidence + corrupted answer

Files are saved to results/examples/.
"""

import sys
from pathlib import Path

import pandas as pd

# Make sure the project root is on the Python path, so "from src...."
# works no matter where we run this file from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_questions, build_paragraphs
from src.corruption import build_conditions, build_donor_pool

# Where the results table and output examples live.
RESULTS_FILE = Path("results/results.csv")
EXAMPLES_DIR = Path("results/examples")

# How many examples to save.
NUM_EXAMPLES = 5


def find_interesting_questions(df):
    """Find questions where clean succeeded but a corrupted condition failed.

    "Interesting" means: the clean answer was a perfect or near-perfect
    match (F1 = 1.0), and at least one corrupted condition scored 0.
    These are the clearest cases of evidence damage causing a failure.

    Returns a list of question IDs, most dramatic failures first.
    """
    interesting = []

    for question_id, group in df.groupby("question_id"):
        # Pull out the F1 score for each condition for this question.
        f1_by_condition = dict(zip(group["condition"], group["f1"]))

        clean_f1 = f1_by_condition.get("clean", 0)

        # We want: clean answer was perfect, and at least one corrupted
        # condition completely failed.
        corrupted_failed = any(
            f1_by_condition.get(c, 1) == 0
            for c in ["remove_one", "remove_all", "contaminated"]
        )

        if clean_f1 == 1.0 and corrupted_failed:
            # Count how many corrupted conditions failed (more = more
            # dramatic, so we sort by it later).
            num_failed = sum(
                1 for c in ["remove_one", "remove_all", "contaminated"]
                if f1_by_condition.get(c, 1) == 0
            )
            interesting.append((question_id, num_failed))

    # Sort so questions with the most corrupted failures come first.
    interesting.sort(key=lambda x: x[1], reverse=True)
    return [qid for qid, _ in interesting]


def save_example(question, df, all_questions):
    """Save one example file for a question.

    Inputs:
        question      — the HotpotQA example (dictionary).
        df            — the full results DataFrame.
        all_questions — all 80 questions (needed to rebuild the donor pool
                        for the contaminated condition).
    """
    question_id = question["id"]
    rows = df[df["question_id"] == question_id]

    # Rebuild the paragraphs and the 4 conditions for this question.
    paragraphs = build_paragraphs(question)
    donor_pool = build_donor_pool(all_questions, question_id)
    conditions = build_conditions(paragraphs, donor_pool, question["question"])

    # Start building the text content of the example file.
    lines = []
    lines.append(f"Question ID: {question_id}")
    lines.append(f"Question: {question['question']}")
    lines.append(f"Reference answer: {question['answer']}")
    lines.append(f"Type: {question['type']} | Level: {question['level']}")
    lines.append("")

    # For each condition, show the evidence and the model's answer.
    for condition_name in ["clean", "remove_one", "remove_all", "contaminated"]:
        row = rows[rows["condition"] == condition_name].iloc[0]
        condition_paragraphs = conditions[condition_name]

        lines.append("=" * 70)
        lines.append(f"CONDITION: {condition_name.upper()}")
        lines.append("=" * 70)
        lines.append(f"Paragraphs available: {len(condition_paragraphs)} "
                     f"({sum(p['is_supporting'] for p in condition_paragraphs)} supporting)")
        lines.append(f"Retrieved (top-4): {row['retrieved_ids']}")
        lines.append(f"Generated answer: {row['generated_answer']}")
        lines.append(f"F1: {row['f1']:.3f} | Exact match: {row['exact_match']}")
        lines.append("")
        lines.append("Evidence paragraphs:")
        for p in condition_paragraphs:
            marker = "SUPPORTING" if p["is_supporting"] else "distractor"
            lines.append(f"  [{p['paragraph_id']}] ({marker}) {p['title']}")
            # Show a short preview of each paragraph (first 200 characters).
            lines.append(f"      {p['text'][:200]}...")
        lines.append("")

    # Write the file.
    out_file = EXAMPLES_DIR / f"example_{question_id}.txt"
    with open(out_file, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_FILE)
    all_questions = load_questions()

    # Find the most interesting questions and save the top N.
    interesting_ids = find_interesting_questions(df)
    print(f"Found {len(interesting_ids)} questions where clean succeeded "
          f"but a corrupted condition failed.\n")

    # Build a lookup from question ID to the full question dictionary.
    question_by_id = {q["id"]: q for q in all_questions}

    for question_id in interesting_ids[:NUM_EXAMPLES]:
        save_example(question_by_id[question_id], df, all_questions)

    print(f"\nSaved {min(NUM_EXAMPLES, len(interesting_ids))} examples to "
          f"{EXAMPLES_DIR}")
