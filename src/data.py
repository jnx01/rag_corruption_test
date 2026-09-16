"""
data.py — Load HotpotQA and build the experimental subset.

This file does two jobs (Module 1 and Module 2 from the spec):

1. DATA LOADER: download HotpotQA (distractor / validation) and pick
   about 80 questions for our experiment — roughly 40 "bridge" and
   40 "comparison" questions, preferring medium/hard difficulty.

2. CORPUS BUILDER: turn each question's context into a simple list of
   paragraph records. Each record has:
       paragraph_id   (e.g. "P0", "P1", ...)
       title          (the Wikipedia article title it came from)
       text           (the paragraph text, sentences joined together)
       is_supporting  (True if HotpotQA says this paragraph contains
                       evidence needed to answer the question)
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Settings (kept at the top so they are easy to find and change)
# ---------------------------------------------------------------------------

# How many questions we want in total, and per question type.
TARGET_TOTAL = 80
TARGET_PER_TYPE = 40  # ~40 bridge + ~40 comparison

# Fixed random seed so we always pick the SAME 80 questions.
# This makes the experiment reproducible: anyone running this file
# gets the exact same subset.
RANDOM_SEED = 42

# Where to save the list of selected question IDs, so we have a
# permanent record of exactly which questions were used.
SELECTED_IDS_FILE = Path("results/selected_question_ids.json")


# ---------------------------------------------------------------------------
# Job 1: load the dataset and select the subset of questions
# ---------------------------------------------------------------------------

def load_questions():
    """Load HotpotQA validation split and select our ~80 questions.

    Returns a list of plain dictionaries (one per question), each with:
        id, question, answer, type, level, supporting_facts, context
    """

    # Download (or load from local cache) the HotpotQA "distractor" config.
    # We use the validation split, exactly as the spec says.
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    # Keep only medium and hard questions. The spec says to prefer these
    # so our subset is not dominated by very easy questions.
    candidates = [ex for ex in dataset if ex["level"] in ("medium", "hard")]

    # Split the candidates into the two question types we care about.
    bridge_questions = [ex for ex in candidates if ex["type"] == "bridge"]
    comparison_questions = [ex for ex in candidates if ex["type"] == "comparison"]

    # Shuffle each group with a fixed seed, then take the first 40 of each.
    # Using a seeded Random object means the shuffle is always identical.
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(bridge_questions)
    rng.shuffle(comparison_questions)

    selected_bridge = bridge_questions[:TARGET_PER_TYPE]
    selected_comparison = comparison_questions[:TARGET_PER_TYPE]

    # Combine the two groups into one list.
    selected = selected_bridge + selected_comparison

    # Save the selected question IDs to a JSON file as a permanent record.
    SELECTED_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTED_IDS_FILE, "w") as f:
        json.dump([ex["id"] for ex in selected], f, indent=2)

    # Print a short summary so we can see what we got.
    print(f"Selected {len(selected)} questions "
          f"({len(selected_bridge)} bridge, {len(selected_comparison)} comparison)")
    level_counts = {}
    for ex in selected:
        level_counts[ex["level"]] = level_counts.get(ex["level"], 0) + 1
    print(f"Difficulty breakdown: {level_counts}")
    print(f"Question IDs saved to: {SELECTED_IDS_FILE}")

    return selected


# ---------------------------------------------------------------------------
# Job 2: turn one question's context into simple paragraph records
# ---------------------------------------------------------------------------

def build_paragraphs(example):
    """Turn one HotpotQA example into a list of paragraph records.

    Input: one HotpotQA example (a dictionary from the dataset).

    Output: a list of about 10 paragraph records, each a dictionary:
        paragraph_id, title, text, is_supporting
    """

    # HotpotQA stores supporting facts as (title, sentence_id) pairs.
    # We only need the titles: a paragraph is "supporting" if its title
    # appears anywhere in the supporting facts list.
    supporting_titles = set(example["supporting_facts"]["title"])

    # The context is stored as two parallel lists:
    #   context["title"]     -> list of article titles (one per paragraph)
    #   context["sentences"] -> list of sentence lists (one per paragraph)
    titles = example["context"]["title"]
    sentences_per_paragraph = example["context"]["sentences"]

    paragraphs = []
    for i, (title, sentences) in enumerate(zip(titles, sentences_per_paragraph)):
        # Join the paragraph's sentences into one plain text string.
        text = " ".join(sentences)

        paragraphs.append({
            "paragraph_id": f"P{i}",
            "title": title,
            "text": text,
            "is_supporting": title in supporting_titles,
        })

    return paragraphs


# ---------------------------------------------------------------------------
# Small test: run "python -m src.data" to check this file works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load the subset and print a quick summary of the first question.
    questions = load_questions()

    first = questions[0]
    paragraphs = build_paragraphs(first)

    print("\n--- Example question ---")
    print("Question:", first["question"])
    print("Answer:", first["answer"])
    print("Type:", first["type"], "| Level:", first["level"])
    print(f"Paragraphs: {len(paragraphs)} total, "
          f"{sum(p['is_supporting'] for p in paragraphs)} supporting")
    for p in paragraphs:
        marker = "SUPPORTING" if p["is_supporting"] else "distractor"
        print(f"  {p['paragraph_id']}: [{marker}] {p['title']}")
