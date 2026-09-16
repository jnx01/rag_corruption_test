"""
generate.py — Ask the LLM to answer a question using retrieved evidence.

This is Module 5 from the spec. For each question + condition we:

    1. build a simple prompt: question + the 4 retrieved paragraphs
    2. send it to the Gemini model
    3. get back one short answer

Two practical features the spec requires:

    CACHING: every answer is saved to a JSON file the moment we get it.
             If the experiment is interrupted (rate limits, network, etc.),
             we simply re-run and skip everything already done.

    RETRY:   if the API call fails (e.g. rate limit), we wait a bit and
             try again, with longer waits each time ("exponential backoff").
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Which LLM provider to use: "groq" (default, generous free tier) or
# "gemini" (Google; free tier is only 20 requests/day, which is too small
# for our 320-call experiment). Set LLM_PROVIDER in the .env file.
DEFAULT_PROVIDER = "groq"

# The model used for each provider. The spec asked for gemini-2.5-flash,
# but Google no longer offers it to new users, so Gemini falls back to
# gemini-3.6-flash. For Groq we use the spec's suggested backup model.
MODELS = {
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-3.6-flash",
}

# Where answers are cached. One JSON file maps a unique key
# (provider + model + question_id + condition) to the model's answer.
CACHE_FILE = Path("results/answer_cache.json")

# Retry settings: try up to 5 times, waiting 5s, 10s, 20s, 40s between tries.
MAX_RETRIES = 5
INITIAL_WAIT_SECONDS = 5


# ---------------------------------------------------------------------------
# Prompt building (exactly the simple prompt from the spec)
# ---------------------------------------------------------------------------

def build_prompt(question_text, retrieved_paragraphs):
    """Build the prompt string from the question and retrieved paragraphs.

    The spec's prompt is intentionally plain: give the evidence, ask for
    only the answer, no explanation, no confidence judgment.
    """

    # Join the retrieved paragraphs into one evidence block, with a blank
    # line between paragraphs so the model can tell them apart.
    evidence = "\n\n".join(p["text"] for p in retrieved_paragraphs)

    prompt = f"""Answer the question using only the evidence below.

Question:
{question_text}

Evidence:
{evidence}

Return only the answer, with no explanation."""

    return prompt


# ---------------------------------------------------------------------------
# Provider layer: one function that calls whichever provider is configured
# ---------------------------------------------------------------------------

def _get_provider():
    """Read LLM_PROVIDER from .env (default: groq)."""
    load_dotenv()
    return os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()


def _call_llm(prompt, client, provider):
    """Send one prompt to the configured provider and return the text reply.

    Inputs:
        prompt   — the full prompt string.
        client   — a client for the chosen provider (created by the caller).
        provider — "groq" or "gemini".

    Output:
        The model's reply as a plain string.
    """
    model = MODELS[provider]

    if provider == "groq":
        # Groq uses an OpenAI-style chat interface.
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    if provider == "gemini":
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text.strip()

    raise ValueError(f"Unknown provider: {provider!r}. Use 'groq' or 'gemini'.")


def make_client(provider):
    """Create a client for the given provider, using the key from .env."""
    load_dotenv()

    if provider == "groq":
        from groq import Groq
        return Groq(api_key=os.getenv("GROQ_API_KEY"))

    if provider == "gemini":
        from google import genai
        return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    raise ValueError(f"Unknown provider: {provider!r}. Use 'groq' or 'gemini'.")


# ---------------------------------------------------------------------------
# Cache helpers (load, check, save)
# ---------------------------------------------------------------------------

def _load_cache():
    """Load the answer cache from disk. Returns an empty dict if none yet."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def _save_to_cache(cache_key, answer):
    """Save one answer to the cache file immediately.

    We re-read the file, add the new entry, and write it back. This is a
    little slow but very safe: every completed answer is on disk right away,
    so a crash never loses more than the single call in progress.
    """
    cache = _load_cache()
    cache[cache_key] = answer
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _make_cache_key(provider, question_id, condition):
    """Build the unique cache key for one question+condition+provider.

    The provider and model are part of the key, so answers from different
    models never get mixed up.
    """
    return f"{provider}::{MODELS[provider]}::{question_id}::{condition}"


def get_cached_answer(provider, question_id, condition):
    """Return the cached answer for this question+condition, or None."""
    cache = _load_cache()
    return cache.get(_make_cache_key(provider, question_id, condition))


# ---------------------------------------------------------------------------
# The main function: generate one answer (with caching and retry)
# ---------------------------------------------------------------------------

def generate_answer(question_id, condition, question_text,
                    retrieved_paragraphs, client=None, provider=None):
    """Get the model's answer for one question under one condition.

    Inputs:
        question_id          — unique ID of the question (for caching).
        condition            — one of the 4 condition names (for caching).
        question_text        — the question string.
        retrieved_paragraphs — the top-4 paragraphs from retrieval.
        client               — optional provider client (pass one in when
                               making many calls, so we don't recreate it).
        provider             — "groq" or "gemini"; if None, read from .env.

    Output:
        The model's answer as a short string.
    """

    # Work out which provider we are using.
    if provider is None:
        provider = _get_provider()

    # Step 1: check the cache. If we already have this answer, return it
    # without making any API call. This is what makes the experiment
    # resumable.
    cache_key = _make_cache_key(provider, question_id, condition)
    cached = get_cached_answer(provider, question_id, condition)
    if cached is not None:
        return cached

    # Step 2: build the prompt.
    prompt = build_prompt(question_text, retrieved_paragraphs)

    # Step 3: create a client if the caller did not give us one.
    if client is None:
        client = make_client(provider)

    # Step 4: call the API, retrying with exponential backoff on failure.
    # "Exponential backoff" just means: wait 5s, then 10s, then 20s, ...
    # so we don't hammer the API when it is rate-limiting us.
    wait_seconds = INITIAL_WAIT_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            answer = _call_llm(prompt, client, provider)

            # Success: save to cache immediately and return.
            _save_to_cache(cache_key, answer)
            return answer

        except Exception as error:
            # If this was our last try, give up and raise the error.
            if attempt == MAX_RETRIES:
                raise

            # Otherwise wait and try again.
            print(f"    API call failed (attempt {attempt}/{MAX_RETRIES}): "
                  f"{type(error).__name__}. Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)
            wait_seconds *= 2  # double the wait each time


# ---------------------------------------------------------------------------
# Small test: run "python -m src.generate" to check this file works
# Makes ONE real API call (then a second call that should hit the cache).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data import load_questions, build_paragraphs
    from src.retrieval import retrieve

    provider = _get_provider()
    print(f"Provider: {provider} | Model: {MODELS[provider]}")

    # Take the first question, retrieve top-4 from its clean condition.
    question = load_questions()[0]
    paragraphs = build_paragraphs(question)
    top4 = retrieve(question["question"], paragraphs)

    print("Question:", question["question"])
    print("Reference answer:", question["answer"])
    print()

    # First call: goes to the API.
    print("Making first API call...")
    answer1 = generate_answer(question["id"], "clean", question["question"], top4)
    print("Model answer:", answer1)

    # Second call with the same inputs: should come from the cache
    # (instant, no API call).
    print("\nMaking second call (should be cached)...")
    answer2 = generate_answer(question["id"], "clean", question["question"], top4)
    print("Cached answer:", answer2)
    print("\nCache file:", CACHE_FILE)
