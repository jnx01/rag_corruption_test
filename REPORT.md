# RAG Evidence Corruption Stress Test — Report

## 1. The question we asked

When a RAG system (a system that looks up documents and then answers questions
based on them) loses some of its evidence, how much worse do its answers get?

Think of it like answering an exam question using a textbook. If someone rips
out the exact pages you need, do you still get the answer right? Do you get it
half right? Or do you fail completely? That is what we measured but for a RAG system.

A second, smaller question: does the quality of the *search step* (finding the
right pages) predict the quality of the *final answer*?

## 2. How we set it up

We used **HotpotQA**, a well-known dataset of questions that require combining
facts from two different Wikipedia paragraphs. Each question comes with 10
paragraphs: 2 that contain the real evidence, and 8 that are unrelated
"distractors."

We picked **80 hard questions** (40 "bridge" questions that chain two facts
together, and 40 "comparison" questions that compare two things).

For each question, we created **4 versions of the evidence**, from least to
most damaged:

| Condition | What we did | Supporting paragraphs left |
|---|---|---|
| **Clean** | Nothing — the original 10 paragraphs | 2 |
| **Remove 1** | Removed one of the 2 supporting paragraphs | 1 |
| **Remove All** | Removed both supporting paragraphs | 0 |
| **Contaminated** | Removed both supporting paragraphs AND added 2 similar-looking paragraphs from other questions | 0 |

Then, for each question and each condition, we ran the same simple pipeline:

1. **Search:** BM25 (a classic keyword-matching algorithm) ranks the available
   paragraphs and keeps the top 4.
2. **Answer:** We give the question plus those 4 paragraphs to an AI model
   (GPT-OSS-120B, a large open model) and ask for just the answer.
3. **Score:** We compare the model's answer to the known correct answer.

That gave us **320 answers** in total (80 questions × 4 conditions).

## 3. How we measured success

Three simple scores:

- **Answer F1** — how much the model's answer overlaps with the correct answer,
  word by word. 1.0 means a perfect match, 0.0 means no overlap. For example,
  if the correct answer is "Honolulu" and the model says "Honolulu County,
  Hawaii," that is a high F1 but not perfect.
- **Exact Match** — 1 if the answer is exactly right, 0 otherwise. Much
  stricter.
- **Evidence Recall** — of the supporting paragraphs that exist, how many did
  the search step actually find? (Only meaningful for Clean and Remove 1,
  because the other two conditions have no supporting paragraphs to find.)

## 4. What we found

Here is the main result. Average scores for each condition:

| Condition | Evidence Recall | Answer F1 | Exact Match |
|---|---:|---:|---:|
| Clean | 0.712 | 0.649 | 0.562 |
| Remove 1 | 0.800 | 0.440 | 0.362 |
| Remove All | n/a | 0.337 | 0.288 |
| Contaminated | n/a | 0.314 | 0.275 |

**The pattern is clear: the more we damaged the evidence, the worse the
answers got.**

- Going from **Clean → Remove 1** caused the biggest single drop. Answer F1
  fell from 0.649 to 0.440. Losing just one of two key paragraphs cut answer
  quality by about a third.
- Going from **Remove 1 → Remove All** dropped it further, to 0.337.
- **Contaminated** (missing evidence + misleading look-alikes) was the worst of
  all, at 0.314. Slightly worse than just missing the evidence.

So our hypothesis was supported: less useful evidence leads to worse answers.

### Does search quality predict answer quality?

Mostly, yes. When the search step found the supporting paragraphs, the model
usually answered well. When it did not, the answer usually suffered. But the
relationship is not perfect: sometimes the model got the right answer even
with incomplete evidence, likely by using knowledge it already had.

### One surprising detail

Evidence recall was actually *higher* in Remove 1 (0.800) than in Clean
(0.712). That sounds backwards, but it makes sense: in Clean, the search has to
find *both* supporting paragraphs to get a perfect score. In Remove 1, there is
only one left, and it is easier for the search to find that single paragraph.
So the search looks "better" even though the answers got worse; a good reminder
that search quality and answer quality are not the same thing.

## 5. A concrete example

Here is one real question from the experiment:

> **Question:** In what year did both Orson Scott Card's novel "Ender's Game"
> and Keri Hulme's "The Bone People" come out?
> **Correct answer:** 1985

- **Clean:** The search found both supporting paragraphs (one about each book).
  The model answered **"1985"** — correct.
- **Remove 1:** We removed the "Ender's Game" paragraph. The model answered
  **"I cannot answer the question based on the provided evidence."** — it gave
  up, because half the information it needed was gone.

This is exactly the kind of failure we set out to measure: the model did not
guess or hallucinate; it simply could not answer because the evidence was
incomplete.

## 6. What this means

For anyone building or relying on RAG systems, the takeaway is simple:

> **The quality of the evidence you feed in directly controls the quality of
> the answers you get out.**

If the retrieval step misses key documents, the answer step cannot fully
recover, even with a strong AI model. And adding misleading look-alike
documents is slightly worse than just missing the documents, which matters for
real-world systems where search results are often noisy.

## 7. Limitations (being honest)

- **Small scale:** 80 questions is enough to see a clear trend, but larger
  studies would give more precise numbers.
- **One model:** We used a single AI model (GPT-OSS-120B). Different models
  might be more or less robust to missing evidence.
- **One dataset:** HotpotQA questions are a specific style (multi-hop Wikipedia
  questions). Results might differ on other kinds of questions.
- **Simple search:** We used BM25 (keyword matching). More advanced search
  methods might find evidence more reliably.
- **Contamination is approximate:** Our "similar distractors" came from other
  questions in the dataset, chosen by keyword similarity. They are plausible
  but not adversarially crafted to fool the model.

## 8. Conclusion

We set out to measure how RAG answer quality changes as supporting evidence is
damaged. The answer, in our experiment: **it degrades steadily and
predictably.** Removing even one of two key paragraphs caused a large drop in
answer quality, and removing both (with or without misleading replacements)
dropped it further. The evidence pipeline is not just a detail; it is the
foundation the whole system stands on.
