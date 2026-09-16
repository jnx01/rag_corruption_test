"""
make_plots.py — Create the 3 required plots from results.csv.

This is Step 9 from the spec. It reads results/results.csv and produces:

    Plot 1 — Answer quality vs evidence damage (bar chart of mean F1
             per condition). Answers: "How sensitive is answer quality
             to evidence damage?"

    Plot 2 — Retrieval recall vs answer F1 (scatter plot, one point per
             question-condition pair). Answers: "Does retrieval quality
             predict answer quality?"

    Plot 3 — Condition summary table (a table image with mean recall,
             F1, and exact match per condition). This is the most useful
             single artifact for the README.

All figures are saved to results/figures/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Make sure the project root is on the Python path, so "from src...."
# works no matter where we run this file from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Where the results table and output figures live.
RESULTS_FILE = Path("results/results.csv")
FIGURES_DIR = Path("results/figures")

# The 4 conditions in the order we want them shown (least to most damaged).
CONDITION_ORDER = ["clean", "remove_one", "remove_all", "contaminated"]

# Human-readable labels for the axes and tables.
CONDITION_LABELS = {
    "clean": "Clean",
    "remove_one": "Remove 1",
    "remove_all": "Remove All",
    "contaminated": "Contaminated",
}


def load_results():
    """Read results.csv into a pandas DataFrame."""
    return pd.read_csv(RESULTS_FILE)


# ---------------------------------------------------------------------------
# Plot 1 — Answer quality vs evidence damage (bar chart)
# ---------------------------------------------------------------------------

def plot_answer_quality_by_condition(df):
    """Bar chart: mean answer F1 for each of the 4 conditions."""

    # Compute the mean F1 for each condition, in our chosen order.
    mean_f1 = df.groupby("condition")["f1"].mean().reindex(CONDITION_ORDER)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        [CONDITION_LABELS[c] for c in CONDITION_ORDER],
        mean_f1.values,
        color=["#4C9F70", "#F0A202", "#D64550", "#7B2CBF"],
    )

    # Label each bar with its exact value so the chart is readable on its own.
    for bar, value in zip(bars, mean_f1.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01,
                f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Evidence condition")
    ax.set_ylabel("Mean Answer F1")
    ax.set_title("Answer quality vs evidence damage")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out = FIGURES_DIR / "plot1_answer_quality_by_condition.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 2 — Retrieval recall vs answer F1 (scatter plot)
# ---------------------------------------------------------------------------

def plot_recall_vs_f1(df):
    """Scatter plot: supporting-evidence recall (x) vs answer F1 (y).

    Each point is one question-condition pair. We only plot rows where
    recall is defined (clean and remove_one) — recall is not meaningful
    when no supporting paragraphs exist (remove_all, contaminated).
    """

    # Keep only rows with a real recall value.
    plot_df = df.dropna(subset=["retrieval_recall"])

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Color points by condition so we can see if the two conditions
    # behave differently.
    for condition in ["clean", "remove_one"]:
        subset = plot_df[plot_df["condition"] == condition]
        ax.scatter(subset["retrieval_recall"], subset["f1"],
                   label=CONDITION_LABELS[condition], alpha=0.6, s=40)

    ax.set_xlabel("Supporting evidence recall")
    ax.set_ylabel("Answer F1")
    ax.set_title("Retrieval recall vs answer F1")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(title="Condition")

    fig.tight_layout()
    out = FIGURES_DIR / "plot2_recall_vs_f1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 3 — Condition summary table (as an image)
# ---------------------------------------------------------------------------

def plot_summary_table(df):
    """A table image: mean recall, F1, and exact match per condition."""

    # Compute the three means per condition, in our chosen order.
    summary = df.groupby("condition").agg(
        evidence_recall=("retrieval_recall", "mean"),
        answer_f1=("f1", "mean"),
        exact_match=("exact_match", "mean"),
    ).reindex(CONDITION_ORDER)

    # Build the table text (recall is blank where it is undefined).
    cell_text = []
    for condition in CONDITION_ORDER:
        row = summary.loc[condition]
        recall_str = f"{row['evidence_recall']:.3f}" if pd.notna(row["evidence_recall"]) else "n/a"
        cell_text.append([
            CONDITION_LABELS[condition],
            recall_str,
            f"{row['answer_f1']:.3f}",
            f"{row['exact_match']:.3f}",
        ])

    columns = ["Condition", "Evidence Recall", "Answer F1", "Exact Match"]

    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.axis("off")  # hide the axes; we only want the table

    table = ax.table(cellText=cell_text, colLabels=columns,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Bold the header row.
    for j in range(len(columns)):
        table[0, j].set_text_props(weight="bold")

    ax.set_title("Condition summary", pad=12)

    fig.tight_layout()
    out = FIGURES_DIR / "plot3_condition_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # Also print the table to the terminal so we can copy it into the README.
    print("\nSummary table (also saved as image):")
    print(summary.round(3).to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    print(f"Loaded {len(df)} rows from {RESULTS_FILE}\n")

    plot_answer_quality_by_condition(df)
    plot_recall_vs_f1(df)
    plot_summary_table(df)

    print("\nAll 3 plots saved to", FIGURES_DIR)
