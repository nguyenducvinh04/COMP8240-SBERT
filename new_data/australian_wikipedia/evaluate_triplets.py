from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "sentence-transformers/bert-base-nli-mean-tokens"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRIPLETS_FILE = (
    DATA_DIR / "australian_wikipedia_triplets.csv"
)


# =========================================================
# Environment
# =========================================================

print("=" * 72)
print("SBERT EVALUATION ON AUSTRALIAN WIKIPEDIA TRIPLETS")
print("=" * 72)

print(f"Model: {MODEL_NAME}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# =========================================================
# Load data
# =========================================================

df = pd.read_csv(TRIPLETS_FILE)

print(f"\nTriplets loaded: {len(df)}")

required_columns = [
    "triplet_id",
    "category",
    "article",
    "anchor",
    "positive",
    "negative",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# =========================================================
# Load model
# =========================================================

print("\nLoading SBERT...")

model = SentenceTransformer(
    MODEL_NAME,
    device=device,
)

print("Model loaded successfully.")


# =========================================================
# Encode unique sentences
# =========================================================

all_sentences = pd.concat(
    [
        df["anchor"],
        df["positive"],
        df["negative"],
    ],
    ignore_index=True,
)

unique_sentences = all_sentences.drop_duplicates().tolist()

print(
    f"\nUnique sentences to encode: "
    f"{len(unique_sentences)}"
)

embeddings = model.encode(
    unique_sentences,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

embedding_lookup = {
    sentence: embedding
    for sentence, embedding in zip(
        unique_sentences,
        embeddings,
    )
}


# =========================================================
# Compute similarities
# =========================================================

positive_scores = []
negative_scores = []
margins = []
correct_values = []


for _, row in df.iterrows():

    anchor_embedding = embedding_lookup[
        row["anchor"]
    ]

    positive_embedding = embedding_lookup[
        row["positive"]
    ]

    negative_embedding = embedding_lookup[
        row["negative"]
    ]

    # Embeddings are normalized, so dot product
    # equals cosine similarity.

    sim_positive = float(
        np.dot(
            anchor_embedding,
            positive_embedding,
        )
    )

    sim_negative = float(
        np.dot(
            anchor_embedding,
            negative_embedding,
        )
    )

    margin = (
        sim_positive
        - sim_negative
    )

    correct = int(
        sim_positive > sim_negative
    )

    positive_scores.append(
        sim_positive
    )

    negative_scores.append(
        sim_negative
    )

    margins.append(
        margin
    )

    correct_values.append(
        correct
    )


df["similarity_anchor_positive"] = positive_scores
df["similarity_anchor_negative"] = negative_scores
df["margin"] = margins
df["correct"] = correct_values


# =========================================================
# Overall results
# =========================================================

overall_accuracy = df["correct"].mean()

mean_positive = (
    df["similarity_anchor_positive"].mean()
)

mean_negative = (
    df["similarity_anchor_negative"].mean()
)

mean_margin = df["margin"].mean()


print("\n" + "=" * 72)
print("OVERALL RESULT")
print("=" * 72)

print(
    f"Triplets: {len(df)}"
)

print(
    f"Correct: "
    f"{df['correct'].sum()}"
)

print(
    f"Incorrect: "
    f"{(1 - df['correct']).sum()}"
)

print(
    f"Triplet accuracy: "
    f"{overall_accuracy:.4f}"
)

print(
    f"Triplet accuracy x 100: "
    f"{overall_accuracy * 100:.2f}%"
)

print(
    f"Mean anchor-positive similarity: "
    f"{mean_positive:.4f}"
)

print(
    f"Mean anchor-negative similarity: "
    f"{mean_negative:.4f}"
)

print(
    f"Mean similarity margin: "
    f"{mean_margin:.4f}"
)


# =========================================================
# Results by category
# =========================================================

category_results = []

for category, group in df.groupby("category"):

    accuracy = group["correct"].mean()

    category_results.append(
        {
            "Category": category,
            "Triplets": len(group),
            "Correct": int(
                group["correct"].sum()
            ),
            "Accuracy": accuracy,
            "Accuracy_percent":
                accuracy * 100,
            "Mean_positive_similarity":
                group[
                    "similarity_anchor_positive"
                ].mean(),
            "Mean_negative_similarity":
                group[
                    "similarity_anchor_negative"
                ].mean(),
            "Mean_margin":
                group["margin"].mean(),
        }
    )


category_df = pd.DataFrame(
    category_results
).sort_values(
    "Accuracy",
    ascending=False,
)


print("\n" + "=" * 72)
print("RESULT BY CATEGORY")
print("=" * 72)

print(
    category_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# =========================================================
# Results by article
# =========================================================

article_results = []

for (category, article), group in df.groupby(
    ["category", "article"]
):

    accuracy = group["correct"].mean()

    article_results.append(
        {
            "Category": category,
            "Article": article,
            "Triplets": len(group),
            "Correct": int(
                group["correct"].sum()
            ),
            "Accuracy": accuracy,
            "Accuracy_percent":
                accuracy * 100,
            "Mean_margin":
                group["margin"].mean(),
        }
    )


article_df = pd.DataFrame(
    article_results
).sort_values(
    "Accuracy",
    ascending=False,
)


print("\n" + "=" * 72)
print("RESULT BY ARTICLE")
print("=" * 72)

print(
    article_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# =========================================================
# Failure analysis
# =========================================================

failures_df = (
    df[df["correct"] == 0]
    .copy()
    .sort_values(
        "margin",
        ascending=True,
    )
)


print("\n" + "=" * 72)
print("FAILURE ANALYSIS")
print("=" * 72)

print(
    f"Failed triplets: "
    f"{len(failures_df)}"
)

print("\nFive strongest failures:")


for _, row in failures_df.head(5).iterrows():

    print("\n" + "-" * 72)

    print(
        f"Category: {row['category']}"
    )

    print(
        f"Article: {row['article']}"
    )

    print(
        f"Positive similarity: "
        f"{row['similarity_anchor_positive']:.4f}"
    )

    print(
        f"Negative similarity: "
        f"{row['similarity_anchor_negative']:.4f}"
    )

    print(
        f"Margin: "
        f"{row['margin']:.4f}"
    )

    print("\nANCHOR:")
    print(row["anchor"])

    print("\nPOSITIVE:")
    print(row["positive"])

    print("\nNEGATIVE:")
    print(row["negative"])


# =========================================================
# Save results
# =========================================================

scored_file = (
    RESULTS_DIR
    / "australian_wikipedia_scored_triplets.csv"
)

category_file = (
    RESULTS_DIR
    / "triplet_accuracy_by_category.csv"
)

article_file = (
    RESULTS_DIR
    / "triplet_accuracy_by_article.csv"
)

failures_file = (
    RESULTS_DIR
    / "triplet_failures.csv"
)

summary_file = (
    RESULTS_DIR
    / "triplet_evaluation_summary.json"
)


df.to_csv(
    scored_file,
    index=False,
)

category_df.to_csv(
    category_file,
    index=False,
)

article_df.to_csv(
    article_file,
    index=False,
)

failures_df.to_csv(
    failures_file,
    index=False,
)


summary = {
    "model": MODEL_NAME,
    "triplets": len(df),
    "correct": int(
        df["correct"].sum()
    ),
    "incorrect": int(
        (1 - df["correct"]).sum()
    ),
    "accuracy": float(
        overall_accuracy
    ),
    "accuracy_percent": float(
        overall_accuracy * 100
    ),
    "mean_positive_similarity": float(
        mean_positive
    ),
    "mean_negative_similarity": float(
        mean_negative
    ),
    "mean_margin": float(
        mean_margin
    ),
}


with open(
    summary_file,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary,
        file,
        indent=2,
    )


print("\n" + "=" * 72)
print("FILES SAVED")
print("=" * 72)

print(scored_file)
print(category_file)
print(article_file)
print(failures_file)
print(summary_file)

print("\nTRIPLET EVALUATION COMPLETE")