from pathlib import Path

import numpy as np
import pandas as pd
import torch

from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "sentence-transformers/bert-base-nli-mean-tokens"

DATASET_REPO = "vkpriya/str-2022"
DATASET_FILE = "sem_text_rel_ranked.csv"

OUTPUT_DIR = Path("new_data/str2022/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Environment
# =========================================================

print("=" * 70)
print("SBERT EVALUATION ON STR-2022")
print("=" * 70)

print(f"Model: {MODEL_NAME}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# =========================================================
# Download STR-2022
# =========================================================

print("\nDownloading/loading STR-2022...")

csv_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename=DATASET_FILE,
    repo_type="dataset",
)

print(f"Dataset file: {csv_path}")


# =========================================================
# Load dataset
# =========================================================

df = pd.read_csv(csv_path)

print(f"\nRows: {len(df)}")
print("Columns:")
print(list(df.columns))

required_columns = {
    "Text",
    "Score",
    "SourceID",
    "SubsetID",
    "PairID",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing expected columns: {missing_columns}"
    )


# =========================================================
# Split sentence pairs
# =========================================================

def split_sentence_pair(text):
    """
    STR-2022 stores the two sentences in one Text field,
    separated by a newline.
    """

    parts = str(text).split("\n", maxsplit=1)

    if len(parts) != 2:
        raise ValueError(
            f"Could not split sentence pair:\n{text}"
        )

    sentence1 = parts[0].strip()
    sentence2 = parts[1].strip()

    return sentence1, sentence2


pairs = df["Text"].apply(split_sentence_pair)

df["Sentence1"] = [
    pair[0] for pair in pairs
]

df["Sentence2"] = [
    pair[1] for pair in pairs
]


print("\nExample pair:")
print("Sentence 1:", df.loc[0, "Sentence1"])
print("Sentence 2:", df.loc[0, "Sentence2"])
print("Human score:", df.loc[0, "Score"])


# =========================================================
# Load SBERT
# =========================================================

print("\nLoading SBERT...")

model = SentenceTransformer(
    MODEL_NAME,
    device=device,
)

print("Model loaded successfully.")


# =========================================================
# Generate embeddings
# =========================================================

print("\nEncoding first sentences...")

embeddings1 = model.encode(
    df["Sentence1"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)


print("\nEncoding second sentences...")

embeddings2 = model.encode(
    df["Sentence2"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)


# =========================================================
# Cosine similarity
# =========================================================

# Embeddings are normalized, so their row-wise dot product
# equals cosine similarity.

cosine_scores = np.sum(
    embeddings1 * embeddings2,
    axis=1,
)

df["SBERT_Cosine"] = cosine_scores


# =========================================================
# Overall Spearman correlation
# =========================================================

gold_scores = df["Score"].astype(float).to_numpy()

overall_rho = spearmanr(
    gold_scores,
    cosine_scores,
).statistic


print("\n" + "=" * 70)
print("OVERALL RESULT")
print("=" * 70)

print(f"Number of pairs: {len(df)}")
print(f"Spearman rho: {overall_rho:.4f}")
print(f"Spearman x 100: {overall_rho * 100:.2f}")


# =========================================================
# Per-source analysis
# =========================================================

print("\n" + "=" * 70)
print("RESULT BY SOURCE")
print("=" * 70)

source_results = []

for source_name, source_df in df.groupby("SourceID"):

    source_rho = spearmanr(
        source_df["Score"].astype(float),
        source_df["SBERT_Cosine"],
    ).statistic

    source_results.append(
        {
            "Source": source_name,
            "Pairs": len(source_df),
            "Spearman": source_rho,
            "Spearman_x100": source_rho * 100,
        }
    )


source_results_df = pd.DataFrame(
    source_results
).sort_values(
    by="Spearman",
    ascending=False,
)


print(
    source_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# =========================================================
# Save predictions
# =========================================================

prediction_columns = [
    "SourceID",
    "SubsetID",
    "PairID",
    "Sentence1",
    "Sentence2",
    "Score",
    "SBERT_Cosine",
]

predictions_file = (
    OUTPUT_DIR / "str2022_predictions.csv"
)

df[prediction_columns].to_csv(
    predictions_file,
    index=False,
)


# =========================================================
# Save summary
# =========================================================

summary_df = pd.DataFrame(
    [
        {
            "Dataset": "STR-2022",
            "Pairs": len(df),
            "Spearman": overall_rho,
            "Spearman_x100": overall_rho * 100,
        }
    ]
)

summary_file = OUTPUT_DIR / "str2022_summary.csv"

summary_df.to_csv(
    summary_file,
    index=False,
)


source_file = (
    OUTPUT_DIR / "str2022_by_source.csv"
)

source_results_df.to_csv(
    source_file,
    index=False,
)


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(predictions_file)
print(summary_file)
print(source_file)

print("\nSTR-2022 EVALUATION COMPLETE")