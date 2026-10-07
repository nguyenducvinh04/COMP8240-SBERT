from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "sentence-transformers/bert-base-nli-mean-tokens"

DATASETS = {
    "STS12": {
        "hf_name": "mteb/sts12-sts",
        "paper_score": 70.97,
    },
    "STS13": {
        "hf_name": "mteb/sts13-sts",
        "paper_score": 76.53,
    },
    "STS14": {
        "hf_name": "mteb/sts14-sts",
        "paper_score": 73.19,
    },
    "STS15": {
        "hf_name": "mteb/sts15-sts",
        "paper_score": 79.09,
    },
    "STS16": {
        "hf_name": "mteb/sts16-sts",
        "paper_score": 74.30,
    },
    "STSb": {
        "hf_name": "mteb/stsbenchmark-sts",
        "paper_score": 77.03,
    },
    "SICK-R": {
        "hf_name": "mteb/sickr-sts",
        "paper_score": 72.91,
    },
}

OUTPUT_DIR = Path("replication/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Environment information
# ---------------------------------------------------------

print("=" * 70)
print("SBERT ORIGINAL STS REPLICATION")
print("=" * 70)

print(f"Model: {MODEL_NAME}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("\nLoading model...")

model = SentenceTransformer(
    MODEL_NAME,
    device=device,
)

print("Model loaded successfully.")


# ---------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------

def evaluate_dataset(dataset_name, hf_name, paper_score):
    print("\n" + "=" * 70)
    print(f"Evaluating {dataset_name}")
    print("=" * 70)

    print(f"Downloading/loading: {hf_name}")

    dataset = load_dataset(
        hf_name,
        split="test",
    )

    print(f"Number of sentence pairs: {len(dataset)}")

    sentence1 = dataset["sentence1"]
    sentence2 = dataset["sentence2"]

    gold_scores = np.asarray(
        dataset["score"],
        dtype=np.float64,
    )

    print("Encoding sentence 1...")

    embeddings1 = model.encode(
        sentence1,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("Encoding sentence 2...")

    embeddings2 = model.encode(
        sentence2,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Because embeddings are normalised,
    # their row-wise dot product is cosine similarity.
    cosine_scores = np.sum(
        embeddings1 * embeddings2,
        axis=1,
    )

    correlation = spearmanr(
        gold_scores,
        cosine_scores,
    ).statistic

    reproduced_score = correlation * 100

    difference = reproduced_score - paper_score

    print(f"\nPaper score:      {paper_score:.2f}")
    print(f"Reproduced score: {reproduced_score:.2f}")
    print(f"Difference:       {difference:+.2f}")

    return {
        "Dataset": dataset_name,
        "Pairs": len(dataset),
        "Paper": paper_score,
        "Reproduced": reproduced_score,
        "Difference": difference,
    }


# ---------------------------------------------------------
# Run all original STS evaluations
# ---------------------------------------------------------

results = []

for dataset_name, config in DATASETS.items():
    try:
        result = evaluate_dataset(
            dataset_name=dataset_name,
            hf_name=config["hf_name"],
            paper_score=config["paper_score"],
        )

        results.append(result)

    except Exception as error:
        print(f"\nERROR evaluating {dataset_name}")
        print(error)

        results.append({
            "Dataset": dataset_name,
            "Pairs": None,
            "Paper": config["paper_score"],
            "Reproduced": None,
            "Difference": None,
        })


# ---------------------------------------------------------
# Results table
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

valid_results = results_df.dropna(
    subset=["Reproduced"]
)

paper_average = valid_results["Paper"].mean()
reproduced_average = valid_results["Reproduced"].mean()

average_row = pd.DataFrame([
    {
        "Dataset": "Average",
        "Pairs": valid_results["Pairs"].sum(),
        "Paper": paper_average,
        "Reproduced": reproduced_average,
        "Difference": reproduced_average - paper_average,
    }
])

results_df = pd.concat(
    [results_df, average_row],
    ignore_index=True,
)


print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

output_file = OUTPUT_DIR / "sts_results.csv"

results_df.to_csv(
    output_file,
    index=False,
)

print(f"\nResults saved to: {output_file}")

print("\nOriginal paper SBERT-NLI-base average: 74.89")
print(
    f"Our reproduced average: "
    f"{reproduced_average:.2f}"
)

print("\nREPLICATION COMPLETE")