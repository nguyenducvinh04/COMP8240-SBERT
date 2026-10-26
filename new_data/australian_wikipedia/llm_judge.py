from pathlib import Path
import json
import os
import random
import re
import time

import pandas as pd
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score


# =========================================================
# Configuration
# =========================================================

RANDOM_SEED = 8240
SAMPLE_SIZE = 100

# Current lightweight API model.
# You can override it with:
# export OPENAI_JUDGE_MODEL="another-model-name"
MODEL_NAME = os.getenv(
    "OPENAI_JUDGE_MODEL",
    "gpt-6-luna",
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = (
    RESULTS_DIR
    / "australian_wikipedia_scored_triplets.csv"
)

SAMPLE_FILE = (
    RESULTS_DIR
    / "llm_judge_sample.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "llm_judge_results.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "llm_judge_summary.json"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

rng = random.Random(RANDOM_SEED)


# =========================================================
# Check API key
# =========================================================

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )

client = OpenAI()


# =========================================================
# Load SBERT-scored triplets
# =========================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 72)
print("LLM-AS-JUDGE EVALUATION")
print("=" * 72)

print(f"Model: {MODEL_NAME}")
print(f"Available triplets: {len(df)}")


# =========================================================
# Create fixed stratified sample
# =========================================================

if SAMPLE_FILE.exists():

    print("\nUsing existing fixed sample...")

    sample_df = pd.read_csv(
        SAMPLE_FILE
    )

else:

    print("\nCreating fixed stratified sample...")

    # 100 examples:
    # Cities        17
    # History       17
    # Landmarks     17
    # Sport         17
    # Universities  16
    # Wildlife      16

    quotas = {
        "Cities": 17,
        "History": 17,
        "Landmarks": 17,
        "Sport": 17,
        "Universities": 16,
        "Wildlife": 16,
    }

    samples = []

    for category, n in quotas.items():

        category_df = df[
            df["category"] == category
        ]

        category_sample = (
            category_df.sample(
                n=n,
                random_state=RANDOM_SEED,
            )
        )

        samples.append(
            category_sample
        )

    sample_df = pd.concat(
        samples,
        ignore_index=True,
    )

    # Shuffle the full sample reproducibly
    sample_df = (
        sample_df.sample(
            frac=1,
            random_state=RANDOM_SEED,
        )
        .reset_index(drop=True)
    )

    sample_df.to_csv(
        SAMPLE_FILE,
        index=False,
    )


print(
    f"Judge sample size: "
    f"{len(sample_df)}"
)


# =========================================================
# Blind candidate ordering
# =========================================================

def get_candidate_order(triplet_id):

    local_rng = random.Random(
        f"{RANDOM_SEED}-{triplet_id}"
    )

    positive_is_a = local_rng.choice(
        [True, False]
    )

    return positive_is_a


# =========================================================
# Prompt
# =========================================================

def build_prompt(
    anchor,
    candidate_a,
    candidate_b,
):

    return f"""
You are an independent semantic-relatedness judge.

Your task is to decide which candidate sentence is more semantically
related to the ANCHOR sentence.

Consider meaning, topic, event, entities, and contextual relationship.
Do not judge grammatical quality.
Do not assume that either candidate must be correct.
Use only the text shown below.

ANCHOR:
{anchor}

CANDIDATE A:
{candidate_a}

CANDIDATE B:
{candidate_b}

Return ONLY one JSON object in exactly this format:

{{
  "choice": "A",
  "confidence": 0.90,
  "reason": "brief explanation"
}}

The value of "choice" must be exactly one of:
"A", "B", or "TIE".

Confidence must be a number between 0 and 1.
Keep the reason under 30 words.
""".strip()


# =========================================================
# Parse response
# =========================================================

def parse_response(text):

    text = text.strip()

    # Remove markdown fences if the model adds them
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    data = json.loads(text)

    choice = (
        str(data["choice"])
        .strip()
        .upper()
    )

    if choice not in {
        "A",
        "B",
        "TIE",
    }:
        raise ValueError(
            f"Invalid choice: {choice}"
        )

    confidence = float(
        data["confidence"]
    )

    reason = str(
        data.get(
            "reason",
            "",
        )
    )

    return (
        choice,
        confidence,
        reason,
    )


# =========================================================
# Resume existing work if present
# =========================================================

existing_results = {}

if OUTPUT_FILE.exists():

    old_df = pd.read_csv(
        OUTPUT_FILE
    )

    for _, row in old_df.iterrows():

        existing_results[
            row["triplet_id"]
        ] = row.to_dict()

    print(
        f"\nResuming: "
        f"{len(existing_results)} "
        f"judgements already saved."
    )


# =========================================================
# Run LLM judge
# =========================================================

result_rows = []


for index, row in sample_df.iterrows():

    triplet_id = row[
        "triplet_id"
    ]

    if triplet_id in existing_results:

        result_rows.append(
            existing_results[
                triplet_id
            ]
        )

        continue

    positive_is_a = (
        get_candidate_order(
            triplet_id
        )
    )

    if positive_is_a:

        candidate_a = row[
            "positive"
        ]

        candidate_b = row[
            "negative"
        ]

        a_label = "POSITIVE"
        b_label = "NEGATIVE"

    else:

        candidate_a = row[
            "negative"
        ]

        candidate_b = row[
            "positive"
        ]

        a_label = "NEGATIVE"
        b_label = "POSITIVE"


    prompt = build_prompt(
        row["anchor"],
        candidate_a,
        candidate_b,
    )


    print(
        f"\n[{index + 1:03d}/"
        f"{len(sample_df):03d}] "
        f"{triplet_id}"
    )


    # Retry a small number of times if parsing/API fails
    last_error = None

    for attempt in range(3):

        try:

            response = (
                client.responses.create(
                    model=MODEL_NAME,
                    input=prompt,
                )
            )

            raw_output = (
                response.output_text
            )

            (
                choice,
                confidence,
                reason,
            ) = parse_response(
                raw_output
            )

            break

        except Exception as error:

            last_error = error

            print(
                f"Attempt "
                f"{attempt + 1} failed: "
                f"{error}"
            )

            time.sleep(2)

    else:

        raise RuntimeError(
            f"Failed to judge "
            f"{triplet_id}: "
            f"{last_error}"
        )


    # Convert blind A/B decision back into
    # POSITIVE / NEGATIVE / TIE.

    if choice == "A":

        llm_decision = a_label

    elif choice == "B":

        llm_decision = b_label

    else:

        llm_decision = "TIE"


    # SBERT's decision:
    # correct = 1 means it ranked positive higher.
    sbert_decision = (
        "POSITIVE"
        if int(row["correct"]) == 1
        else "NEGATIVE"
    )


    weak_label_agreement = int(
        llm_decision == "POSITIVE"
    )

    sbert_agreement = int(
        llm_decision
        == sbert_decision
    )


    result = {

        "triplet_id":
            triplet_id,

        "category":
            row["category"],

        "article":
            row["article"],

        "anchor":
            row["anchor"],

        "candidate_a":
            candidate_a,

        "candidate_b":
            candidate_b,

        "candidate_a_true_label":
            a_label,

        "candidate_b_true_label":
            b_label,

        "llm_choice_blind":
            choice,

        "llm_decision":
            llm_decision,

        "llm_confidence":
            confidence,

        "llm_reason":
            reason,

        "sbert_decision":
            sbert_decision,

        "sbert_positive_similarity":
            row[
                "similarity_anchor_positive"
            ],

        "sbert_negative_similarity":
            row[
                "similarity_anchor_negative"
            ],

        "sbert_margin":
            row["margin"],

        "llm_agrees_weak_label":
            weak_label_agreement,

        "llm_agrees_sbert":
            sbert_agreement,
    }


    result_rows.append(
        result
    )


    # Save after each response so the experiment
    # can safely resume if interrupted.

    pd.DataFrame(
        result_rows
    ).to_csv(
        OUTPUT_FILE,
        index=False,
    )


# =========================================================
# Analysis
# =========================================================

results_df = pd.DataFrame(
    result_rows
)

total = len(
    results_df
)

ties = int(
    (
        results_df[
            "llm_decision"
        ]
        == "TIE"
    ).sum()
)

positive_choices = int(
    (
        results_df[
            "llm_decision"
        ]
        == "POSITIVE"
    ).sum()
)

negative_choices = int(
    (
        results_df[
            "llm_decision"
        ]
        == "NEGATIVE"
    ).sum()
)


weak_agreement = (
    results_df[
        "llm_agrees_weak_label"
    ].mean()
)

sbert_agreement = (
    results_df[
        "llm_agrees_sbert"
    ].mean()
)


# =========================================================
# Agreement excluding LLM ties
# =========================================================

decisive_df = results_df[
    results_df[
        "llm_decision"
    ]
    != "TIE"
].copy()


if len(decisive_df) > 0:

    decisive_weak_agreement = (
        decisive_df[
            "llm_agrees_weak_label"
        ].mean()
    )

    decisive_sbert_agreement = (
        decisive_df[
            "llm_agrees_sbert"
        ].mean()
    )

    kappa = cohen_kappa_score(
        decisive_df[
            "llm_decision"
        ],
        decisive_df[
            "sbert_decision"
        ],
    )

else:

    decisive_weak_agreement = None
    decisive_sbert_agreement = None
    kappa = None


# =========================================================
# Category results
# =========================================================

category_rows = []

for category, group in (
    results_df.groupby(
        "category"
    )
):

    category_rows.append(
        {
            "category":
                category,

            "n":
                len(group),

            "llm_positive_rate":
                (
                    group[
                        "llm_decision"
                    ]
                    == "POSITIVE"
                ).mean(),

            "llm_sbert_agreement":
                group[
                    "llm_agrees_sbert"
                ].mean(),

            "mean_confidence":
                group[
                    "llm_confidence"
                ].mean(),
        }
    )


category_df = pd.DataFrame(
    category_rows
)


# =========================================================
# Print summary
# =========================================================

print("\n")
print("=" * 72)
print("LLM-AS-JUDGE RESULTS")
print("=" * 72)

print(
    f"Model: {MODEL_NAME}"
)

print(
    f"Judged triplets: {total}"
)

print(
    f"LLM chose labelled positive: "
    f"{positive_choices}"
)

print(
    f"LLM chose labelled negative: "
    f"{negative_choices}"
)

print(
    f"LLM ties: {ties}"
)

print(
    f"\nAgreement with weak labels: "
    f"{weak_agreement * 100:.2f}%"
)

print(
    f"Agreement with SBERT: "
    f"{sbert_agreement * 100:.2f}%"
)


if decisive_weak_agreement is not None:

    print(
        f"\nWeak-label agreement "
        f"(excluding ties): "
        f"{decisive_weak_agreement * 100:.2f}%"
    )

    print(
        f"SBERT agreement "
        f"(excluding ties): "
        f"{decisive_sbert_agreement * 100:.2f}%"
    )

    print(
        f"Cohen's kappa "
        f"(LLM vs SBERT): "
        f"{kappa:.4f}"
    )


print("\nResults by category:")

print(
    category_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# =========================================================
# Disagreement cases
# =========================================================

disagreements = results_df[
    (
        results_df[
            "llm_decision"
        ]
        != results_df[
            "sbert_decision"
        ]
    )
    &
    (
        results_df[
            "llm_decision"
        ]
        != "TIE"
    )
].copy()


print(
    f"\nLLM/SBERT disagreements: "
    f"{len(disagreements)}"
)


# =========================================================
# Save summary
# =========================================================

summary = {

    "judge_model":
        MODEL_NAME,

    "sample_size":
        total,

    "random_seed":
        RANDOM_SEED,

    "positive_choices":
        positive_choices,

    "negative_choices":
        negative_choices,

    "ties":
        ties,

    "weak_label_agreement":
        float(
            weak_agreement
        ),

    "sbert_agreement":
        float(
            sbert_agreement
        ),

    "decisive_weak_label_agreement":
        (
            float(
                decisive_weak_agreement
            )
            if decisive_weak_agreement
            is not None
            else None
        ),

    "decisive_sbert_agreement":
        (
            float(
                decisive_sbert_agreement
            )
            if decisive_sbert_agreement
            is not None
            else None
        ),

    "cohen_kappa_llm_sbert":
        (
            float(kappa)
            if kappa is not None
            else None
        ),
}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary,
        file,
        indent=2,
    )


category_df.to_csv(
    RESULTS_DIR
    / "llm_judge_by_category.csv",
    index=False,
)

disagreements.to_csv(
    RESULTS_DIR
    / "llm_sbert_disagreements.csv",
    index=False,
)


print("\nFiles saved:")
print(SAMPLE_FILE)
print(OUTPUT_FILE)
print(SUMMARY_FILE)

print(
    RESULTS_DIR
    / "llm_judge_by_category.csv"
)

print(
    RESULTS_DIR
    / "llm_sbert_disagreements.csv"
)

print("\nLLM-AS-JUDGE COMPLETE")