from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

TRIPLETS_FILE = (
    BASE_DIR
    / "data"
    / "australian_wikipedia_triplets.csv"
)


print("=" * 72)
print("AUSTRALIAN WIKIPEDIA TRIPLET VALIDATION")
print("=" * 72)


# =========================================================
# Load dataset
# =========================================================

df = pd.read_csv(TRIPLETS_FILE)

print(f"\nTriplets loaded: {len(df)}")


# =========================================================
# 1. Missing values
# =========================================================

required_columns = [
    "triplet_id",
    "category",
    "article",
    "anchor_section",
    "positive_section",
    "negative_section",
    "anchor",
    "positive",
    "negative",
]

missing_values = (
    df[required_columns]
    .isna()
    .sum()
)

print("\n" + "=" * 72)
print("MISSING VALUES")
print("=" * 72)

print(missing_values)


# =========================================================
# 2. Check weak-supervision rules
# =========================================================

same_positive_section = (
    df["anchor_section"]
    == df["positive_section"]
)

different_negative_section = (
    df["anchor_section"]
    != df["negative_section"]
)


print("\n" + "=" * 72)
print("SECTION RULE CHECK")
print("=" * 72)

print(
    "Anchor and positive in same section:",
    f"{same_positive_section.sum()}/{len(df)}"
)

print(
    "Negative in different section:",
    f"{different_negative_section.sum()}/{len(df)}"
)


# =========================================================
# 3. Duplicate triplets
# =========================================================

duplicate_triplets = df.duplicated(
    subset=[
        "anchor",
        "positive",
        "negative",
    ]
).sum()


print("\n" + "=" * 72)
print("DUPLICATE CHECK")
print("=" * 72)

print(
    f"Exact duplicate triplets: "
    f"{duplicate_triplets}"
)


# =========================================================
# 4. Check identical sentences
# =========================================================

anchor_positive_same = (
    df["anchor"] == df["positive"]
).sum()

anchor_negative_same = (
    df["anchor"] == df["negative"]
).sum()

positive_negative_same = (
    df["positive"] == df["negative"]
).sum()


print("\n" + "=" * 72)
print("IDENTICAL SENTENCE CHECK")
print("=" * 72)

print(
    f"Anchor == positive: "
    f"{anchor_positive_same}"
)

print(
    f"Anchor == negative: "
    f"{anchor_negative_same}"
)

print(
    f"Positive == negative: "
    f"{positive_negative_same}"
)


# =========================================================
# 5. Unique sentence usage
# =========================================================

print("\n" + "=" * 72)
print("SENTENCE USAGE")
print("=" * 72)

print(
    "Unique anchors:",
    df["anchor"].nunique()
)

print(
    "Unique positives:",
    df["positive"].nunique()
)

print(
    "Unique negatives:",
    df["negative"].nunique()
)


# =========================================================
# 6. Distribution by category
# =========================================================

category_counts = (
    df.groupby("category")
    .size()
    .sort_values(ascending=False)
)


print("\n" + "=" * 72)
print("CATEGORY DISTRIBUTION")
print("=" * 72)

print(category_counts)


# =========================================================
# 7. Distribution by article
# =========================================================

article_counts = (
    df.groupby(
        ["category", "article"]
    )
    .size()
)


print("\n" + "=" * 72)
print("ARTICLE DISTRIBUTION")
print("=" * 72)

print(article_counts)


# =========================================================
# 8. Most frequently reused sentences
# =========================================================

all_sentence_usage = pd.concat(
    [
        df["anchor"],
        df["positive"],
        df["negative"],
    ]
).value_counts()


print("\n" + "=" * 72)
print("MOST FREQUENTLY USED SENTENCES")
print("=" * 72)

print(
    all_sentence_usage
    .head(10)
    .to_string()
)


# =========================================================
# 9. Show sample triplets
# =========================================================

print("\n" + "=" * 72)
print("RANDOM SAMPLE")
print("=" * 72)

sample = df.sample(
    n=min(5, len(df)),
    random_state=8240,
)

for _, row in sample.iterrows():

    print("\n" + "-" * 72)

    print(
        f"Category: {row['category']}"
    )

    print(
        f"Article: {row['article']}"
    )

    print(
        f"Anchor section: "
        f"{row['anchor_section']}"
    )

    print(
        f"Positive section: "
        f"{row['positive_section']}"
    )

    print(
        f"Negative section: "
        f"{row['negative_section']}"
    )

    print("\nANCHOR:")
    print(row["anchor"])

    print("\nPOSITIVE:")
    print(row["positive"])

    print("\nNEGATIVE:")
    print(row["negative"])


print("\n" + "=" * 72)
print("VALIDATION COMPLETE")
print("=" * 72)