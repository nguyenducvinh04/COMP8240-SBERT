from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

FILE = (
    BASE_DIR
    / "data"
    / "manual_review_sample.csv"
)

df = pd.read_csv(FILE)

reviewed = df[
    df["valid_triplet"].notna()
].copy()

reviewed["valid_triplet"] = (
    reviewed["valid_triplet"]
    .astype(int)
)

total = len(reviewed)
valid = int(
    reviewed["valid_triplet"].sum()
)
invalid = total - valid

valid_rate = (
    valid / total
    if total > 0
    else 0
)

print("=" * 70)
print("MANUAL REVIEW SUMMARY")
print("=" * 70)

print(f"Reviewed triplets: {total}")
print(f"Valid triplets:    {valid}")
print(f"Invalid triplets:  {invalid}")
print(
    f"Estimated label validity: "
    f"{valid_rate * 100:.2f}%"
)

print("\nInvalid examples:")

invalid_df = reviewed[
    reviewed["valid_triplet"] == 0
]

for _, row in invalid_df.iterrows():

    print("\n" + "-" * 70)
    print(f"Article: {row['article']}")
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

    print("\nAnchor:")
    print(row["anchor"])

    print("\nPositive:")
    print(row["positive"])

    print("\nNegative:")
    print(row["negative"])

    print("\nNotes:")
    print(row["review_notes"])