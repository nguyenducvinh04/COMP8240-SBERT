from pathlib import Path
from collections import defaultdict
from urllib.parse import quote
import json
import random
import re

import nltk
import pandas as pd
import wikipediaapi
from nltk.tokenize import sent_tokenize


# =========================================================
# Configuration
# =========================================================

RANDOM_SEED = 8240

TRIPLETS_PER_ARTICLE = 50

MIN_SENTENCE_WORDS = 7
MIN_SENTENCE_CHARS = 40
MAX_SENTENCE_CHARS = 400

MANUAL_REVIEW_SIZE = 50

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

rng = random.Random(RANDOM_SEED)


# =========================================================
# Australian Wikipedia article selection
# =========================================================

ARTICLE_GROUPS = {

    "Cities": [
        "Sydney",
        "Melbourne",
        "Brisbane",
        "Perth",
    ],

    "Universities": [
        "Macquarie University",
        "University of Sydney",
        "University of Melbourne",
        "Australian National University",
    ],

    "Landmarks": [
        "Sydney Opera House",
        "Sydney Harbour Bridge",
        "Uluru",
        "Great Barrier Reef",
    ],

    "Sport": [
        "Australian rules football",
        "Cricket in Australia",
        "Rugby league in Australia",
        "Tennis in Australia",
    ],

    "Wildlife": [
        "Kangaroo",
        "Koala",
        "Platypus",
        "Tasmanian devil",
    ],

    "History": [
        "History of Australia",
        "Federation of Australia",
        "Australian gold rushes",
        "Eureka Rebellion",
    ],
}


# =========================================================
# NLTK sentence tokenizer
# =========================================================

print("=" * 72)
print("AUSTRALIAN WIKIPEDIA DATASET CONSTRUCTION")
print("=" * 72)

print("\nChecking sentence tokenizer...")

try:
    sent_tokenize("This is a test sentence.")
except LookupError:

    print("Downloading NLTK tokenizer resources...")

    nltk.download(
        "punkt",
        quiet=True
    )

    nltk.download(
        "punkt_tab",
        quiet=True
    )

print("Sentence tokenizer ready.")


# =========================================================
# Wikipedia client
# =========================================================

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent=(
        "COMP8240-SBERT/1.0 "
        "(https://github.com/nguyenducvinh04/COMP8240-SBERT)"
    ),
)


# =========================================================
# Cleaning functions
# =========================================================

EXCLUDED_SECTIONS = {
    "references",
    "reference",
    "external links",
    "external link",
    "see also",
    "further reading",
    "bibliography",
    "sources",
    "notes",
    "citations",
    "gallery",
}


def clean_text(text):
    """
    Perform lightweight cleaning while preserving
    natural sentence content.
    """

    text = str(text)

    # Remove citation-style markers such as [1], [24], etc.
    text = re.sub(
        r"\[\d+\]",
        " ",
        text,
    )

    # Remove excessive whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def valid_sentence(sentence):
    """
    Filter sentences that are too short, too long,
    or otherwise unsuitable for the experiment.
    """

    sentence = sentence.strip()

    if len(sentence) < MIN_SENTENCE_CHARS:
        return False

    if len(sentence) > MAX_SENTENCE_CHARS:
        return False

    words = sentence.split()

    if len(words) < MIN_SENTENCE_WORDS:
        return False

    # Avoid obvious URLs
    if "http://" in sentence or "https://" in sentence:
        return False

    # Require at least some alphabetic content
    alpha_count = sum(
        character.isalpha()
        for character in sentence
    )

    if alpha_count < 20:
        return False

    return True


def split_into_sentences(text):

    cleaned = clean_text(text)

    sentences = sent_tokenize(cleaned)

    sentences = [
        sentence.strip()
        for sentence in sentences
        if valid_sentence(sentence)
    ]

    return sentences


# =========================================================
# Recursive section extraction
# =========================================================

def extract_sections(
    sections,
    parent_path="",
):
    """
    Recursively extract Wikipedia section text while
    retaining hierarchical section paths.
    """

    extracted = []

    for section in sections:

        title = section.title.strip()

        if title.lower() in EXCLUDED_SECTIONS:
            continue

        if parent_path:
            section_path = (
                parent_path
                + " > "
                + title
            )
        else:
            section_path = title

        sentences = split_into_sentences(
            section.text
        )

        if len(sentences) >= 2:

            extracted.append(
                {
                    "section_title": title,
                    "section_path": section_path,
                    "sentences": sentences,
                }
            )

        # Recursively process subsections
        extracted.extend(
            extract_sections(
                section.sections,
                section_path,
            )
        )

    return extracted


# =========================================================
# Collect Wikipedia data
# =========================================================

article_records = []
section_records = []
sentence_records = []

article_section_sentences = defaultdict(dict)

processed_articles = []

article_counter = 0


print("\n" + "=" * 72)
print("COLLECTING ARTICLES")
print("=" * 72)


for category, article_titles in ARTICLE_GROUPS.items():

    for article_title in article_titles:

        article_counter += 1

        print(
            f"\n[{article_counter:02d}] "
            f"{category}: {article_title}"
        )

        page = wiki.page(article_title)

        if not page.exists():

            print("  WARNING: page not found.")

            article_records.append(
                {
                    "category": category,
                    "article": article_title,
                    "found": False,
                    "url": "",
                    "sections_kept": 0,
                    "sentences_kept": 0,
                }
            )

            continue

        url = (
            "https://en.wikipedia.org/wiki/"
            + quote(
                page.title.replace(" ", "_")
            )
        )

        sections = extract_sections(
            page.sections
        )

        print(
            f"  Sections retained: {len(sections)}"
        )

        section_count = 0
        sentence_count = 0

        processed_article = {
            "category": category,
            "article": page.title,
            "url": url,
            "sections": [],
        }

        for section_index, section in enumerate(
            sections,
            start=1,
        ):

            section_count += 1

            article_slug = re.sub(
                r"[^A-Za-z0-9]+",
                "_",
                page.title,
            ).strip("_")

            section_id = (
                f"{article_slug}"
                f"_SEC_{section_index:03d}"
            )

            sentences = section["sentences"]

            section_records.append(
                {
                    "category": category,
                    "article": page.title,
                    "article_url": url,
                    "section_id": section_id,
                    "section_title": section[
                        "section_title"
                    ],
                    "section_path": section[
                        "section_path"
                    ],
                    "sentence_count": len(sentences),
                }
            )

            section_sentence_records = []

            for sentence_index, sentence in enumerate(
                sentences,
                start=1,
            ):

                sentence_count += 1

                sentence_id = (
                    f"{section_id}"
                    f"_SENT_{sentence_index:03d}"
                )

                record = {
                    "category": category,
                    "article": page.title,
                    "article_url": url,
                    "section_id": section_id,
                    "section_title": section[
                        "section_title"
                    ],
                    "section_path": section[
                        "section_path"
                    ],
                    "sentence_id": sentence_id,
                    "sentence": sentence,
                }

                sentence_records.append(
                    record
                )

                section_sentence_records.append(
                    record
                )

            article_section_sentences[
                page.title
            ][section_id] = (
                section_sentence_records
            )

            processed_article[
                "sections"
            ].append(
                {
                    "section_id": section_id,
                    "section_title": section[
                        "section_title"
                    ],
                    "section_path": section[
                        "section_path"
                    ],
                    "sentences": sentences,
                }
            )

        article_records.append(
            {
                "category": category,
                "article": page.title,
                "found": True,
                "url": url,
                "sections_kept": section_count,
                "sentences_kept": sentence_count,
            }
        )

        processed_articles.append(
            processed_article
        )

        print(
            f"  Sentences retained: {sentence_count}"
        )


# =========================================================
# Generate triplets
# =========================================================

print("\n" + "=" * 72)
print("GENERATING TRIPLETS")
print("=" * 72)


triplet_records = []

article_triplet_counts = defaultdict(int)

triplet_id_counter = 0


for article, sections in article_section_sentences.items():

    section_ids = list(
        sections.keys()
    )

    # Need at least two sections so that
    # negatives can come from another section.
    if len(section_ids) < 2:
        continue

    valid_positive_sections = [
        section_id
        for section_id in section_ids
        if len(sections[section_id]) >= 2
    ]

    if not valid_positive_sections:
        continue

    seen_triplets = set()

    attempts = 0
    max_attempts = (
        TRIPLETS_PER_ARTICLE * 50
    )

    while (
        article_triplet_counts[article]
        < TRIPLETS_PER_ARTICLE
        and attempts < max_attempts
    ):

        attempts += 1

        positive_section_id = rng.choice(
            valid_positive_sections
        )

        positive_section_sentences = (
            sections[positive_section_id]
        )

        anchor_record, positive_record = (
            rng.sample(
                positive_section_sentences,
                2,
            )
        )

        negative_section_options = [
            section_id
            for section_id in section_ids
            if section_id
            != positive_section_id
            and len(sections[section_id]) > 0
        ]

        if not negative_section_options:
            continue

        negative_section_id = rng.choice(
            negative_section_options
        )

        negative_record = rng.choice(
            sections[negative_section_id]
        )

        triplet_key = (
            anchor_record["sentence_id"],
            positive_record["sentence_id"],
            negative_record["sentence_id"],
        )

        if triplet_key in seen_triplets:
            continue

        seen_triplets.add(
            triplet_key
        )

        triplet_id_counter += 1

        triplet_id = (
            f"TRIPLET_{triplet_id_counter:05d}"
        )

        triplet_records.append(
            {
                "triplet_id": triplet_id,

                "category":
                    anchor_record["category"],

                "article":
                    article,

                "article_url":
                    anchor_record["article_url"],

                "anchor_section":
                    anchor_record["section_path"],

                "positive_section":
                    positive_record["section_path"],

                "negative_section":
                    negative_record["section_path"],

                "anchor_sentence_id":
                    anchor_record["sentence_id"],

                "positive_sentence_id":
                    positive_record["sentence_id"],

                "negative_sentence_id":
                    negative_record["sentence_id"],

                "anchor":
                    anchor_record["sentence"],

                "positive":
                    positive_record["sentence"],

                "negative":
                    negative_record["sentence"],
            }
        )

        article_triplet_counts[
            article
        ] += 1


# =========================================================
# Create DataFrames
# =========================================================

articles_df = pd.DataFrame(
    article_records
)

sections_df = pd.DataFrame(
    section_records
)

sentences_df = pd.DataFrame(
    sentence_records
)

triplets_df = pd.DataFrame(
    triplet_records
)


# =========================================================
# Add triplet counts to article statistics
# =========================================================

if not articles_df.empty:

    articles_df[
        "triplets_generated"
    ] = articles_df["article"].map(
        article_triplet_counts
    ).fillna(0).astype(int)


# =========================================================
# Manual review sample
# =========================================================

if len(triplets_df) > 0:

    review_size = min(
        MANUAL_REVIEW_SIZE,
        len(triplets_df),
    )

    manual_review_df = (
        triplets_df.sample(
            n=review_size,
            random_state=RANDOM_SEED,
        )
        .copy()
    )

    manual_review_df[
        "valid_triplet"
    ] = ""

    manual_review_df[
        "review_notes"
    ] = ""

else:

    manual_review_df = pd.DataFrame()


# =========================================================
# Save files
# =========================================================

articles_file = (
    DATA_DIR / "article_statistics.csv"
)

sections_file = (
    DATA_DIR / "sections.csv"
)

sentences_file = (
    DATA_DIR / "sentences.csv"
)

triplets_file = (
    DATA_DIR
    / "australian_wikipedia_triplets.csv"
)

processed_json_file = (
    DATA_DIR
    / "processed_wikipedia_sections.json"
)

manual_review_file = (
    DATA_DIR
    / "manual_review_sample.csv"
)


articles_df.to_csv(
    articles_file,
    index=False,
)

sections_df.to_csv(
    sections_file,
    index=False,
)

sentences_df.to_csv(
    sentences_file,
    index=False,
)

triplets_df.to_csv(
    triplets_file,
    index=False,
)

manual_review_df.to_csv(
    manual_review_file,
    index=False,
)


with open(
    processed_json_file,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        processed_articles,
        file,
        indent=2,
        ensure_ascii=False,
    )


# =========================================================
# Dataset statistics
# =========================================================

articles_requested = sum(
    len(titles)
    for titles in ARTICLE_GROUPS.values()
)

articles_found = int(
    articles_df["found"].sum()
) if not articles_df.empty else 0

sections_kept = len(
    sections_df
)

sentences_kept = len(
    sentences_df
)

triplets_generated = len(
    triplets_df
)


stats = {
    "random_seed": RANDOM_SEED,

    "articles_requested":
        articles_requested,

    "articles_found":
        articles_found,

    "sections_kept":
        sections_kept,

    "sentences_kept":
        sentences_kept,

    "triplets_generated":
        triplets_generated,

    "target_triplets_per_article":
        TRIPLETS_PER_ARTICLE,

    "manual_review_sample_size":
        len(manual_review_df),
}


stats_file = (
    RESULTS_DIR / "dataset_statistics.json"
)

with open(
    stats_file,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        stats,
        file,
        indent=2,
    )


# =========================================================
# Category statistics
# =========================================================

if not triplets_df.empty:

    category_stats = (
        triplets_df
        .groupby("category")
        .size()
        .reset_index(
            name="triplets"
        )
    )

else:

    category_stats = pd.DataFrame(
        columns=[
            "category",
            "triplets",
        ]
    )


category_stats_file = (
    RESULTS_DIR
    / "triplets_by_category.csv"
)

category_stats.to_csv(
    category_stats_file,
    index=False,
)


# =========================================================
# Print final summary
# =========================================================

print("\n" + "=" * 72)
print("DATASET CONSTRUCTION COMPLETE")
print("=" * 72)

print(
    f"Articles requested: "
    f"{articles_requested}"
)

print(
    f"Articles found:     "
    f"{articles_found}"
)

print(
    f"Sections retained:  "
    f"{sections_kept}"
)

print(
    f"Sentences retained: "
    f"{sentences_kept}"
)

print(
    f"Triplets generated: "
    f"{triplets_generated}"
)

print(
    f"Manual review set:  "
    f"{len(manual_review_df)}"
)


print("\nTriplets by category:")

if not category_stats.empty:

    print(
        category_stats.to_string(
            index=False
        )
    )


print("\nFiles saved:")

print(articles_file)
print(sections_file)
print(sentences_file)
print(triplets_file)
print(processed_json_file)
print(manual_review_file)
print(stats_file)
print(category_stats_file)


print(
    "\nAUSTRALIAN WIKIPEDIA "
    "DATASET READY"
)