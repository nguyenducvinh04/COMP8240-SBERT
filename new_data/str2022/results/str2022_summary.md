# SBERT Generalisation Evaluation on STR-2022

Model:
sentence-transformers/bert-base-nli-mean-tokens

Evaluation:
Zero-shot evaluation with no STR-2022 fine-tuning.
Cosine similarity between SBERT sentence embeddings was compared
with human semantic-relatedness scores using Spearman correlation.

## Overall Result

Pairs: 5,500
Spearman rho: 0.6965
Spearman x 100: 69.65

## Results by Source

| Source | Pairs | Spearman x 100 |
|---|---:|---:|
| STS | 250 | 77.78 |
| Formality | 1000 | 75.93 |
| SNLI | 750 | 72.30 |
| ParaNMT | 750 | 70.70 |
| Wikipedia | 1000 | 60.61 |
| Goodreads | 1000 | 42.89 |
| Stance | 750 | 29.56 |

## Interpretation

SBERT generalises reasonably well overall to STR-2022, but
performance varies substantially across data sources.

Performance remains strong on STS, Formality, SNLI, and ParaNMT,
while it is considerably weaker on Wikipedia and Goodreads and
especially weak on Stance data.

This suggests that the semantic representations learned by the
original SBERT-NLI model generalise better to some types of semantic
relationship than others.