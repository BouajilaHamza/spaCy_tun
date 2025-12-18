## Getting started

### Install

```bash
pip install -e .
```

### Optional extras

```bash
pip install -e ".[processing]"
pip install -e ".[training]"
pip install -e ".[docs]"
```

## Core mental model

- **Normalization**: conservative cleanup for noisy text (Arabizi digits), plus optional phonology exemplar tagging (Qaf/Gaf).
- **Feature extraction**: detect a small set of **high-precision** Derja discriminators (future markers, circumfix negation, paradigm shift, possession, discourse particles).
- **Authenticity scoring**: transparent rule-based scoring for filtering, with counts and notes that tell you *why* a line passed/failed.

## Quickstart

### 1) Score a sentence

```python
from tun_linguist import DialectScorer

scorer = DialectScorer()
res = scorer.calculate_authenticity_score("ما-قلت-لها-ش")

print(res.score)
print(res.positives)
print(res.negatives)
print(res.notes)
```

What you’ll typically do in a pipeline:

- Keep lines with `res.is_authentic(threshold=...) == True`
- Store `positives/negatives/notes` for auditing and error analysis

### 2) Extract specific features (for debugging / analytics)

```python
from tun_linguist import VerbAnalyzer, NegationParser

text = "غدوة باش نكتبلك ما ننساش"

verbs = VerbAnalyzer().find_features(text)
neg = NegationParser().find(text)

print(verbs)
print(neg)
```

### 3) Run the dataset scripts

This repo ships a dataset processing pipeline in `scripts/` (see the “Dataset pipeline” guide).
