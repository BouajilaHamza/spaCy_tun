## tun_linguist

`tun_linguist` is a production-oriented library for **Tunisian Arabic (Derja)** linguistic feature extraction, normalization, and dataset filtering.

It is designed for **high-precision dataset curation**: reward definitive Tunisian markers (e.g., `bash/besh`, `ma…sh`, `mta3`) while penalizing strong MSA constraints (e.g., `sawfa`, `lan/lam/laysa`, `سـ` future prefix).

### What you can do with tun_linguist

- **Filter** mixed Arabic corpora to keep authentic Derja.
- **Score** sentences for “Tunisian-ness” with transparent, debuggable features.
- **Generate** preference pairs (SPO/DPO) and contrastive pairs for training.
- **Normalize** common noisy forms (Arabizi digits; Qaf/Gaf exemplar unification) for better grouping and retrieval.

### Quick example

```python
from tun_linguist import DialectScorer

scorer = DialectScorer()
result = scorer.calculate_authenticity_score("غدوة باش نمشي للسوق")

print(result.score)
print(result.positives)
print(result.negatives)
print(result.notes)
```

### Install

```bash
pip install -e .
```

If you want to build these docs locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```
