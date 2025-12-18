## Installation

### Library only

```bash
pip install -e .
```

### Dataset processing / training extras

```bash
pip install -e ".[processing]"
pip install -e ".[training]"
```

## Core concepts

- **Normalization**: lightweight cleanup of noisy user text (e.g., Arabizi digits) without attempting full orthographic reconstruction.
- **Feature extraction**: detect high-precision Tunisian grammar markers (future, negation, paradigm shift, possession).
- **Authenticity scoring**: transparent rule-based scoring for filtering and diagnostics.

## Quick start

### Score a sentence

```python
from tun_linguist import DialectScorer

scorer = DialectScorer()
res = scorer.calculate_authenticity_score("ما-قلت-لها-ش")

print(res.score)
print(res.positives)
print(res.negatives)
print(res.notes)
```

### Extract specific features

```python
from tun_linguist import VerbAnalyzer, NegationParser

text = "غدوة باش نكتبلك ما ننساش"

verbs = VerbAnalyzer().find_features(text)
neg = NegationParser().find(text)

print(verbs)
print(neg)
```
