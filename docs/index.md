## tun_linguist

**Tunisian Derja linguistic feature extraction + dataset filtering.**

`tun_linguist` is built for *high-precision* dialect identification and dataset curation. It focuses on a small set of **strong discriminators** that reliably separate Tunisian Derja from MSA in noisy, real-world text.

<div class="grid cards" markdown>

-   ### Score authenticity
    Debuggable rule-based scoring (positives/negatives + notes).

    **Best for**: filtering corpora, ranking MT outputs, dataset curation.

-   ### Extract markers
    Verbal morphology, negation circumfix, genitive, discourse particles, interrogatives.

    **Best for**: analysis, feature engineering, QA.

-   ### Handle noisy writing
    Conservative normalization (Arabizi digits) + phonology exemplar tagging (Qaf/Gaf).

    **Best for**: grouping, retrieval, diagnostics.

-   ### Generate training data
    Use the `scripts/` pipeline to produce scored JSONL, high-quality subsets, and preference pairs.

</div>

### 60‑second quickstart

```python
from tun_linguist import DialectScorer

scorer = DialectScorer()

examples = [
    "غدوة باش نمشي للسوق",          # future marker باش/بش
    "ما-قلت-لها-ش",                 # ma…sh with trapped clitic
    "هاذوما صحابي",                 # demonstrative هاذوما
    "سوف أذهب إلى السوق غداً",      # strong MSA marker
]

for t in examples:
    r = scorer.calculate_authenticity_score(t)
    print("---", t)
    print("score:", r.score)
    print("positives:", r.positives)
    print("negatives:", r.negatives)
    print("notes:", r.notes)
```

### Install

```bash
pip install -e .
```

### Build these docs locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

!!! tip "Not seeing the menu/tabs?"
    On small screens, MkDocs Material collapses navigation into the **hamburger menu** (top-left).
