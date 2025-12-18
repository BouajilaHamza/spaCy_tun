## tun_linguist — Tunisian Derja linguistic toolkit

`tun_linguist` is a production-oriented library for **Tunisian Arabic (Derja)** linguistic feature extraction, normalization, and dataset filtering.

It is designed for **high-precision dataset curation**: detect *definitive* Tunisian markers (e.g., `bash/besh`, `ma…sh`, `mta3`, discourse particles) and penalize *strong* MSA constraints (e.g., `sawfa`, `lan/lam/laysa`, `سـ` future prefix).

### Documentation (GitHub Pages)

This repo includes a modern documentation site built with **MkDocs Material** and deployed via **GitHub Actions → GitHub Pages**.

- **Build locally**:

```bash
pip install -e ".[docs]"
mkdocs serve
```

- **Enable on GitHub** (one-time): go to **Repository Settings → Pages → Source** and select **GitHub Actions**. The workflow is in `.github/workflows/docs.yml`.

## Installation

### Library only

```bash
pip install -e .
```

### Extras

```bash
# Dataset processing scripts
pip install -e ".[processing]"

# Training scripts
pip install -e ".[training]"

# Everything
pip install -e ".[all]"
```

## Quick start

### Score a sentence for Tunisian authenticity

```python
from tun_linguist import DialectScorer

scorer = DialectScorer()
result = scorer.calculate_authenticity_score("غدوة باش نمشي للسوق")

print(result.score)
print(result.positives)  # Tunisian markers
print(result.negatives)  # MSA constraints
print(result.notes)      # diagnostics / normalization notes
```

### Detect key grammatical markers

```python
from tun_linguist import NegationParser, VerbAnalyzer

text = "ما-قلت-لها-ش باش نمشي"

print(NegationParser().find(text))
print(VerbAnalyzer().find_features(text))
```

## What’s inside

### Core library (`tun_linguist/`)

- **`DialectScorer`** (`tun_linguist/filter.py`)
  - Transparent rule-based authenticity score + marker counts.
  - Rewards Tunisian markers, penalizes strong MSA constraints.
  - Adds an explicit bonus for **clitic trapping** in negation (e.g., `ma-qolt-el-ha-sh`).
  - Rewards distinct Tunisian **demonstratives** (`hadhouma`, `haka`, `haki` and Arabic-script forms).
  - Records Qaf/Gaf exemplar variation (heart: qalb/galb) as notes.

- **`VerbAnalyzer`** (`tun_linguist/morphology.py`)
  - **Paradigm shift**: imperfective `n-` patterns (1sg vs 1pl heuristics).
  - **Future**: `bash/besh` (`باش/بش`).
  - **Progressive**: `qa3id` (`قاعد/قاعدة/قاعدين`).

- **`NounAnalyzer`** (`tun_linguist/morphology.py`)
  - **Possession/genitive**: `mta3` (`متاع`).

- **`NegationParser`** (`tun_linguist/syntax.py`)
  - Detects `ma … sh/ch` circumfix (Arabic + Latin) and marks **trapped clitics**.

- **`PhonologyNormalizer`** (`tun_linguist/normalizer.py`)
  - High-precision **Qaf/Gaf** exemplar tagging and unification for “heart”:
    - Latin/Arabizi: `qalb/galb/9alb`
    - Arabic script: `قلب/ڨلب/گلب/ݣلب`

- **`ArabiziConverter`** (`tun_linguist/normalizer.py`)
  - Converts common Arabizi digits: `3→ع`, `7→ح`, `9→ق`, `5→خ`.

- **`DiscourseMarkerDetector` / `FalseFriendDetector`** (`tun_linguist/lexicon.py`)
  - High-salience Tunisian discourse particles and false-friend diagnostics.

## Dataset pipeline scripts (`scripts/`)

- `scripts/demo_tun_linguist.py`: quick demo.
- `scripts/process_derja_dataset.py`: process the Linagora dataset and write curated JSONL artifacts.
- `scripts/score_mt_outputs.py`: score MT outputs (interactive, file scoring, system comparison).

## License

See `LICENSE`.

## Acknowledgments

- Linagora Tunisian Derja Dataset: `https://huggingface.co/datasets/linagora/Tunisian_Derja_Dataset`
