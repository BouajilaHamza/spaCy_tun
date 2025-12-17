# Tunisian Derja Dataset Processing Pipeline

A comprehensive toolkit for creating high-quality Tunisian Arabic (Derja) training data using the **tun_linguist** library (originally spaCy_tun). This pipeline processes the [Linagora Tunisian Derja Dataset](https://huggingface.co/datasets/linagora/Tunisian_Derja_Dataset) and generates data suitable for:

- **LLM Fine-tuning** (instruction-following format)
- **SPO/DPO Training** (preference pairs based on linguistic authenticity)
- **Encoder Training** (linguistic quality scoring models)

## Features

### tun_linguist Library

The core library provides linguistic analysis tools specifically designed for Tunisian Derja:

| Component | Description |
|-----------|-------------|
| `DialectScorer` | Computes authenticity scores based on Tunisian vs MSA markers |
| `VerbAnalyzer` | Detects Tunisian verbal morphology (n-prefix, bash/besh future, qa3id progressive) |
| `NounAnalyzer` | Detects nominal features (mta3 possession, plurals) |
| `NegationParser` | Parses Tunisian ma...sh circumfix negation |
| `InterrogativeParser` | Detects Tunisian interrogatives and wh-in-situ |
| `ArabiziConverter` | Converts Arabizi digits (3,7,9,5) to Arabic letters |
| `DiscourseMarkerDetector` | Finds Tunisian discourse particles (ti, yaxxi, bara, mela) |
| `FalseFriendDetector` | Identifies Tunisian-MSA semantic shifts |

### Authenticity Scoring

The scoring model rewards definitive Tunisian markers while penalizing MSA markers:

**Positive Indicators (Tunisian):**
- Future particles: `باش/بش` (bash/besh)
- Possession: `متاع` (mta3)
- Negation: `ما...ش` (ma...sh circumfix)
- Imperfective 1sg/1pl: n- prefix paradigm
- Progressive: `قاعد` (qa3id)
- Discourse markers: `تي`, `ياخي`, `برا`, `مالا`

**Negative Indicators (MSA):**
- Future: `سوف` (sawfa), `س` prefix
- Negation: `لن`, `لم`, `ليس`
- Interrogatives: `ماذا`, `لماذا`, `كيف`, `أين`
- Imperfective 1sg: أ- prefix

## Installation

```bash
pip install -e .
pip install datasets huggingface_hub torch
```

## Quick Start

### 1. Demo the Library

```bash
python scripts/demo_tun_linguist.py
```

### 2. Process the Full Dataset

```bash
python scripts/process_derja_dataset.py \
    --output-dir data/processed \
    --sample-size 100000 \
    --max-per-config 30000 \
    --spo-pairs 10000 \
    --contrastive-pairs 20000
```

### 3. Score MT Outputs

```bash
# Interactive mode
python scripts/score_mt_outputs.py --interactive

# Score a file
python scripts/score_mt_outputs.py --input mt_outputs.jsonl --output scored.jsonl

# Compare systems
python scripts/score_mt_outputs.py --compare system1.jsonl system2.jsonl
```

## Output Formats

### 1. Scored Dataset (`tunisian_derja_scored.jsonl`)

Full dataset with linguistic features:

```json
{
  "text": "غدوة باش نمشي للبلدية نقد اوراقي",
  "source": "TunBERT",
  "authenticity_score": 12.0,
  "quality_tier": "high",
  "is_authentic": true,
  "positives": {"future_bash": 1, "n_prefix_1sg": 2},
  "negatives": {},
  "word_count": 5,
  "negation_circumfix_count": 0,
  "discourse_markers": [],
  "verb_features": {"future_bash": 1, "imperfective_paradigm_1sg": 2}
}
```

### 2. High-Quality Dataset (`tunisian_derja_high_quality.jsonl`)

Filtered samples with score ≥ 8.0:

```json
{"text": "غدوة باش نمشي للبلدية نقد اوراقي", "score": 12.0}
```

### 3. LLM Instructions (`tunisian_derja_instructions.jsonl`)

ChatML/ShareGPT format for instruction tuning:

```json
{
  "messages": [
    {"role": "system", "content": "أنت مساعد متخصص في اللهجة التونسية..."},
    {"role": "user", "content": "حلل بنية النفي في هذه الجملة..."},
    {"role": "assistant", "content": "تستخدم هذه الجملة نمط النفي التونسي..."}
  ],
  "metadata": {"source": "TunBERT", "instruction_type": "grammar_negation"}
}
```

### 4. SPO/DPO Pairs (`tunisian_derja_spo_pairs.jsonl`)

Preference pairs for alignment training:

```json
{
  "prompt": "اكتب جملة بالدارجة التونسية الأصيلة:",
  "chosen": "غدوة باش نمشي للسوق متاعنا",
  "rejected": "سوف أذهب إلى السوق غداً",
  "chosen_score": 12.5,
  "rejected_score": -10.0,
  "chosen_features": {"positives": {"future_bash": 1, "n_prefix_1sg": 1}},
  "rejected_features": {"negatives": {"msa_marker": 1}}
}
```

### 5. Encoder Training Data (`tunisian_derja_encoder_data.jsonl`)

For training linguistic quality scoring models:

```json
{
  "text": "ما فهمتش شنوة قالتلي",
  "source": "Derja_tunsi",
  "features": {
    "authenticity_score": 4.5,
    "is_authentic": false,
    "has_negation_circumfix": true,
    "has_discourse_markers": false,
    "positive_marker_count": 1,
    "negative_marker_count": 0,
    "quality_tier": "medium"
  }
}
```

### 6. Contrastive Pairs (`tunisian_derja_contrastive_pairs.jsonl`)

For contrastive encoder training:

```json
{
  "text_a": "باش نكتب رسالة",
  "text_b": "باش نقرا كتاب",
  "label": 1,
  "tier_a": "high",
  "tier_b": "high",
  "score_a": 8.5,
  "score_b": 8.0
}
```

## Training a Quality Encoder

```bash
python scripts/train_quality_encoder.py \
    --data-path data/processed/tunisian_derja_encoder_data.jsonl \
    --output-dir models/quality_encoder \
    --epochs 10 \
    --batch-size 32 \
    --hidden-size 256
```

## Dataset Statistics (100K Sample)

| Metric | Value |
|--------|-------|
| Total Samples | 100,000 |
| High Quality (score ≥ 8.0) | 4,109 (4.1%) |
| Medium Quality (score ≥ 3.0) | 23,524 (23.5%) |
| Low Quality | 72,367 (72.4%) |

**Feature Prevalence:**
- Negation (ma...sh): 8.6%
- Discourse Markers: 2.5%
- Interrogatives: 4.2%
- MSA Markers: 25.8%

## Use Cases

### 1. Train a Tunisian LLM

Use `tunisian_derja_instructions.jsonl` for supervised fine-tuning:

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="data/processed/tunisian_derja_instructions.jsonl")
# Fine-tune with your favorite framework (transformers, axolotl, etc.)
```

### 2. Align with SPO/DPO

Use `tunisian_derja_spo_pairs.jsonl` for preference optimization:

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="data/processed/tunisian_derja_spo_pairs.jsonl")
# Use with TRL's DPOTrainer or similar
```

### 3. Train a Quality Scoring Model

Use for filtering MT outputs or ranking translations:

```python
from tun_linguist import DialectScorer

scorer = DialectScorer()
result = scorer.calculate_authenticity_score("غدوة باش نمشي للسوق")
print(f"Score: {result.score}, Authentic: {result.is_authentic()}")
```

### 4. Evaluate MT Systems

```bash
python scripts/score_mt_outputs.py --compare baseline.jsonl improved.jsonl
```

## API Reference

### DialectScorer

```python
from tun_linguist import DialectScorer

scorer = DialectScorer()
result = scorer.calculate_authenticity_score(text, normalize_arabizi_digits=True)

# Result contains:
# - score: float (higher = more authentic Tunisian)
# - positives: dict[str, int] (Tunisian markers found)
# - negatives: dict[str, int] (MSA markers found)
# - notes: list[str] (additional information)
# - is_authentic(threshold=5.0) -> bool
```

### VerbAnalyzer

```python
from tun_linguist import VerbAnalyzer

analyzer = VerbAnalyzer()
findings = analyzer.find_features(text)

# Each finding contains:
# - span: (start, end)
# - surface: str (matched text)
# - feature: str (e.g., "future_bash", "imperfective_paradigm")
# - person: str | None ("1sg", "1pl")
# - script: str | None ("arabic", "latin")
```

### NegationParser

```python
from tun_linguist import NegationParser

parser = NegationParser()
findings = parser.find(text)

# Each finding contains:
# - span: (start, end)
# - text: str (matched negation)
# - kind: str ("circumfix" or "pseudo_verb")
```

## License

See [LICENSE](LICENSE) file.

## Acknowledgments

- [Linagora Tunisian Derja Dataset](https://huggingface.co/datasets/linagora/Tunisian_Derja_Dataset)
- Original spaCy_tun concept by [BouajilaHamza](https://github.com/BouajilaHamza/spaCy_tun)
