#!/usr/bin/env python3
"""Process Tunisian Derja Dataset using tun_linguist for high-quality data curation.

This script creates multiple output formats for different ML training purposes:
1. LLM instruction-tuning data (ChatML format)
2. SPO/DPO preference pairs (chosen/rejected based on linguistic authenticity)
3. Encoder training data (text + linguistic quality scores)

Usage:
    python scripts/process_derja_dataset.py --output-dir data/processed --sample-size 10000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Literal

# Add parent directory to path for tun_linguist import
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset, Dataset
from tun_linguist import (
    DialectScorer,
    DialectScore,
    VerbAnalyzer,
    NounAnalyzer,
    NegationParser,
    InterrogativeParser,
    ArabiziConverter,
)
from tun_linguist.lexicon import DiscourseMarkerDetector, FalseFriendDetector


# Dataset configs to load
DATASET_CONFIGS = [
    "Derja_tunsi",
    "HkayetErwi",
    "MADAR_TunisianDialect",
    "TuDiCOI",
    "TunBERT",
    "TunSwitchTunisiaOnly",
    "TunisianSentimentAnalysis",
    "Tweet_TN",
    "TA_Segmentation",
]

# Configs with potentially noisy or non-Tunisian data (excluded by default)
NOISY_CONFIGS = [
    "QADI_TunisianDialect",  # Contains other dialects
    "Sentiment_Derja",  # Mixed quality
    "TSAC",  # Contains offensive content
    "TunSwitchCodeSwitching",  # Heavy code-switching
    "Tunisian_Dialectic_English_Derja",  # Very large, mixed quality
]


@dataclass
class LinguisticFeatures:
    """Comprehensive linguistic feature extraction for a text sample."""
    
    # Authenticity score (main metric)
    authenticity_score: float
    is_authentic: bool
    
    # Positive markers found
    positives: dict[str, int]
    
    # Negative markers (MSA indicators)
    negatives: dict[str, int]
    
    # Detailed feature counts
    negation_circumfix_count: int = 0
    negation_pseudo_verb_count: int = 0
    discourse_markers: list[str] = field(default_factory=list)
    interrogatives: list[str] = field(default_factory=list)
    has_wh_in_situ: bool = False
    verb_features: dict[str, int] = field(default_factory=dict)
    noun_features: dict[str, int] = field(default_factory=dict)
    false_friends: list[str] = field(default_factory=list)
    
    # Metadata
    char_count: int = 0
    word_count: int = 0
    has_arabizi: bool = False
    notes: list[str] = field(default_factory=list)


class TunisianLinguistAnalyzer:
    """Comprehensive Tunisian Derja analyzer using tun_linguist components."""
    
    def __init__(self):
        self.scorer = DialectScorer()
        self.verb_analyzer = VerbAnalyzer()
        self.noun_analyzer = NounAnalyzer()
        self.negation_parser = NegationParser()
        self.interrogative_parser = InterrogativeParser()
        self.arabizi_converter = ArabiziConverter()
        self.discourse_detector = DiscourseMarkerDetector()
        self.false_friend_detector = FalseFriendDetector()
    
    def analyze(self, text: str, threshold: float = 5.0) -> LinguisticFeatures:
        """Perform comprehensive linguistic analysis on text."""
        
        # Get main authenticity score
        score_result = self.scorer.calculate_authenticity_score(text)
        
        # Negation analysis
        negations = self.negation_parser.find(text)
        circumfix_count = sum(1 for n in negations if n.kind == "circumfix")
        pseudo_verb_count = sum(1 for n in negations if n.kind == "pseudo_verb")
        
        # Discourse markers
        discourse_markers = self.discourse_detector.find(text)
        
        # Interrogatives
        interrogatives = self.interrogative_parser.find_particles(text)
        has_wh_in_situ = self.interrogative_parser.is_wh_in_situ(text)
        
        # Verb features
        verb_findings = self.verb_analyzer.find_features(text)
        verb_features: dict[str, int] = {}
        for vf in verb_findings:
            key = f"{vf.feature}_{vf.person}" if vf.person else vf.feature
            verb_features[key] = verb_features.get(key, 0) + 1
        
        # Noun features
        noun_findings = self.noun_analyzer.find_features(text)
        noun_features: dict[str, int] = {}
        for nf in noun_findings:
            noun_features[nf.feature] = noun_features.get(nf.feature, 0) + 1
        
        # False friends
        false_friends = [ff.form for ff in self.false_friend_detector.find(text)]
        
        # Check for Arabizi (text changes after conversion)
        converted = self.arabizi_converter.to_arabic(text)
        has_arabizi = converted != text
        
        # Word/char counts
        words = text.split()
        
        return LinguisticFeatures(
            authenticity_score=score_result.score,
            is_authentic=score_result.is_authentic(threshold),
            positives=score_result.positives,
            negatives=score_result.negatives,
            negation_circumfix_count=circumfix_count,
            negation_pseudo_verb_count=pseudo_verb_count,
            discourse_markers=discourse_markers,
            interrogatives=interrogatives,
            has_wh_in_situ=has_wh_in_situ,
            verb_features=verb_features,
            noun_features=noun_features,
            false_friends=false_friends,
            char_count=len(text),
            word_count=len(words),
            has_arabizi=has_arabizi,
            notes=score_result.notes,
        )


@dataclass
class ProcessedSample:
    """A processed text sample with linguistic features."""
    
    text: str
    source_config: str
    features: LinguisticFeatures
    quality_tier: Literal["high", "medium", "low"]
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source_config,
            "authenticity_score": self.features.authenticity_score,
            "quality_tier": self.quality_tier,
            "is_authentic": self.features.is_authentic,
            "positives": self.features.positives,
            "negatives": self.features.negatives,
            "word_count": self.features.word_count,
            "char_count": self.features.char_count,
            "has_arabizi": self.features.has_arabizi,
            "negation_circumfix_count": self.features.negation_circumfix_count,
            "discourse_markers": self.features.discourse_markers,
            "interrogatives": self.features.interrogatives,
            "has_wh_in_situ": self.features.has_wh_in_situ,
            "verb_features": self.features.verb_features,
            "noun_features": self.features.noun_features,
            "false_friends": self.features.false_friends,
        }


def load_all_configs(
    configs: list[str] | None = None,
    include_noisy: bool = False,
    max_per_config: int | None = None,
) -> Iterator[tuple[str, str]]:
    """Load text samples from all dataset configs.
    
    Yields:
        Tuples of (config_name, text)
    """
    if configs is None:
        configs = DATASET_CONFIGS.copy()
        if include_noisy:
            configs.extend(NOISY_CONFIGS)
    
    for config in configs:
        print(f"Loading config: {config}")
        try:
            ds = load_dataset("linagora/Tunisian_Derja_Dataset", config)
            train = ds["train"]
            
            if max_per_config and len(train) > max_per_config:
                indices = random.sample(range(len(train)), max_per_config)
                for idx in indices:
                    text = train[idx]["text"]
                    if text and isinstance(text, str) and text.strip():
                        yield config, text.strip()
            else:
                for row in train:
                    text = row["text"]
                    if text and isinstance(text, str) and text.strip():
                        yield config, text.strip()
                        
        except Exception as e:
            print(f"  Warning: Failed to load {config}: {e}")


def classify_quality_tier(features: LinguisticFeatures) -> Literal["high", "medium", "low"]:
    """Classify text quality based on linguistic features."""
    
    score = features.authenticity_score
    
    # High quality: strong Tunisian markers, no MSA markers, sufficient length
    if score >= 8.0 and not features.negatives and features.word_count >= 5:
        return "high"
    
    # Medium quality: positive score or has some Tunisian features
    if score >= 3.0 or (features.positives and not features.negatives):
        return "medium"
    
    return "low"


def process_dataset(
    analyzer: TunisianLinguistAnalyzer,
    configs: list[str] | None = None,
    include_noisy: bool = False,
    sample_size: int | None = None,
    max_per_config: int | None = None,
    min_word_count: int = 3,
    seed: int = 42,
) -> list[ProcessedSample]:
    """Process dataset and extract linguistic features."""
    
    random.seed(seed)
    
    samples: list[ProcessedSample] = []
    seen_texts: set[str] = set()
    
    for config, text in load_all_configs(configs, include_noisy, max_per_config):
        # Skip duplicates
        if text in seen_texts:
            continue
        seen_texts.add(text)
        
        # Skip very short texts
        if len(text.split()) < min_word_count:
            continue
        
        # Analyze text
        features = analyzer.analyze(text)
        quality_tier = classify_quality_tier(features)
        
        sample = ProcessedSample(
            text=text,
            source_config=config,
            features=features,
            quality_tier=quality_tier,
        )
        samples.append(sample)
        
        if sample_size and len(samples) >= sample_size:
            break
        
        if len(samples) % 10000 == 0:
            print(f"  Processed {len(samples)} samples...")
    
    return samples


# ============================================================================
# Output Format Generators
# ============================================================================

def generate_llm_instruction_data(
    samples: list[ProcessedSample],
    output_path: Path,
    min_quality: Literal["high", "medium", "low"] = "medium",
) -> int:
    """Generate instruction-tuning data in ChatML/ShareGPT format.
    
    Creates diverse instruction types for Tunisian Derja:
    - Text continuation
    - Grammar analysis
    - Dialect identification
    - Sentence completion
    """
    
    quality_order = {"high": 0, "medium": 1, "low": 2}
    min_quality_rank = quality_order[min_quality]
    
    instructions = []
    
    for sample in samples:
        if quality_order[sample.quality_tier] > min_quality_rank:
            continue
        
        text = sample.text
        features = sample.features
        
        # Generate different instruction types based on features
        
        # 1. Basic text understanding
        instructions.append({
            "messages": [
                {"role": "system", "content": "أنت مساعد متخصص في اللهجة التونسية (الدارجة التونسية)."},
                {"role": "user", "content": f"اشرح معنى هذا النص بالدارجة التونسية:\n{text}"},
                {"role": "assistant", "content": f"هذا نص بالدارجة التونسية. {_generate_explanation(features)}"},
            ],
            "metadata": {
                "source": sample.source_config,
                "authenticity_score": features.authenticity_score,
                "quality_tier": sample.quality_tier,
            }
        })
        
        # 2. If text has Tunisian negation, generate grammar instruction
        if features.negation_circumfix_count > 0:
            instructions.append({
                "messages": [
                    {"role": "system", "content": "أنت مساعد متخصص في قواعد اللهجة التونسية."},
                    {"role": "user", "content": f"حلل بنية النفي في هذه الجملة التونسية:\n{text}"},
                    {"role": "assistant", "content": f"تستخدم هذه الجملة نمط النفي التونسي المميز 'ما...ش' (ma...sh circumfix). هذا النمط خاص باللهجة التونسية ويختلف عن العربية الفصحى."},
                ],
                "metadata": {
                    "source": sample.source_config,
                    "instruction_type": "grammar_negation",
                }
            })
        
        # 3. If text has discourse markers, generate pragmatics instruction
        if features.discourse_markers:
            markers_str = "، ".join(features.discourse_markers[:3])
            instructions.append({
                "messages": [
                    {"role": "system", "content": "أنت متخصص في اللسانيات التونسية."},
                    {"role": "user", "content": f"ما هي علامات الخطاب في هذا النص التونسي؟\n{text}"},
                    {"role": "assistant", "content": f"يحتوي هذا النص على علامات خطاب تونسية مميزة: {markers_str}. هذه الكلمات تُستخدم في المحادثات اليومية للتأكيد أو جذب الانتباه."},
                ],
                "metadata": {
                    "source": sample.source_config,
                    "instruction_type": "discourse_markers",
                }
            })
        
        # 4. If text has interrogatives, generate question analysis
        if features.interrogatives:
            instructions.append({
                "messages": [
                    {"role": "system", "content": "أنت مساعد متخصص في اللهجة التونسية."},
                    {"role": "user", "content": f"حلل أدوات الاستفهام في هذا النص:\n{text}"},
                    {"role": "assistant", "content": f"يستخدم هذا النص أدوات استفهام تونسية مثل: {', '.join(features.interrogatives)}. {'هذا مثال على wh-in-situ حيث تأتي أداة الاستفهام في نهاية الجملة.' if features.has_wh_in_situ else ''}"},
                ],
                "metadata": {
                    "source": sample.source_config,
                    "instruction_type": "interrogatives",
                }
            })
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for inst in instructions:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")
    
    return len(instructions)


def _generate_explanation(features: LinguisticFeatures) -> str:
    """Generate a brief explanation based on features."""
    parts = []
    
    if features.positives:
        if "future_bash" in features.positives:
            parts.append("يستخدم 'باش/بش' للمستقبل")
        if "neg_ma_sh" in features.positives:
            parts.append("يستخدم نمط النفي 'ما...ش'")
        if "genitive_mta3" in features.positives:
            parts.append("يستخدم 'متاع' للملكية")
        if "discourse_marker" in features.positives:
            parts.append("يحتوي على علامات خطاب تونسية")
    
    if not parts:
        return "هذا نص عادي بالدارجة التونسية."
    
    return "من السمات اللغوية: " + "، ".join(parts) + "."


def generate_spo_preference_data(
    samples: list[ProcessedSample],
    output_path: Path,
    n_pairs: int = 5000,
    score_gap_threshold: float = 5.0,
) -> int:
    """Generate SPO/DPO preference pairs for training.
    
    Creates (prompt, chosen, rejected) tuples where:
    - chosen: high authenticity Tunisian text
    - rejected: lower authenticity or MSA-influenced text
    """
    
    # Sort by authenticity score
    sorted_samples = sorted(samples, key=lambda s: s.features.authenticity_score, reverse=True)
    
    # Split into high and low quality pools
    high_pool = [s for s in sorted_samples if s.features.authenticity_score >= 5.0]
    low_pool = [s for s in sorted_samples if s.features.authenticity_score < 2.0]
    
    if len(high_pool) < 100 or len(low_pool) < 100:
        print(f"Warning: Not enough samples for preference pairs. High: {len(high_pool)}, Low: {len(low_pool)}")
        return 0
    
    pairs = []
    used_high = set()
    used_low = set()
    
    random.shuffle(high_pool)
    random.shuffle(low_pool)
    
    prompts = [
        "اكتب جملة بالدارجة التونسية الأصيلة:",
        "أعطني مثالاً على اللهجة التونسية:",
        "كيف يقول التونسي هذا؟",
        "تكلم بالدارجة التونسية:",
        "اكتب نص تونسي:",
    ]
    
    for i, (high, low) in enumerate(zip(high_pool, low_pool)):
        if i >= n_pairs:
            break
        
        # Skip if texts are too similar in length (might be same content)
        if abs(len(high.text) - len(low.text)) < 10 and high.text[:20] == low.text[:20]:
            continue
        
        # Ensure sufficient score gap
        if high.features.authenticity_score - low.features.authenticity_score < score_gap_threshold:
            continue
        
        prompt = random.choice(prompts)
        
        pair = {
            "prompt": prompt,
            "chosen": high.text,
            "rejected": low.text,
            "chosen_score": high.features.authenticity_score,
            "rejected_score": low.features.authenticity_score,
            "chosen_features": {
                "positives": high.features.positives,
                "negatives": high.features.negatives,
            },
            "rejected_features": {
                "positives": low.features.positives,
                "negatives": low.features.negatives,
            },
        }
        pairs.append(pair)
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    return len(pairs)


def generate_encoder_training_data(
    samples: list[ProcessedSample],
    output_path: Path,
) -> int:
    """Generate encoder training data with linguistic quality scores.
    
    Output format suitable for training a quality scoring encoder:
    - text: input text
    - score: authenticity score (continuous)
    - features: detailed feature vector
    """
    
    records = []
    
    for sample in samples:
        features = sample.features
        
        # Create feature vector for encoder training
        feature_vector = {
            # Main score (regression target)
            "authenticity_score": features.authenticity_score,
            
            # Binary classification targets
            "is_authentic": features.is_authentic,
            "has_negation_circumfix": features.negation_circumfix_count > 0,
            "has_discourse_markers": len(features.discourse_markers) > 0,
            "has_interrogatives": len(features.interrogatives) > 0,
            "has_wh_in_situ": features.has_wh_in_situ,
            "has_arabizi": features.has_arabizi,
            "has_msa_markers": len(features.negatives) > 0,
            
            # Count features
            "negation_count": features.negation_circumfix_count + features.negation_pseudo_verb_count,
            "discourse_marker_count": len(features.discourse_markers),
            "positive_marker_count": sum(features.positives.values()),
            "negative_marker_count": sum(features.negatives.values()),
            "word_count": features.word_count,
            "char_count": features.char_count,
            
            # Categorical
            "quality_tier": sample.quality_tier,
        }
        
        record = {
            "text": sample.text,
            "source": sample.source_config,
            "features": feature_vector,
        }
        records.append(record)
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    return len(records)


def generate_contrastive_pairs(
    samples: list[ProcessedSample],
    output_path: Path,
    n_pairs: int = 10000,
) -> int:
    """Generate contrastive learning pairs for encoder training.
    
    Creates pairs of:
    - Similar texts (same quality tier) -> positive pairs
    - Different quality texts -> negative pairs
    """
    
    # Group by quality tier
    by_tier = {"high": [], "medium": [], "low": []}
    for s in samples:
        by_tier[s.quality_tier].append(s)
    
    pairs = []
    
    # Positive pairs: same tier (should be similar)
    for tier in ["high", "medium"]:
        tier_samples = by_tier[tier]
        if len(tier_samples) < 2:
            continue
        
        for _ in range(min(n_pairs // 4, len(tier_samples) // 2)):
            s1, s2 = random.sample(tier_samples, 2)
            pairs.append({
                "text_a": s1.text,
                "text_b": s2.text,
                "label": 1,  # Similar (both authentic)
                "tier_a": tier,
                "tier_b": tier,
                "score_a": s1.features.authenticity_score,
                "score_b": s2.features.authenticity_score,
            })
    
    # Negative pairs: high vs low tier (should be different)
    high_samples = by_tier["high"]
    low_samples = by_tier["low"]
    
    if high_samples and low_samples:
        for _ in range(min(n_pairs // 2, len(high_samples), len(low_samples))):
            s_high = random.choice(high_samples)
            s_low = random.choice(low_samples)
            pairs.append({
                "text_a": s_high.text,
                "text_b": s_low.text,
                "label": 0,  # Different (authentic vs not)
                "tier_a": "high",
                "tier_b": "low",
                "score_a": s_high.features.authenticity_score,
                "score_b": s_low.features.authenticity_score,
            })
    
    random.shuffle(pairs)
    pairs = pairs[:n_pairs]
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    return len(pairs)


def print_statistics(samples: list[ProcessedSample]) -> None:
    """Print dataset statistics."""
    
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal samples: {len(samples)}")
    
    # Quality tier distribution
    tier_counts = {"high": 0, "medium": 0, "low": 0}
    for s in samples:
        tier_counts[s.quality_tier] += 1
    
    print("\nQuality tier distribution:")
    for tier, count in tier_counts.items():
        pct = count / len(samples) * 100
        print(f"  {tier}: {count} ({pct:.1f}%)")
    
    # Score distribution
    scores = [s.features.authenticity_score for s in samples]
    print(f"\nAuthenticity score statistics:")
    print(f"  Min: {min(scores):.2f}")
    print(f"  Max: {max(scores):.2f}")
    print(f"  Mean: {sum(scores)/len(scores):.2f}")
    
    # Feature prevalence
    feature_counts = {
        "negation_circumfix": 0,
        "discourse_markers": 0,
        "interrogatives": 0,
        "wh_in_situ": 0,
        "arabizi": 0,
        "msa_markers": 0,
    }
    
    for s in samples:
        if s.features.negation_circumfix_count > 0:
            feature_counts["negation_circumfix"] += 1
        if s.features.discourse_markers:
            feature_counts["discourse_markers"] += 1
        if s.features.interrogatives:
            feature_counts["interrogatives"] += 1
        if s.features.has_wh_in_situ:
            feature_counts["wh_in_situ"] += 1
        if s.features.has_arabizi:
            feature_counts["arabizi"] += 1
        if s.features.negatives:
            feature_counts["msa_markers"] += 1
    
    print("\nFeature prevalence:")
    for feat, count in feature_counts.items():
        pct = count / len(samples) * 100
        print(f"  {feat}: {count} ({pct:.1f}%)")
    
    # Source distribution
    source_counts: dict[str, int] = {}
    for s in samples:
        source_counts[s.source_config] = source_counts.get(s.source_config, 0) + 1
    
    print("\nSource distribution:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = count / len(samples) * 100
        print(f"  {source}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Process Tunisian Derja Dataset")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Maximum number of samples to process (None = all)")
    parser.add_argument("--max-per-config", type=int, default=50000,
                        help="Maximum samples per dataset config")
    parser.add_argument("--include-noisy", action="store_true",
                        help="Include potentially noisy dataset configs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--min-word-count", type=int, default=3,
                        help="Minimum word count for samples")
    parser.add_argument("--spo-pairs", type=int, default=5000,
                        help="Number of SPO/DPO preference pairs to generate")
    parser.add_argument("--contrastive-pairs", type=int, default=10000,
                        help="Number of contrastive pairs for encoder training")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Initializing Tunisian linguist analyzer...")
    analyzer = TunisianLinguistAnalyzer()
    
    print("\nProcessing dataset...")
    samples = process_dataset(
        analyzer,
        include_noisy=args.include_noisy,
        sample_size=args.sample_size,
        max_per_config=args.max_per_config,
        min_word_count=args.min_word_count,
        seed=args.seed,
    )
    
    print_statistics(samples)
    
    # Generate outputs
    print("\n" + "=" * 60)
    print("GENERATING OUTPUT FILES")
    print("=" * 60)
    
    # 1. Full processed dataset with features
    full_path = output_dir / "tunisian_derja_scored.jsonl"
    print(f"\n1. Full scored dataset: {full_path}")
    with full_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    print(f"   Wrote {len(samples)} samples")
    
    # 2. High-quality filtered dataset
    high_quality = [s for s in samples if s.quality_tier == "high"]
    high_path = output_dir / "tunisian_derja_high_quality.jsonl"
    print(f"\n2. High-quality dataset: {high_path}")
    with high_path.open("w", encoding="utf-8") as f:
        for s in high_quality:
            f.write(json.dumps({"text": s.text, "score": s.features.authenticity_score}, ensure_ascii=False) + "\n")
    print(f"   Wrote {len(high_quality)} samples")
    
    # 3. LLM instruction-tuning data
    llm_path = output_dir / "tunisian_derja_instructions.jsonl"
    print(f"\n3. LLM instruction data: {llm_path}")
    n_instructions = generate_llm_instruction_data(samples, llm_path, min_quality="medium")
    print(f"   Wrote {n_instructions} instruction samples")
    
    # 4. SPO/DPO preference data
    spo_path = output_dir / "tunisian_derja_spo_pairs.jsonl"
    print(f"\n4. SPO/DPO preference pairs: {spo_path}")
    n_pairs = generate_spo_preference_data(samples, spo_path, n_pairs=args.spo_pairs)
    print(f"   Wrote {n_pairs} preference pairs")
    
    # 5. Encoder training data
    encoder_path = output_dir / "tunisian_derja_encoder_data.jsonl"
    print(f"\n5. Encoder training data: {encoder_path}")
    n_encoder = generate_encoder_training_data(samples, encoder_path)
    print(f"   Wrote {n_encoder} encoder samples")
    
    # 6. Contrastive pairs
    contrastive_path = output_dir / "tunisian_derja_contrastive_pairs.jsonl"
    print(f"\n6. Contrastive pairs: {contrastive_path}")
    n_contrastive = generate_contrastive_pairs(samples, contrastive_path, n_pairs=args.contrastive_pairs)
    print(f"   Wrote {n_contrastive} contrastive pairs")
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
