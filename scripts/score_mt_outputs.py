#!/usr/bin/env python3
"""Score machine translation outputs for Tunisian Derja linguistic quality.

This script evaluates MT model outputs by computing linguistic authenticity scores
using the tun_linguist library. Useful for:
- Evaluating Tunisian MT models
- Quality filtering of MT outputs for training data
- Creating MT quality rankings for model comparison

Usage:
    # Score a single file
    python scripts/score_mt_outputs.py --input mt_outputs.jsonl --output scored_outputs.jsonl
    
    # Compare multiple systems
    python scripts/score_mt_outputs.py --compare system1.jsonl system2.jsonl system3.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from tun_linguist import DialectScorer, DialectScore


@dataclass
class MTScoringResult:
    """Result of scoring an MT output."""
    
    source: str
    hypothesis: str
    reference: Optional[str]
    
    # Linguistic scores
    authenticity_score: float
    is_authentic: bool
    positives: dict[str, int]
    negatives: dict[str, int]
    
    # Metadata
    system: Optional[str] = None
    notes: list[str] = None
    
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "hypothesis": self.hypothesis,
            "reference": self.reference,
            "authenticity_score": self.authenticity_score,
            "is_authentic": self.is_authentic,
            "positives": self.positives,
            "negatives": self.negatives,
            "system": self.system,
            "notes": self.notes or [],
        }


class MTQualityScorer:
    """Scores MT outputs for Tunisian Derja linguistic quality."""
    
    def __init__(self, threshold: float = 5.0):
        self.scorer = DialectScorer()
        self.threshold = threshold
    
    def score(
        self,
        hypothesis: str,
        source: Optional[str] = None,
        reference: Optional[str] = None,
        system: Optional[str] = None,
    ) -> MTScoringResult:
        """Score a single MT output."""
        
        result = self.scorer.calculate_authenticity_score(hypothesis)
        
        return MTScoringResult(
            source=source or "",
            hypothesis=hypothesis,
            reference=reference,
            authenticity_score=result.score,
            is_authentic=result.is_authentic(self.threshold),
            positives=result.positives,
            negatives=result.negatives,
            system=system,
            notes=result.notes,
        )
    
    def score_file(
        self,
        input_path: Path,
        output_path: Path,
        hypothesis_key: str = "hypothesis",
        source_key: str = "source",
        reference_key: str = "reference",
        system: Optional[str] = None,
    ) -> dict[str, float]:
        """Score all MT outputs in a JSONL file.
        
        Returns aggregate statistics.
        """
        results = []
        scores = []
        authentic_count = 0
        
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                
                hypothesis = record.get(hypothesis_key, "")
                source = record.get(source_key, "")
                reference = record.get(reference_key)
                
                result = self.score(
                    hypothesis=hypothesis,
                    source=source,
                    reference=reference,
                    system=system,
                )
                results.append(result)
                scores.append(result.authenticity_score)
                
                if result.is_authentic:
                    authentic_count += 1
        
        # Write results
        with output_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        
        # Compute statistics
        n = len(scores)
        return {
            "count": n,
            "mean_score": sum(scores) / n if n > 0 else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "authentic_ratio": authentic_count / n if n > 0 else 0,
            "authentic_count": authentic_count,
        }
    
    def compare_systems(
        self,
        system_files: list[tuple[str, Path]],
    ) -> dict[str, dict[str, float]]:
        """Compare multiple MT systems.
        
        Args:
            system_files: List of (system_name, file_path) tuples
            
        Returns:
            Dictionary mapping system names to their statistics
        """
        results = {}
        
        for system_name, file_path in system_files:
            output_path = file_path.parent / f"{file_path.stem}_scored.jsonl"
            stats = self.score_file(
                input_path=file_path,
                output_path=output_path,
                system=system_name,
            )
            results[system_name] = stats
        
        return results


def score_interactive(scorer: MTQualityScorer):
    """Interactive scoring mode."""
    print("\nInteractive Tunisian MT Scoring")
    print("Enter MT outputs to score (Ctrl+C to exit)")
    print("-" * 50)
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
            if not text:
                continue
            
            result = scorer.score(text)
            
            print(f"\nAuthenticity Score: {result.authenticity_score:.2f}")
            print(f"Is Authentic: {result.is_authentic}")
            
            if result.positives:
                print(f"Positive Markers: {result.positives}")
            if result.negatives:
                print(f"Negative Markers (MSA): {result.negatives}")
            if result.notes:
                print(f"Notes: {result.notes}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(description="Score MT outputs for Tunisian quality")
    parser.add_argument("--input", type=str, help="Input JSONL file")
    parser.add_argument("--output", type=str, help="Output JSONL file")
    parser.add_argument("--compare", nargs="+", help="Compare multiple system files")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Authenticity threshold")
    parser.add_argument("--hypothesis-key", type=str, default="hypothesis",
                        help="Key for hypothesis in input JSON")
    parser.add_argument("--source-key", type=str, default="source",
                        help="Key for source in input JSON")
    parser.add_argument("--reference-key", type=str, default="reference",
                        help="Key for reference in input JSON")
    
    args = parser.parse_args()
    
    scorer = MTQualityScorer(threshold=args.threshold)
    
    if args.interactive:
        score_interactive(scorer)
        return
    
    if args.compare:
        # Compare multiple systems
        system_files = []
        for path_str in args.compare:
            path = Path(path_str)
            system_name = path.stem
            system_files.append((system_name, path))
        
        print("Comparing MT systems...")
        results = scorer.compare_systems(system_files)
        
        print("\n" + "=" * 60)
        print("MT SYSTEM COMPARISON")
        print("=" * 60)
        
        # Sort by mean score
        sorted_systems = sorted(results.items(), key=lambda x: -x[1]["mean_score"])
        
        for rank, (system, stats) in enumerate(sorted_systems, 1):
            print(f"\n{rank}. {system}")
            print(f"   Mean Score: {stats['mean_score']:.2f}")
            print(f"   Authentic Ratio: {stats['authentic_ratio']:.2%}")
            print(f"   Score Range: [{stats['min_score']:.2f}, {stats['max_score']:.2f}]")
            print(f"   Samples: {stats['count']}")
        
        # Create comparison summary
        summary_path = Path(args.compare[0]).parent / "mt_comparison_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSummary saved to: {summary_path}")
        
    elif args.input:
        # Score single file
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_scored.jsonl"
        
        print(f"Scoring {input_path}...")
        stats = scorer.score_file(
            input_path=input_path,
            output_path=output_path,
            hypothesis_key=args.hypothesis_key,
            source_key=args.source_key,
            reference_key=args.reference_key,
        )
        
        print("\n" + "=" * 40)
        print("SCORING RESULTS")
        print("=" * 40)
        print(f"Samples: {stats['count']}")
        print(f"Mean Score: {stats['mean_score']:.2f}")
        print(f"Authentic Ratio: {stats['authentic_ratio']:.2%}")
        print(f"Score Range: [{stats['min_score']:.2f}, {stats['max_score']:.2f}]")
        print(f"\nOutput: {output_path}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
