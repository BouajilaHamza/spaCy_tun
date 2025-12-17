#!/usr/bin/env python3
"""Demo script showing tun_linguist capabilities for Tunisian Derja analysis.

This script demonstrates:
1. Dialect authenticity scoring
2. Morphological analysis (verbs, nouns)
3. Negation pattern detection
4. Interrogative parsing
5. Arabizi conversion
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tun_linguist import (
    DialectScorer,
    VerbAnalyzer,
    NounAnalyzer,
    NegationParser,
    InterrogativeParser,
    ArabiziConverter,
)
from tun_linguist.lexicon import DiscourseMarkerDetector, FalseFriendDetector


def demo_dialect_scoring():
    """Demonstrate dialect authenticity scoring."""
    
    print("\n" + "=" * 60)
    print("DIALECT AUTHENTICITY SCORING")
    print("=" * 60)
    
    scorer = DialectScorer()
    
    # Test sentences with different authenticity levels
    test_sentences = [
        # High authenticity (Tunisian markers)
        ("غدوة باش نمشي للبلدية نقد اوراقي", "High - باش + n-prefix"),
        ("ما فهمتش شنوة قالتلي", "High - ma...sh negation"),
        ("الكرهبة متاعي خسرت", "High - متاع possession"),
        ("تي شكون قالك هكاكة؟", "High - تي discourse marker"),
        
        # Medium authenticity
        ("أنا نحب الأكل التونسي برشة", "Medium - n-prefix only"),
        ("الجو اليوم باهي", "Medium - Tunisian lexicon"),
        
        # Low authenticity (MSA markers)
        ("سوف أذهب إلى المدرسة غداً", "Low - MSA future سوف"),
        ("لن أستطيع الحضور", "Low - MSA negation لن"),
        ("ماذا تريد؟", "Low - MSA interrogative"),
    ]
    
    for text, description in test_sentences:
        result = scorer.calculate_authenticity_score(text)
        
        print(f"\nText: {text}")
        print(f"Description: {description}")
        print(f"Score: {result.score:.1f}")
        print(f"Is Authentic: {result.is_authentic()}")
        if result.positives:
            print(f"Positives: {result.positives}")
        if result.negatives:
            print(f"Negatives: {result.negatives}")


def demo_verb_analysis():
    """Demonstrate verbal morphology analysis."""
    
    print("\n" + "=" * 60)
    print("VERBAL MORPHOLOGY ANALYSIS")
    print("=" * 60)
    
    analyzer = VerbAnalyzer()
    
    test_sentences = [
        "باش نكتب رسالة",  # bash + n-prefix (1sg)
        "قاعدين نخدمو في المشروع",  # qa3din + n-prefix (1pl)
        "نحب نقرا كتاب",  # n-prefix imperfective
        "أكتب رسالة",  # MSA a-prefix (should be flagged)
    ]
    
    for text in test_sentences:
        print(f"\nText: {text}")
        findings = analyzer.find_features(text)
        
        if findings:
            for f in findings:
                print(f"  Feature: {f.feature}, Person: {f.person}, Surface: '{f.surface}'")
        else:
            print("  No features detected")


def demo_negation_analysis():
    """Demonstrate negation pattern detection."""
    
    print("\n" + "=" * 60)
    print("NEGATION PATTERN DETECTION")
    print("=" * 60)
    
    parser = NegationParser()
    
    test_sentences = [
        "ما فهمتش",  # ma...sh circumfix
        "مانيش باش نجي",  # manish pseudo-verb
        "ما عندوش فلوس",  # ma...sh with object
        "موش صحيح",  # mush/moush
        "ما كتبتلوش الرسالة",  # circumfix with trapped clitic
    ]
    
    for text in test_sentences:
        print(f"\nText: {text}")
        findings = parser.find(text)
        
        if findings:
            for f in findings:
                print(f"  Kind: {f.kind}, Match: '{f.text}'")
        else:
            print("  No negation patterns detected")


def demo_interrogatives():
    """Demonstrate interrogative parsing."""
    
    print("\n" + "=" * 60)
    print("INTERROGATIVE PARSING")
    print("=" * 60)
    
    parser = InterrogativeParser()
    
    test_sentences = [
        "شكون جاء؟",  # shkun (who)
        "وين مشيت؟",  # win (where)
        "كيفاش تعمل هاذا؟",  # kifash (how)
        "علاش ما جيتش؟",  # 3lash (why)
        "وقتاش باش تجي؟",  # waqtash (when)
        "شنوة صار؟",  # shnuwa (what)
        "مشيت وين؟",  # wh-in-situ
    ]
    
    for text in test_sentences:
        print(f"\nText: {text}")
        particles = parser.find_particles(text)
        is_in_situ = parser.is_wh_in_situ(text)
        
        print(f"  Particles: {particles}")
        print(f"  WH-in-situ: {is_in_situ}")


def demo_arabizi_conversion():
    """Demonstrate Arabizi to Arabic conversion."""
    
    print("\n" + "=" * 60)
    print("ARABIZI CONVERSION")
    print("=" * 60)
    
    converter = ArabiziConverter()
    
    test_texts = [
        "ana n7eb el 5obz",  # 7->ح, 5->خ
        "3lech ma jitesh?",  # 3->ع
        "9a3ed nesta9rer",  # 9->ق
        "mta3 qalbi",  # qalb normalization
        "mta3 galbi",  # galb (bedouin) normalization
    ]
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        converted = converter.to_arabic(text)
        print(f"Converted: {converted}")
        
        # Qalb/galb normalization
        normalized, findings = converter.normalize_qaf_gaf_heart(text)
        if findings:
            print(f"Qalb/Galb normalized: {normalized}")
            for f in findings:
                print(f"  Style: {f.style}")


def demo_discourse_markers():
    """Demonstrate discourse marker detection."""
    
    print("\n" + "=" * 60)
    print("DISCOURSE MARKER DETECTION")
    print("=" * 60)
    
    detector = DiscourseMarkerDetector()
    
    test_sentences = [
        "تي شنوة تحكي؟",  # ti
        "ياخي صحيح؟",  # yaxxi
        "برا روح خدم",  # bara
        "مالا شنوة نعملو؟",  # mela
        "ترا ما تنساش",  # tra
    ]
    
    for text in test_sentences:
        print(f"\nText: {text}")
        markers = detector.find(text)
        print(f"  Markers: {markers}")


def demo_false_friends():
    """Demonstrate false friend detection."""
    
    print("\n" + "=" * 60)
    print("FALSE FRIEND DETECTION")
    print("=" * 60)
    print("(Words with different meanings in Tunisian vs MSA)")
    
    detector = FalseFriendDetector()
    
    test_sentences = [
        "مشى للدار",  # msha - go (Tunisian) vs walk (MSA)
        "روح للدار",  # rwah - go home (Tunisian) vs spirit (MSA)
        "لاباس عليك",  # labas - fine (Tunisian) vs no harm (MSA)
        "باهي مليح",  # bahi - ok/good (Tunisian) vs brilliant (MSA)
    ]
    
    for text in test_sentences:
        print(f"\nText: {text}")
        findings = detector.find(text)
        
        if findings:
            for f in findings:
                print(f"  Form: '{f.form}'")
                print(f"    Tunisian meaning: {f.tunisian_gloss}")
                print(f"    MSA meaning: {f.msa_gloss}")


def main():
    print("=" * 60)
    print("tun_linguist DEMO")
    print("Tunisian Derja Linguistic Analysis Toolkit")
    print("=" * 60)
    
    demo_dialect_scoring()
    demo_verb_analysis()
    demo_negation_analysis()
    demo_interrogatives()
    demo_arabizi_conversion()
    demo_discourse_markers()
    demo_false_friends()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
