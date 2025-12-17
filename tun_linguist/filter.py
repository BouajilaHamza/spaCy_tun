from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Final

from .lexicon import DiscourseMarkerDetector, FalseFriendDetector
from .morphology import NounAnalyzer, VerbAnalyzer
from .normalizer import ArabiziConverter
from .syntax import NegationParser


@dataclass(frozen=True, slots=True)
class DialectScore:
    """Result container for dataset filtering decisions."""

    score: float
    positives: dict[str, int] = field(default_factory=dict)
    negatives: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def is_authentic(self, threshold: float = 5.0) -> bool:
        return self.score >= threshold


class DialectScorer:
    """Scores text for Tunisian dialect authenticity (high precision).

    Design goals:
    - Reward definitive Tunisian markers: bash/besh, ma...sh, mta3, discourse particles, n- (1sg) paradigm.
    - Penalize strong MSA markers: sawfa/سوف, sa- future prefix, lam/لم, lan/لن, laysa/ليس,
      MSA interrogatives (ماذا/لماذا/كيف/أين), and a-/أ- 1sg imperfective.
    - Avoid penalizing loanwords/substrate vocabulary; absence of penalties is the rule.

    The scoring model is deliberately transparent and rule-based for dataset curation.
    """

    _arabizi: Final[ArabiziConverter] = ArabiziConverter()
    _verbs: Final[VerbAnalyzer] = VerbAnalyzer()
    _nouns: Final[NounAnalyzer] = NounAnalyzer()
    _neg: Final[NegationParser] = NegationParser()
    _false_friends: Final[FalseFriendDetector] = FalseFriendDetector()
    _discourse: Final[DiscourseMarkerDetector] = DiscourseMarkerDetector()

    # --- Strong MSA markers (negative constraints) ---
    _msa_markers_re: Final[re.Pattern[str]] = re.compile(
        r"(?xi)(?:\b(?:sawfa|lan|lam|laysa|madha|limadha|kayfa|ayna)\b|(?:^|\s)(?:سوف|لن|لم|ليس|ماذا|لماذا|كيف|أين)(?:\s|$))"
    )

    # Arabic sa- future prefix: سـ + present tense verb (very heuristic).
    _msa_sa_future_re: Final[re.Pattern[str]] = re.compile(r"(?x)(?:^|\s)س(?=[\u0600-\u06FF]{3,})")

    # Tunisian positives.
    _tun_future_re: Final[re.Pattern[str]] = re.compile(r"(?xi)(?:\b(?:bash|besh)\b|(?:^|\s)(?:باش|بش)(?=\s))")
    _tun_genitive_re: Final[re.Pattern[str]] = re.compile(r"(?xi)(?:\bmta3\b|(?:^|\s)متاع(?:\s|$))")

    # Romance/Berber examples: explicitly *not* penalized (kept as a reminder / whitelist seed).
    _loanword_whitelist: Final[set[str]] = {"karhba", "krahib", "brika", "fakrun", "كرهبـة", "كراهب", "بريكة", "فكرون"}

    # Weights tuned for high precision: one definitive marker should typically dominate.
    _W_FUTURE_BASH: Final[float] = 4.0
    _W_GENITIVE_MTA3: Final[float] = 4.0
    _W_NEG_MA_SH: Final[float] = 4.5
    _W_DISCOURSE: Final[float] = 3.5
    _W_N_PREFIX_1SG: Final[float] = 2.5
    _W_N_PREFIX_1PL: Final[float] = 1.5
    _W_QA3ID: Final[float] = 2.0

    _P_MSA_MARKER: Final[float] = -6.0
    _P_MSA_SA_FUTURE: Final[float] = -5.0
    _P_MSA_A_PREFIX_VERB: Final[float] = -4.0

    def calculate_authenticity_score(self, text: str, *, normalize_arabizi_digits: bool = True) -> DialectScore:
        """Compute a rule-based authenticity score.

        Higher => more likely authentic Tunisian (Derja).
        """

        original = text
        if normalize_arabizi_digits:
            text = self._arabizi.to_arabic(text)

        positives: dict[str, int] = {}
        negatives: dict[str, int] = {}
        notes: list[str] = []
        score = 0.0

        # --- Negatives: strong MSA constraints ---
        msa_hits = list(self._msa_markers_re.finditer(text))
        if msa_hits:
            negatives["msa_marker"] = len(msa_hits)
            score += self._P_MSA_MARKER * len(msa_hits)

        sa_hits = list(self._msa_sa_future_re.finditer(text))
        if sa_hits:
            negatives["msa_sa_future_prefix"] = len(sa_hits)
            score += self._P_MSA_SA_FUTURE * len(sa_hits)

        # Verb-level penalties (a-/أ- imperfective 1sg).
        verb_findings = self._verbs.find_features(text)
        msa_a = [f for f in verb_findings if f.feature == "msa_a_prefix_imperfective"]
        if msa_a:
            negatives["msa_a_prefix_imperfective"] = len(msa_a)
            score += self._P_MSA_A_PREFIX_VERB * len(msa_a)

        # --- Positives: Tunisian constraints ---
        bash_hits = list(self._tun_future_re.finditer(text))
        if bash_hits:
            positives["future_bash"] = len(bash_hits)
            score += self._W_FUTURE_BASH * len(bash_hits)

        mta3_hits = list(self._tun_genitive_re.finditer(text))
        if mta3_hits:
            positives["genitive_mta3"] = len(mta3_hits)
            score += self._W_GENITIVE_MTA3 * len(mta3_hits)

        neg_findings = self._neg.find(text)
        ma_sh = [f for f in neg_findings if f.kind == "circumfix"]
        if ma_sh:
            positives["neg_ma_sh"] = len(ma_sh)
            score += self._W_NEG_MA_SH * len(ma_sh)

        # Discourse particles.
        discourse = self._discourse.find(text)
        if discourse:
            positives["discourse_marker"] = len(discourse)
            score += self._W_DISCOURSE * len(discourse)

        # Verb morphology: n- paradigm.
        n_1sg = [f for f in verb_findings if f.feature == "imperfective_paradigm" and f.person == "1sg"]
        n_1pl = [f for f in verb_findings if f.feature == "imperfective_paradigm" and f.person == "1pl"]
        qa3id = [f for f in verb_findings if f.feature == "progressive_qa3id"]

        if n_1sg:
            positives["n_prefix_1sg"] = len(n_1sg)
            score += self._W_N_PREFIX_1SG * len(n_1sg)
        if n_1pl:
            positives["n_prefix_1pl"] = len(n_1pl)
            score += self._W_N_PREFIX_1PL * len(n_1pl)
        if qa3id:
            positives["progressive_qa3id"] = len(qa3id)
            score += self._W_QA3ID * len(qa3id)

        # False friends: do not directly affect score; they are diagnostics for curation.
        ff = self._false_friends.find(text)
        if ff:
            notes.append(f"false_friends={','.join(e.form for e in ff)}")

        # Loanwords: explicitly preserved (no penalty). Add note only.
        lowered_tokens = {t.lower() for t in re.findall(r"\b\w+\b", text)}
        preserved = sorted(self._loanword_whitelist.intersection(lowered_tokens))
        if preserved:
            notes.append(f"loanwords_preserved={','.join(preserved)}")

        # Helpful note if Arabizi normalization changed text.
        if normalize_arabizi_digits and text != original:
            notes.append("arabizi_digits_normalized")

        return DialectScore(score=score, positives=positives, negatives=negatives, notes=notes)
