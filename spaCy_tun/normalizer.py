from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final


_ARABIZI_DIGIT_MAP: Final[dict[str, str]] = {
    "3": "ع",  # ayin
    "7": "ح",  # ha (emphatic/voiceless pharyngeal)
    "9": "ق",  # qaf
    "5": "خ",  # kha
}


@dataclass(frozen=True, slots=True)
class NormalizedQafGaf:
    """Represents a qaf/gaf-normalized lemma with a style tag."""

    lemma: str
    style: str  # "urban_q" | "bedouin_g" | "unknown"


class ArabiziConverter:
    """Lightweight Arabizi -> Arabic script converter.

    Notes:
    - This is intentionally conservative: it focuses on high-precision mappings
      needed for downstream filtering, not full orthographic reconstruction.
    """

    _digits_re: Final[re.Pattern[str]] = re.compile(r"[3579]")

    # Bedouin /g/ can be written as Latin 'g' (Arabizi) or Arabic 'ڨ'/'گ' or even 'ق'.
    # We keep the input char in text, but provide a helper for semantic normalization.
    _qaf_gaf_heart_re: Final[re.Pattern[str]] = re.compile(
        r"\b(?P<form>qalb|galb)\b", re.IGNORECASE
    )

    def to_arabic(self, text: str) -> str:
        """Convert common Arabizi digits (3/7/9/5) to Arabic letters.

        This is designed for dataset normalization pipelines where digits are
        the most disruptive tokenization artifacts.
        """

        def _sub(m: re.Match[str]) -> str:
            return _ARABIZI_DIGIT_MAP[m.group(0)]

        return self._digits_re.sub(_sub, text)

    def normalize_qaf_gaf_heart(self, text: str) -> tuple[str, list[NormalizedQafGaf]]:
        """Normalize qalb/galb to a shared lemma while tagging the variation.

        Returns:
        - normalized_text: with qalb/galb replaced by "qalb" (latin lemma) in-place
        - findings: list of NormalizedQafGaf with style tags.

        This is a narrow, *semantic lemma* normalization for the exemplar pair
        (qalb vs galb). It does not attempt global q<->g rewriting.
        """

        findings: list[NormalizedQafGaf] = []

        def _sub(m: re.Match[str]) -> str:
            form = m.group("form")
            if form.lower().startswith("g"):
                findings.append(NormalizedQafGaf(lemma="qalb", style="bedouin_g"))
            elif form.lower().startswith("q"):
                findings.append(NormalizedQafGaf(lemma="qalb", style="urban_q"))
            else:
                findings.append(NormalizedQafGaf(lemma="qalb", style="unknown"))
            return "qalb"

        return self._qaf_gaf_heart_re.sub(_sub, text), findings
