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


@dataclass(frozen=True, slots=True)
class PhonologyFinding:
    """A detected phonological variant with a normalized lemma and a style tag."""

    span: tuple[int, int]
    surface: str
    lemma: str
    style: str  # "urban_q" | "bedouin_g" | "unknown"


class PhonologyNormalizer:
    """Phonological unification helpers for Tunisian Derja.

    This module targets *high-precision* cases that matter for dataset filtering,
    retrieval, and scoring. It intentionally does not attempt global q<->g rewriting.

    Currently implemented:
    - Qaf/Gaf unification for the exemplar pair "heart": qalb/galb/9alb and Arabic-script variants.
    """

    # Heart exemplar:
    # - Latin/Arabizi: qalb, qelb, galb, gelb, 9alb, 9elb
    # - Arabic script: قلب (qaf) vs ڨلب/گلب/ݣلب (gaf variants)
    _heart_re: Final[re.Pattern[str]] = re.compile(
        r"(?xi)"
        r"(?P<latin>\b(?P<latin_form>(?:q|g|9)(?:a|e)?lb)\b)"
        r"|(?P<arabic>(?<![\w\u0600-\u06FF])(?P<ar_form>[قڨگݣ]لب)(?![\w\u0600-\u06FF]))"
    )

    def tag_qaf_gaf_heart(self, text: str) -> list[PhonologyFinding]:
        """Return style-tagged findings for qalb/galb variants without changing text."""
        out: list[PhonologyFinding] = []
        for m in self._heart_re.finditer(text):
            surface = m.group(0)
            style = "unknown"
            if m.group("latin") is not None:
                lf = m.group("latin_form").lower()
                if lf.startswith("g"):
                    style = "bedouin_g"
                elif lf.startswith(("q", "9")):
                    style = "urban_q"
            elif m.group("arabic") is not None:
                ar = m.group("ar_form")
                if ar.startswith("ق"):
                    style = "urban_q"
                elif ar[0] in {"ڨ", "گ", "ݣ"}:
                    style = "bedouin_g"

            out.append(
                PhonologyFinding(
                    span=m.span(),
                    surface=surface,
                    lemma="qalb",
                    style=style,
                )
            )
        return out

    def normalize_qaf_gaf_heart(self, text: str) -> tuple[str, list[NormalizedQafGaf]]:
        """Normalize qalb/galb variants to the latin lemma "qalb" while tagging style.

        This is designed for semantic *lemma matching* (e.g., retrieval or grouping).
        """
        findings: list[NormalizedQafGaf] = []

        def _sub(m: re.Match[str]) -> str:
            surface = m.group(0)
            style = "unknown"
            if m.group("latin") is not None:
                low = surface.lower()
                if low.startswith("g"):
                    style = "bedouin_g"
                elif low.startswith(("q", "9")):
                    style = "urban_q"
            elif m.group("arabic") is not None:
                if surface.startswith("ق"):
                    style = "urban_q"
                elif surface[0] in {"ڨ", "گ", "ݣ"}:
                    style = "bedouin_g"
            findings.append(NormalizedQafGaf(lemma="qalb", style=style))
            return "qalb"

        return self._heart_re.sub(_sub, text), findings


class ArabiziConverter:
    """Lightweight Arabizi -> Arabic script converter.

    Notes:
    - This is intentionally conservative: it focuses on high-precision mappings
      needed for downstream filtering, not full orthographic reconstruction.
    """

    _digits_re: Final[re.Pattern[str]] = re.compile(r"[3579]")

    _phonology: Final[PhonologyNormalizer] = PhonologyNormalizer()

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
        return self._phonology.normalize_qaf_gaf_heart(text)
