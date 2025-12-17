from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final


@dataclass(frozen=True, slots=True)
class NegationFinding:
    span: tuple[int, int]
    text: str
    kind: str  # "circumfix" | "pseudo_verb"


class NegationParser:
    """Detect Tunisian negation patterns.

    Focus: the ma...sh circumfix, including clitic trapping.

    Regex notes:
    - We allow optional hyphens and whitespace used in informal spelling.
    - We allow a limited internal window to avoid runaway matches.
    """

    # --- Core ma...sh circumfix (Arabic script + Latin) ---
    # Arabic: ما + (token-ish content, including clitics like -لُه-) + ش
    # Latin: ma + ... + ch/sh (Tunisian often uses 'ch' in Latin; we accept sh too)
    #
    # Handles: "ما مشيتش", "ما-كتبتلوش", "ma mshit-ch", "ma ktabt-luh-sh"
    MA_SH_CIRCUMFIX_RE: Final[re.Pattern[str]] = re.compile(
        r"(?xi)\b(?:ما|ma)\s*[-–—]?\s*"  # ma / ما
        r"(?P<body>"  # the negated predicate (with possible trapped clitics)
        r"[\w\u0600-\u06FF]+"  # main stem chunk
        r"(?:\s*[-–—]?\s*[\w\u0600-\u06FF]+){0,3}"  # up to 3 extra chunks
        r")\s*[-–—]?\s*"  # optional separator before -sh/-ch
        r"(?:ش|ch|sh)\b"  # neg suffix
    )

    # --- Pseudo-verbs: negated pronouns (Arabic script + common Latin) ---
    # Arabic: ما ني ش / ماكش / ماهوش / ماهيش / ماناش + also "مش/موش"
    # Latin: manish/maksh/mahush/mahish/manash + mish/mush/mouch/mosh
    PSEUDO_VERB_RE: Final[re.Pattern[str]] = re.compile(
        r"(?xi)\b(?:"  # word boundary
        r"(?:ما\s*[-–—]?\s*ني\s*[-–—]?\s*ش)"  # ما ني ش
        r"|(?:ما\s*[-–—]?\s*ك\s*[-–—]?\s*ش)"  # ما ك ش
        r"|(?:ما\s*[-–—]?\s*هو\s*[-–—]?\s*ش)"  # ما هو ش
        r"|(?:ما\s*[-–—]?\s*هي\s*[-–—]?\s*ش)"  # ما هي ش
        r"|(?:ما\s*[-–—]?\s*نا\s*[-–—]?\s*ش)"  # ما نا ش
        r"|(?:مش|موش|مِش|موش)"  # contracted forms
        r"|(?:manish|maksh|mahush|mahish|manash|mish|mush|mosh|mouch)"
        r")\b"
    )

    def find(self, text: str) -> list[NegationFinding]:
        out: list[NegationFinding] = []
        for m in self.MA_SH_CIRCUMFIX_RE.finditer(text):
            out.append(NegationFinding(span=m.span(), text=m.group(0), kind="circumfix"))
        for m in self.PSEUDO_VERB_RE.finditer(text):
            out.append(NegationFinding(span=m.span(), text=m.group(0), kind="pseudo_verb"))
        return sorted(out, key=lambda f: f.span[0])


class InterrogativeParser:
    """Detect Tunisian interrogatives and wh-in-situ usage."""

    _wh_re: Final[re.Pattern[str]] = re.compile(
        r"(?xi)(?:\bwin\b|\bwaqtash\b|\bshkun\b|\bkifash\b|\b3lash\b|\bshnowa\b|\bshnuwa\b|(?:^|\s)وين(?:\s|$)|(?:^|\s)وقتاش(?:\s|$)|(?:^|\s)شكون(?:\s|$)|(?:^|\s)كيفاش(?:\s|$)|(?:^|\s)علاش(?:\s|$)|(?:^|\s)شنوة(?:\s|$)|(?:^|\s)شنو(?:\s|$))"
    )

    # wh-in-situ heuristic: question word appears near end before punctuation/end.
    _wh_in_situ_re: Final[re.Pattern[str]] = re.compile(
        r"(?xi)\b(?:win|waqtash|shkun|kifash|3lash|shnowa|shnuwa|وين|وقتاش|شكون|كيفاش|علاش|شنوة|شنو)\b\s*[?!…\.\)]*\s*$"
    )

    def find_particles(self, text: str) -> list[str]:
        return [m.group(0).strip() for m in self._wh_re.finditer(text)]

    def is_wh_in_situ(self, text: str) -> bool:
        return self._wh_in_situ_re.search(text) is not None
