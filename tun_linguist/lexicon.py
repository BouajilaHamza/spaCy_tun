from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final


@dataclass(frozen=True, slots=True)
class FalseFriendEntry:
    form: str
    tunisian_gloss: str
    msa_gloss: str


class FalseFriendDetector:
    """Detects high-salience Tunisian-vs-MSA semantic shifts (false friends)."""

    _entries: Final[dict[str, FalseFriendEntry]] = {
        "msha": FalseFriendEntry("msha", "to go (generic)", "to walk"),
        "rwaḥ": FalseFriendEntry("rwaḥ", "to go home", "wind/spirit"),
        "rwah": FalseFriendEntry("rwah", "to go home", "wind/spirit"),
        "labas": FalseFriendEntry("labas", "fine / greeting response", "no harm"),
        "bahi": FalseFriendEntry("bahi", "ok / good", "brilliant/showy"),
        # Arabic script variants (minimal set; extend as needed)
        "مشى": FalseFriendEntry("مشى", "to go (generic)", "to walk"),
        "روح": FalseFriendEntry("روح", "to go home", "wind/spirit"),
        "لاباس": FalseFriendEntry("لاباس", "fine / greeting response", "no harm"),
        "باهي": FalseFriendEntry("باهي", "ok / good", "brilliant/showy"),
    }

    _token_re: Final[re.Pattern[str]] = re.compile(r"\b\w+\b", re.UNICODE)

    def find(self, text: str) -> list[FalseFriendEntry]:
        out: list[FalseFriendEntry] = []
        for tok in self._token_re.findall(text):
            entry = self._entries.get(tok)
            if entry is not None:
                out.append(entry)
        return out


class DiscourseMarkerDetector:
    """Detect Tunisian discourse particles with very high Tunisian-ness weight."""

    # Latin + Arabic script, conservative boundaries.
    _markers_re: Final[re.Pattern[str]] = re.compile(
        r"(?i)(?:\bti\b|\byaxxi\b|\bya\s*xi\b|\btra\b|\bbara\b|\bmela\b|(?:^|\s)تي(?:\s|$)|(?:^|\s)ياخي(?:\s|$)|(?:^|\s)ترا(?:\s|$)|(?:^|\s)برا(?:\s|$)|(?:^|\s)مالا(?:\s|$))"
    )

    def find(self, text: str) -> list[str]:
        return [m.group(0).strip() for m in self._markers_re.finditer(text)]
