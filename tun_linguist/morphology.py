from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final, Iterable


_ARABIC_LETTERS: Final[str] = "\u0600-\u06FF"


@dataclass(frozen=True, slots=True)
class VerbFinding:
    """A single detected verbal morphology feature."""

    span: tuple[int, int]
    surface: str
    feature: str
    person: str | None = None  # "1sg" | "1pl" | None
    script: str | None = None  # "arabic" | "latin" | None


class VerbAnalyzer:
    """Detects key Tunisian verbal morphology markers.

    Primary goal: high-precision authenticity features, not full parsing.

    Implemented discriminators:
    - Imperfective 1sg: n- prefix (Tunisian) vs a- (MSA) [critical]
    - Imperfective 1pl: n- ... -u (often -u/-ou/-w or Arabic suffix و/وا)
    - Future: bash/besh (باش/بش)
    - Progressive: qa3id variants (قاعد/قاعدة/قاعدين ; qa3id/qa3da/qa3din)
    """

    # Token-ish boundary matcher. We avoid heavy tokenization dependencies.
    _token_re: Final[re.Pattern[str]] = re.compile(rf"\b[\w{_ARABIC_LETTERS}]+\b", re.UNICODE)

    # --- Tunisian future marker (definitive dialect marker) ---
    _future_re: Final[re.Pattern[str]] = re.compile(
        r"(?xi)(?:\b(?:bash|besh)\b|(?:^|\s)(?:باش|بش)(?=\s))"
    )

    # --- Progressive auxiliary qa3id variants ---
    _progressive_re: Final[re.Pattern[str]] = re.compile(
        r"(?xi)(?:\b(?:qa3id|qa3da|qa3din)\b|(?:^|\s)(?:قاعد|قاعدة|قاعدين)(?=\s))"
    )

    # --- Imperfective paradigm shift ---
    # Tunisian 1pl often ends with -u/-ou or Arabic suffix -و / -وا.
    _tun_1pl_latin_re: Final[re.Pattern[str]] = re.compile(r"(?xi)\b(?:n)[a-z]{2,}(?:u|ou|w)\b")
    _tun_1pl_ar_re: Final[re.Pattern[str]] = re.compile(rf"(?x)(?:^|\s)(ن[{_ARABIC_LETTERS}]{{2,}}(?:وا|و))\b")

    # Tunisian 1sg: starts with n- and does NOT end with plural marker.
    # This is intentionally conservative to reduce false positives.
    _tun_1sg_latin_re: Final[re.Pattern[str]] = re.compile(
        r"(?xi)\b(?:n)[a-z]{2,}(?<!u)(?<!ou)(?<!w)\b"
    )
    _tun_1sg_ar_re: Final[re.Pattern[str]] = re.compile(
        rf"(?x)(?:^|\s)(ن[{_ARABIC_LETTERS}]{{2,}})(?<!وا)(?<!و)\b"
    )

    # MSA 1sg imperfective: a- (Latin) or أ- (Arabic). We keep this high precision.
    # Note: we do NOT attempt to detect all a- verbs; we target canonical MSA patterns.
    _msa_1sg_latin_re: Final[re.Pattern[str]] = re.compile(r"(?xi)\b(?:a)[a-z]{3,}\b")
    _msa_1sg_ar_re: Final[re.Pattern[str]] = re.compile(rf"(?x)(?:^|\s)(أ[{_ARABIC_LETTERS}]{{3,}})\b")

    # K-T-B exemplar (explicit): nektib vs nektbu vs aktubu
    _ktb_exemplar_re: Final[re.Pattern[str]] = re.compile(
        r"(?xi)\b(?:(?P<tun1sg>nektib|nektb)|(?P<tun1pl>nektbu)|(?P<msa1sg>aktubu))\b"
    )

    # --- Context-aware imperfective detection (Arabic script + Latin) ---
    # This is substantially higher precision for Arabic script, where ن- alone is ambiguous (MSA "we" vs Tunisian "I").
    _after_future_capture_re: Final[re.Pattern[str]] = re.compile(
        rf"(?xi)(?:\b(?:bash|besh)\b|(?:^|\s)(?:باش|بش))\s+(?P<verb>[\w{_ARABIC_LETTERS}]+)"
    )
    _after_progressive_capture_re: Final[re.Pattern[str]] = re.compile(
        rf"(?xi)(?:\b(?:qa3id|qa3da|qa3din)\b|(?:^|\s)(?:قاعد|قاعدة|قاعدين))\s+(?P<verb>[\w{_ARABIC_LETTERS}]+)"
    )

    # Arabic imperfective prefixes: أ/ن/ت/ي . We only need أ vs ن here.
    _arabic_imperfective_prefix_re: Final[re.Pattern[str]] = re.compile(r"^[أنتي]")

    def _classify_imperfective_token(self, token: str, span: tuple[int, int]) -> VerbFinding | None:
        """Classify a single verb-like token by its imperfective prefix/suffix heuristics."""
        if not token:
            return None

        # Arabic script
        first = token[0]
        if "\u0600" <= first <= "\u06FF":
            if not self._arabic_imperfective_prefix_re.match(token):
                return None
            if first == "أ" and len(token) >= 4:
                return VerbFinding(span, token, feature="msa_a_prefix_imperfective", person="1sg", script="arabic")
            if first == "ن" and len(token) >= 3:
                if token.endswith(("وا", "و")):
                    return VerbFinding(span, token, feature="imperfective_paradigm_contextual", person="1pl", script="arabic")
                return VerbFinding(span, token, feature="imperfective_paradigm_contextual", person="1sg", script="arabic")
            return None

        # Latin script (Arabizi-ish without digits)
        low = token.lower()
        if low.startswith("a") and len(low) >= 4:
            return VerbFinding(span, token, feature="msa_a_prefix_imperfective", person="1sg", script="latin")
        if low.startswith("n") and len(low) >= 3:
            if low.endswith(("u", "ou", "w")):
                return VerbFinding(span, token, feature="imperfective_paradigm_contextual", person="1pl", script="latin")
            return VerbFinding(span, token, feature="imperfective_paradigm_contextual", person="1sg", script="latin")
        return None

    def find_features(self, text: str) -> list[VerbFinding]:
        findings: list[VerbFinding] = []

        # Exemplar first (highest precision, explicit requirement).
        for m in self._ktb_exemplar_re.finditer(text):
            if m.group("tun1sg") is not None:
                findings.append(
                    VerbFinding(m.span(), m.group(0), feature="imperfective_paradigm", person="1sg", script="latin")
                )
            elif m.group("tun1pl") is not None:
                findings.append(
                    VerbFinding(m.span(), m.group(0), feature="imperfective_paradigm", person="1pl", script="latin")
                )
            elif m.group("msa1sg") is not None:
                findings.append(
                    VerbFinding(m.span(), m.group(0), feature="msa_a_prefix_imperfective", person="1sg", script="latin")
                )

        # Future marker.
        for m in self._future_re.finditer(text):
            findings.append(VerbFinding(m.span(), m.group(0).strip(), feature="future_bash"))

        # Progressive marker.
        for m in self._progressive_re.finditer(text):
            findings.append(VerbFinding(m.span(), m.group(0).strip(), feature="progressive_qa3id"))

        # Context-aware imperfective: verb immediately after bash/besh or qa3id (high precision for Arabic script).
        for m in self._after_future_capture_re.finditer(text):
            tok = m.group("verb")
            vf = self._classify_imperfective_token(tok, m.span("verb"))
            if vf is not None:
                findings.append(vf)
        for m in self._after_progressive_capture_re.finditer(text):
            tok = m.group("verb")
            vf = self._classify_imperfective_token(tok, m.span("verb"))
            if vf is not None:
                findings.append(vf)

        # General imperfective detection (heuristic).
        for m in self._tun_1pl_latin_re.finditer(text):
            findings.append(
                VerbFinding(m.span(), m.group(0), feature="imperfective_paradigm", person="1pl", script="latin")
            )
        for m in self._tun_1pl_ar_re.finditer(text):
            findings.append(
                VerbFinding(m.span(1), m.group(1), feature="imperfective_paradigm", person="1pl", script="arabic")
            )

        for m in self._tun_1sg_latin_re.finditer(text):
            # prevent double-counting 1pl latin already caught
            tok = m.group(0)
            if self._tun_1pl_latin_re.fullmatch(tok):
                continue
            findings.append(
                VerbFinding(m.span(), tok, feature="imperfective_paradigm", person="1sg", script="latin")
            )
        for m in self._tun_1sg_ar_re.finditer(text):
            tok = m.group(1)
            if self._tun_1pl_ar_re.search(tok) is not None:
                continue
            findings.append(
                VerbFinding(m.span(1), tok, feature="imperfective_paradigm", person="1sg", script="arabic")
            )

        # MSA a-/أ- prefix (low recall by design).
        # We de-duplicate obvious false positives by excluding common Tunisian markers.
        for m in self._msa_1sg_ar_re.finditer(text):
            findings.append(
                VerbFinding(m.span(1), m.group(1), feature="msa_a_prefix_imperfective", person="1sg", script="arabic")
            )

        for m in self._msa_1sg_latin_re.finditer(text):
            tok = m.group(0)
            # Avoid penalizing bash/besh/qa3id etc.
            if tok.lower() in {"bash", "besh", "qa3id", "qa3da", "qa3din"}:
                continue
            findings.append(
                VerbFinding(m.span(), tok, feature="msa_a_prefix_imperfective", person="1sg", script="latin")
            )

        # Stable ordering.
        return sorted(findings, key=lambda f: (f.span[0], f.span[1], f.feature))


@dataclass(frozen=True, slots=True)
class NounFinding:
    span: tuple[int, int]
    surface: str
    feature: str


class NounAnalyzer:
    """Detects nominal morphology markers useful for dataset filtering."""

    # Possession particle mta3 / متاع (high precision).
    _mta3_re: Final[re.Pattern[str]] = re.compile(
        r"(?xi)(?:\bmta3\b|(?:^|\s)متاع(?:\s|$))"
    )

    # Sound masculine plural marker: -in / ـين. (Not exclusive to Tunisian, but useful.)
    _smp_in_ar_re: Final[re.Pattern[str]] = re.compile(rf"(?x)\b([{_ARABIC_LETTERS}]{{2,}}ين)\b")
    _smp_un_ar_re: Final[re.Pattern[str]] = re.compile(rf"(?x)\b([{_ARABIC_LETTERS}]{{2,}}ون)\b")
    _smp_in_latin_re: Final[re.Pattern[str]] = re.compile(r"(?xi)\b[a-z]{2,}in\b")
    _smp_un_latin_re: Final[re.Pattern[str]] = re.compile(r"(?xi)\b[a-z]{2,}un\b")

    def find_features(self, text: str) -> list[NounFinding]:
        out: list[NounFinding] = []

        for m in self._mta3_re.finditer(text):
            out.append(NounFinding(m.span(), m.group(0).strip(), feature="genitive_mta3"))

        # Prefer -in over -un when both occur; we emit both so downstream scoring can decide.
        for m in self._smp_in_ar_re.finditer(text):
            out.append(NounFinding(m.span(1), m.group(1), feature="plural_in"))
        for m in self._smp_un_ar_re.finditer(text):
            out.append(NounFinding(m.span(1), m.group(1), feature="plural_un"))
        for m in self._smp_in_latin_re.finditer(text):
            out.append(NounFinding(m.span(), m.group(0), feature="plural_in"))
        for m in self._smp_un_latin_re.finditer(text):
            out.append(NounFinding(m.span(), m.group(0), feature="plural_un"))

        return sorted(out, key=lambda f: (f.span[0], f.span[1], f.feature))

    def has_mta3(self, text: str) -> bool:
        return self._mta3_re.search(text) is not None


def iter_tokens(text: str) -> Iterable[str]:
    """Simple unicode token iterator used by multiple modules."""
    token_re: Final[re.Pattern[str]] = re.compile(rf"\b[\w{_ARABIC_LETTERS}]+\b", re.UNICODE)
    yield from token_re.findall(text)
