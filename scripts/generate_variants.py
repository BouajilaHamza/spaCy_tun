from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from datasets import load_dataset


AR_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
TATWEEL_RE = re.compile(r"\u0640")
MULTISPACE_RE = re.compile(r"\s{2,}")

# Western <-> Arabic-Indic digits
WEST_TO_AR_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
AR_INDIC_TO_WEST = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


@dataclass(frozen=True, slots=True)
class Variant:
    english: str
    tunisian_original: str
    tunisian_variant: str
    variant_type: str


def strip_diacritics(text: str) -> str:
    text = AR_DIACRITICS_RE.sub("", text)
    text = TATWEEL_RE.sub("", text)
    return text


def normalize_spaces(text: str) -> str:
    text = text.replace("\u00A0", " ")
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def normalize_punct(text: str) -> str:
    # normalize some common punctuation spacing
    text = re.sub(r"\s*،\s*", "، ", text)
    text = re.sub(r"\s*\?\s*", "؟ ", text)  # prefer Arabic question mark
    text = re.sub(r"\s*؟\s*", "؟ ", text)
    text = re.sub(r"\s*\.\s*", ". ", text)
    return normalize_spaces(text)


def contract_bash(text: str) -> str:
    # باش -> بش as a conservative, very common reduction
    return re.sub(r"(?<!\S)باش(?!\S)", "بش", text)


def expand_besh(text: str) -> str:
    # بش -> باش (opposite direction) when standalone
    return re.sub(r"(?<!\S)بش(?!\S)", "باش", text)


def split_greeting(text: str) -> str:
    # مسالخير/مساءالخير -> مسا الخير
    text = re.sub(r"(?<!\S)مسالخير(?!\S)", "مسا الخير", text)
    text = re.sub(r"(?<!\S)مساء\s*الخير(?!\S)", "مسا الخير", text)
    return text


def normalize_shnawa(text: str) -> str:
    # purely orthographic: شنوّة/شنيا/شنوة -> شنوة (doesn't change meaning)
    text = text.replace("شنوّة", "شنوة")
    text = text.replace("شنيا", "شنوة")
    return text


def digits_to_arabic_indic(text: str) -> str:
    return text.translate(WEST_TO_AR_INDIC)


def digits_to_western(text: str) -> str:
    return text.translate(AR_INDIC_TO_WEST)


def strip_leading_label(text: str) -> str:
    # Some rows can accidentally include a "Tunisian:" prefix.
    return re.sub(r"^\s*Tunisian:\s*", "", text, flags=re.IGNORECASE)


TRANSFORMS: list[tuple[str, Callable[[str], str]]] = [
    ("strip_diacritics", strip_diacritics),
    ("normalize_punct", normalize_punct),
    ("normalize_spaces", normalize_spaces),
    ("contract_bash", contract_bash),
    ("expand_besh", expand_besh),
    ("split_greeting", split_greeting),
    ("normalize_shnawa", normalize_shnawa),
    ("digits_to_arabic_indic", digits_to_arabic_indic),
    ("digits_to_western", digits_to_western),
    ("strip_leading_label", strip_leading_label),
]


def apply_chain(text: str, chain: Iterable[tuple[str, Callable[[str], str]]]) -> tuple[str, list[str]]:
    kinds: list[str] = []
    out = text
    for kind, fn in chain:
        out2 = fn(out)
        if out2 != out:
            kinds.append(kind)
        out = out2
    return out, kinds


def make_variants_for_row(tunisian: str, *, rng: random.Random, target: int = 2) -> list[tuple[str, str]]:
    """Return up to `target` (variant_text, variant_type) pairs."""

    variants: list[tuple[str, str]] = []
    seen: set[str] = {tunisian}

    # Try multiple random chains; keep only meaning-preserving orthographic changes.
    for _ in range(40):
        chain_len = rng.choice([1, 2, 2, 3])
        chain = rng.sample(TRANSFORMS, k=chain_len)
        out, kinds = apply_chain(tunisian, chain)
        out = normalize_spaces(out)
        if not kinds:
            continue
        if out in seen:
            continue
        # avoid over-short outputs
        if len(out) < 3:
            continue
        variants.append((out, "+".join(kinds)))
        seen.add(out)
        if len(variants) >= target:
            break

    return variants


def main() -> None:
    rng = random.Random(42)
    ds = load_dataset("NadiaGHEZAIEL/English_to_Tunisian_Dataset")
    train = ds["train"]

    # Choose 25 rows deterministically -> 50 variants
    idxs = rng.sample(range(len(train)), 25)

    out_path = Path("/workspace/data/variants_50.jsonl")
    out: list[Variant] = []

    for i in idxs:
        row = train[int(i)]
        eng = str(row["English"]).strip()
        tun = str(row["Tunisian"]).strip()

        for v_text, v_kind in make_variants_for_row(tun, rng=rng, target=2):
            out.append(
                Variant(
                    english=eng,
                    tunisian_original=tun,
                    tunisian_variant=v_text,
                    variant_type=v_kind,
                )
            )

    # Ensure exactly 50 (top-up if a few rows produced <2)
    if len(out) < 50:
        # add more rows until we reach 50
        remaining = [j for j in range(len(train)) if j not in idxs]
        for j in remaining:
            if len(out) >= 50:
                break
            row = train[int(j)]
            eng = str(row["English"]).strip()
            tun = str(row["Tunisian"]).strip()
            for v_text, v_kind in make_variants_for_row(tun, rng=rng, target=1):
                out.append(Variant(english=eng, tunisian_original=tun, tunisian_variant=v_text, variant_type=v_kind))
                if len(out) >= 50:
                    break

    out = out[:50]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for v in out:
            f.write(
                json.dumps(
                    {
                        "English": v.english,
                        "Tunisian_original": v.tunisian_original,
                        "Tunisian_variant": v.tunisian_variant,
                        "variant_type": v.variant_type,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {len(out)} variants to {out_path}")


if __name__ == "__main__":
    main()
