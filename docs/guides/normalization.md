## Normalization

`tun_linguist` keeps normalization **conservative**: it fixes highly disruptive artifacts while avoiding aggressive rewriting.

!!! warning "Not a full transliteration system"
    The goal is *dataset curation*, not perfect orthography. If you need full Arabizi→Arabic reconstruction, you should add a dedicated transliteration component upstream.

### Arabizi digits (`ArabiziConverter.to_arabic`)

Converts common digits used as letters:

- `3 → ع`
- `7 → ح`
- `9 → ق`
- `5 → خ`

This is intended to reduce tokenization noise, not to fully convert Arabizi to Arabic script.

### Phonological unification (Qaf/Gaf)

Some Tunisian varieties realize **ق** as **g** (often written in Arabic script as `ڨ/گ/ݣ` or in Latin as `g`).

`tun_linguist` provides a **high-precision exemplar unifier** for the “heart” pair:

- `qalb / galb / 9alb` and `قلب / ڨلب / گلب / ݣلب`

Use:

```python
from tun_linguist import PhonologyNormalizer

pn = PhonologyNormalizer()
print(pn.tag_qaf_gaf_heart("ڨلبو كبير"))
```

The scorer records these variants in `DialectScore.notes` (non-destructive).

### When should I use phonology normalization?

- **For scoring**: by default the scorer only *tags* these variants (it does not rewrite your text).
- **For retrieval/grouping**: use `normalize_qaf_gaf_heart(...)` if you want variants to share a common lemma (`qalb`) while still keeping a style tag.
