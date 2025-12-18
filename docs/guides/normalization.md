## Normalization

`tun_linguist` keeps normalization **conservative**: it fixes highly disruptive artifacts while avoiding aggressive rewriting.

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
