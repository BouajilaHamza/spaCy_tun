## Authenticity scoring

`DialectScorer` computes a transparent, rule-based score.

### Positives (Tunisian)

- **Future**: `bash/besh` (`باش/بش`)
- **Genitive**: `mta3` (`متاع`)
- **Negation**: `ma … sh/ch` (Arabic + Latin)
  - Extra bonus when **clitic trapping** is detected (e.g., `ma-qolt-el-ha-sh`)
- **Discourse markers** (very strong): `ti`, `yaxxi`, `tra`, `bara`, `mela`
- **Demonstratives**: `hadhouma/هاذوما`, `haka/هكا/هاكا`, `haki/هاكي`
- **Paradigm shift**: imperfective `n-` patterns (and contextual detection after `bash`/`qa3id`)

### Negatives (MSA constraints)

- `sawfa/سوف`, `lan/لن`, `lam/لم`, `laysa/ليس`
- Arabic future prefix `سـ` (heuristic)
- MSA 1sg imperfective `a-/أ-` (heuristic)

### Debugging

The result provides counts and notes:

```python
from tun_linguist import DialectScorer

r = DialectScorer().calculate_authenticity_score("ما قلت لهاش")
print(r.score)
print(r.positives)
print(r.negatives)
print(r.notes)
```
