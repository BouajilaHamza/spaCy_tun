## Linguistic features

`tun_linguist` focuses on **high-signal discriminators** from Tunisian Derja.

### Verbal morphology (`VerbAnalyzer`)

- **Paradigm shift (critical)**: imperfective **`n-`** prefix as 1sg vs 1pl heuristics.
- **Future**: `bash/besh` (`باش/بش`).
- **Progressive**: `qa3id` (`قاعد/قاعدة/قاعدين`).

### Nominal markers (`NounAnalyzer`)

- **Genitive / possession**: `mta3` (`متاع`).

### Syntax (`NegationParser`, `InterrogativeParser`)

- **Negation**: `ma … sh/ch` circumfix (Arabic + Latin) including **clitic trapping**.
- **Pseudo-verbs**: contracted negation forms like `مش/موش`.
- **Interrogatives**: Tunisian wh particles and a wh-in-situ heuristic.

### Lexicon diagnostics

- **Discourse particles**: `ti`, `yaxxi`, `tra`, `bara`, `mela` (+ Arabic-script forms).
- **False friends**: small, high-salience Tunisian-vs-MSA semantic shift list for curation diagnostics.
