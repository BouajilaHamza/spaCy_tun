## Dataset pipeline (scripts)

This repository includes scripts that use `tun_linguist` for dataset generation.

### Demo

```bash
python scripts/demo_tun_linguist.py
```

### Process dataset

```bash
python scripts/process_derja_dataset.py \
  --output-dir data/processed \
  --sample-size 100000 \
  --max-per-config 30000 \
  --spo-pairs 10000 \
  --contrastive-pairs 20000
```

### Score MT outputs

```bash
python scripts/score_mt_outputs.py --interactive
python scripts/score_mt_outputs.py --input mt_outputs.jsonl --output scored.jsonl
python scripts/score_mt_outputs.py --compare system1.jsonl system2.jsonl
```
