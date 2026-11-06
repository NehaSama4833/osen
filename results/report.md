# osen results

Corpus: docstring paragraphs (30-160 words) from the Python standard library, split BY MODULE into training text and eval text (seed 0: 999 training paragraphs). In each seed, 25 eval items per contamination type are planted into the training text (in the listed altered form); the remaining eval items are clean and are split in half: calibration (sets each threshold at 5% false flags) and test (reported). Averages over 3 module splits. Same-library docstrings share boilerplate, so clean items overlap with training text naturally.

Detection rate on planted items at the calibrated threshold (mean of 3 seeds); last columns are the false-flag rate on clean test items and how much natural overlap clean calibration items already have.

| detector | exact copy | case+punctuation+spacing | 10% words replaced | 30% words replaced | first half only | sentences reordered | embedded in longer text | false flags (clean test) | calib 95th pct score | calib max score |
|---|---|---|---|---|---|---|---|---|---|---|
| exact (normalised) | 100% | 32% | 0% | 0% | 0% | 0% | 0% | 0.2% | 0.00 | 0.67 |
| 13-gram overlap | 100% | 85% | 89% | 9% | 100% | 85% | 100% | 5.3% | 0.04 | 1.00 |
| 8-gram overlap | 100% | 93% | 99% | 35% | 100% | 100% | 100% | 6.8% | 0.10 | 1.00 |
| 5-gram overlap | 100% | 93% | 100% | 51% | 99% | 100% | 100% | 6.8% | 0.21 | 1.00 |
| char-5gram cosine (nearest doc) | 100% | 97% | 100% | 61% | 100% | 100% | 24% | 5.9% | 0.51 | 0.93 |