# osen — 汚染

**汚染 (osen)** means "contamination". If evaluation questions leaked into training data, benchmark scores are inflated. Detectors usually check word n-gram overlap. How well do simple detectors catch leaks that were slightly altered, and how often do they raise false alarms? This project plants known leaks into real text and measures it.

## Setup
Real technical prose: 30-160-word docstring paragraphs from the Python standard library (parsed with `ast`, not imported), split **by module** into training text (about 1000 paragraphs) and eval text. In each of 3 seeds, 25 eval items per leak type are planted into the training text in altered form: exact copy, case/punctuation/spacing changes, 10% and 30% of words replaced, first half only, sentences reordered, embedded in longer text. Each detector's threshold is calibrated on clean eval items to 5% false flags; reported on separate clean test items.

## Results (mean of 3 seeds; `results/report.md`)
| detector | exact | case+punct | 10% words | 30% words | half only | reordered | embedded | false flags |
|---|---|---|---|---|---|---|---|---|
| exact (normalised) | 100% | 32% | 0% | 0% | 0% | 0% | 0% | 0.2% |
| 13-gram overlap | 100% | 85% | 89% | **9%** | 100% | 85% | 100% | 5.3% |
| 8-gram overlap | 100% | 93% | 99% | 35% | 100% | 100% | 100% | 6.8% |
| 5-gram overlap | 100% | 93% | 100% | 51% | 99% | 100% | 100% | 6.8% |
| char-5gram cosine | 100% | 97% | 100% | 61% | 100% | 100% | **24%** | 5.9% |

- Exact matching catches only literal copies. The standard 13-gram overlap misses 91% of leaks with 30% of words changed; shorter n-grams and character similarity do better but never reach full recall at 5% false flags.
- Char-cosine fails on text embedded in longer text (the extra text dilutes similarity); n-gram overlap handles that perfectly. No single detector wins every case.
- Clean eval items already overlap training text (some calibration items score 1.00): docstrings in one library repeat boilerplate and sometimes whole paragraphs. So **an overlap hit is not proof of leakage**; thresholds need a clean calibration set, as done here.
- The "case+punct" transform also strips in-word punctuation (`e.g.` becomes `eg`), which changes the word tokens; that is why the normalised exact match catches only 32%.

## Caveats
Only about 1000 paragraphs and 25 items per type; leaks are synthetic edits, not real model memorization; no paraphrase or translation leaks; a detector on training text, not a probe of a model. A first version of the reordering transform did nothing to single-sentence items (27% of them) and made the exact detector look better than it was; it was fixed and the test now checks for a changed text.

## Run
```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q          # 8 tests
.venv/bin/python -m osen.experiment    # ~20 s
```
