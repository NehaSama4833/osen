"""python -m osen.experiment -> results/report.md"""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
from .contaminate import TYPES
from .corpus import split_by_module, stdlib_paragraphs
from .detectors import CharCosine, ExactNormalized, NgramOverlap, threshold_at_fpr

PER_TYPE, FPR = 25, 0.05


def run(seed):
    by_mod = stdlib_paragraphs()
    train, ev = split_by_module(by_mod, seed)
    rng = random.Random(seed)
    rng.shuffle(ev)
    vocab = sorted({w for d in train[:3000] for w in d.split()})[:5000]
    planted = {}
    pos = 0
    for name in TYPES:
        planted[name] = ev[pos:pos + PER_TYPE]; pos += PER_TYPE
    clean = ev[pos:]
    half = len(clean) // 2
    calib, test_clean = clean[:half], clean[half:]
    leaked_docs = []
    for name, fn in TYPES.items():
        leaked_docs += [fn(t, rng, vocab) for t in planted[name]]
    index_docs = train + leaked_docs
    dets = [ExactNormalized(), NgramOverlap(13), NgramOverlap(8), NgramOverlap(5), CharCosine()]
    rows, info = [], dict(n_train=len(train), n_leaked=len(leaked_docs), n_calib=len(calib), n_test_clean=len(test_clean))
    for d in dets:
        d.index(index_docs)
        cs = d.scores(calib)
        thr = threshold_at_fpr(cs, FPR) if cs.max() > 0 else 0.5
        thr = max(thr, 1e-9) if d.name.startswith("exact") else thr
        flags_clean = d.scores(test_clean) > thr
        row = {"detector": d.name, "thr": thr, "fpr_test": float(flags_clean.mean()),
               "calib_p95": float(np.quantile(cs, 0.95)), "calib_max": float(cs.max())}
        for name in TYPES:
            row[name] = float((d.scores(planted[name]) > thr).mean())
        rows.append(row)
    return rows, info


def main():
    out = Path("results"); out.mkdir(exist_ok=True)
    all_rows = [run(s) for s in (0, 1, 2)]
    info = all_rows[0][1]
    L = ["# osen results\n",
         "Corpus: docstring paragraphs (30-160 words) from the Python standard library, split BY MODULE into training text and eval text "
         f"(seed 0: {info['n_train']} training paragraphs). In each seed, {PER_TYPE} eval items per contamination type are planted into the training text "
         "(in the listed altered form); the remaining eval items are clean and are split in half: calibration (sets each threshold at 5% false flags) and test (reported). "
         "Averages over 3 module splits. Same-library docstrings share boilerplate, so clean items overlap with training text naturally.\n",
         "Detection rate on planted items at the calibrated threshold (mean of 3 seeds); last columns are the false-flag rate on clean test items "
         "and how much natural overlap clean calibration items already have.\n"]
    types = list(TYPES)
    L.append("| detector | " + " | ".join(types) + " | false flags (clean test) | calib 95th pct score | calib max score |")
    L.append("|---|" + "---|" * (len(types) + 3))
    for di in range(len(all_rows[0][0])):
        rs = [r[0][di] for r in all_rows]
        L.append(f"| {rs[0]['detector']} | " + " | ".join(f"{np.mean([r[t] for r in rs]):.0%}" for t in types) +
                 f" | {np.mean([r['fpr_test'] for r in rs]):.1%} | {np.mean([r['calib_p95'] for r in rs]):.2f} | {np.mean([r['calib_max'] for r in rs]):.2f} |")
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
