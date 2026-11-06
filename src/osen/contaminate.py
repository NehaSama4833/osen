"""Ways an eval item can leak into training data."""
from __future__ import annotations
import random
import re


def exact(t, rng, vocab): return t


def case_whitespace(t, rng, vocab):
    return re.sub(r"[^\w\s]", "", t.lower()).replace(" ", "  ")


def word_edit(frac):
    def f(t, rng, vocab):
        w = t.split()
        for i in rng.sample(range(len(w)), max(1, int(frac * len(w)))):
            w[i] = rng.choice(vocab)
        return " ".join(w)
    return f


def truncated(t, rng, vocab):
    w = t.split()
    return " ".join(w[: len(w) // 2])


def reordered(t, rng, vocab):
    """Shuffle sentences; if there are fewer than 3, shuffle 12-word chunks instead. Never a no-op."""
    pieces = re.split(r"(?<=[.!?])\s+", t)
    if len(pieces) < 3:
        w = t.split()
        pieces = [" ".join(w[i:i + 12]) for i in range(0, len(w), 12)]
    orig = list(pieces)
    for _ in range(10):
        rng.shuffle(pieces)
        if pieces != orig:
            break
    else:
        pieces = orig[1:] + orig[:1]
    return " ".join(pieces)


def embedded(t, rng, vocab):
    pre = " ".join(rng.choices(vocab, k=60)); post = " ".join(rng.choices(vocab, k=60))
    return f"{pre} {t} {post}"


TYPES = {"exact copy": exact, "case+punctuation+spacing": case_whitespace, "10% words replaced": word_edit(0.10),
         "30% words replaced": word_edit(0.30), "first half only": truncated, "sentences reordered": reordered,
         "embedded in longer text": embedded}
