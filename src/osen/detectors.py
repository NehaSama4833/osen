"""Detectors: each scores an eval item against an index of the training corpus. Higher = more likely leaked."""
from __future__ import annotations
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def words(t): return re.findall(r"\w+", t.lower())


class NgramOverlap:
    """Fraction of an item's word n-grams that occur anywhere in the training corpus."""
    def __init__(self, n):
        self.n = n; self.name = f"{n}-gram overlap"

    def index(self, train):
        self.seen = set()
        for d in train:
            w = words(d)
            self.seen.update(hash(tuple(w[i:i + self.n])) for i in range(len(w) - self.n + 1))
        return self

    def scores(self, items):
        out = []
        for t in items:
            w = words(t)
            g = [hash(tuple(w[i:i + self.n])) for i in range(len(w) - self.n + 1)]
            out.append(sum(x in self.seen for x in g) / len(g) if g else 0.0)
        return np.array(out)


class ExactNormalized:
    name = "exact (normalised)"

    def index(self, train):
        self.s = {" ".join(words(d)) for d in train}
        return self

    def scores(self, items):
        return np.array([1.0 if " ".join(words(t)) in self.s else 0.0 for t in items])


class CharCosine:
    """Max cosine similarity of char 5-gram TF-IDF vectors with any training document."""
    name = "char-5gram cosine (nearest doc)"

    def index(self, train):
        self.v = TfidfVectorizer(analyzer="char", ngram_range=(5, 5), min_df=1, sublinear_tf=True, dtype=np.float32)
        self.M = self.v.fit_transform(train)
        return self

    def scores(self, items):
        X = self.v.transform(items)
        return np.asarray((X @ self.M.T).max(axis=1).todense()).ravel()


def threshold_at_fpr(clean_scores, fpr):
    return float(np.quantile(clean_scores, 1 - fpr, method="higher"))
