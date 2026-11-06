import random
from osen.contaminate import TYPES, case_whitespace, embedded, exact, reordered, truncated, word_edit
from osen.corpus import split_by_module, stdlib_paragraphs
from osen.detectors import CharCosine, ExactNormalized, NgramOverlap, threshold_at_fpr, words

TEXT = ("The quick brown fox jumps over the lazy dog while the cat watches from the old stone wall, "
        "and nobody notices the small bird that sings above them all through the long afternoon.")
TRAIN = [TEXT, "Completely unrelated text about network sockets and file descriptors in a long running server process "
               "that reads from many clients and writes replies back without ever blocking the main loop at all."]


def test_corpus_is_real_nonempty_and_split_has_no_shared_paragraphs():
    bm = stdlib_paragraphs()
    tr, ev = split_by_module(bm, 0)
    assert len(tr) > 500 and len(ev) > 200
    assert not (set(tr) & set(ev))
    assert all(30 <= len(p.split()) <= 160 for p in tr[:50])


def test_exact_copy_scores_one_on_every_overlap_detector():
    for d in (ExactNormalized(), NgramOverlap(13), NgramOverlap(5), CharCosine()):
        s = d.index(TRAIN).scores([TEXT])[0]
        assert s > 0.99, d.name


def test_unrelated_text_scores_low():
    other = "Pandas of the mountain forests eat bamboo shoots for most of the day and rest in the shade of tall pines when it rains."
    for d in (NgramOverlap(8), CharCosine()):
        assert d.index(TRAIN).scores([other])[0] < 0.3


def test_hand_computed_ngram_fraction():
    # item has 5 words -> 3 trigrams; train contains the first two
    d = NgramOverlap(3).index(["a b c d"])
    assert d.scores(["a b c d x"])[0] == 2 / 3


def test_contamination_transforms_behave():
    rng = random.Random(0)
    assert exact(TEXT, rng, []) == TEXT
    assert "." not in case_whitespace(TEXT, rng, [])
    assert len(truncated(TEXT, rng, []).split()) == len(TEXT.split()) // 2
    e = word_edit(0.3)(TEXT, rng, ["ZZZ"])
    assert 0 < sum(w == "ZZZ" for w in e.split()) <= int(0.3 * len(TEXT.split())) + 1
    r = reordered(TEXT, rng, [])
    assert sorted(words(r)) == sorted(words(TEXT)) and r != TEXT
    one = ' '.join(f'w{i}' for i in range(40)) + '.'   # realistic: one long sentence, no commas
    assert reordered(one, rng, []) != one
    assert TEXT in embedded(TEXT, rng, ["w"])


def test_ngram_beats_exact_on_embedded_but_fails_on_heavy_edits():
    rng = random.Random(1)
    vocab = ["alpha", "beta", "gamma", "delta"] * 5
    emb = embedded(TEXT, rng, vocab)
    assert ExactNormalized().index([emb]).scores([TEXT])[0] == 0.0
    assert NgramOverlap(8).index([emb]).scores([TEXT])[0] > 0.99
    heavy = word_edit(0.5)(TEXT, rng, vocab)
    assert NgramOverlap(8).index([heavy]).scores([TEXT])[0] < 0.2


def test_threshold_gives_at_most_target_flags_on_calibration():
    import numpy as np
    s = np.random.default_rng(0).random(200)
    assert (s > threshold_at_fpr(s, 0.05)).mean() <= 0.05


def test_all_types_registered():
    assert len(TYPES) == 7
