"""Real technical prose: docstring paragraphs from the Python standard library (parsed with ast, never imported)."""
from __future__ import annotations
import ast, random, re, sysconfig
from pathlib import Path


def stdlib_paragraphs(min_words=30, max_words=160):
    root = Path(sysconfig.get_paths()["stdlib"])
    by_module: dict[str, list[str]] = {}
    for f in sorted(root.rglob("*.py")):
        rel = str(f.relative_to(root))
        if "test" in rel.split("/")[0] or "idlelib" in rel or "site-packages" in rel or "lib2to3" in rel:
            continue
        try:
            tree = ast.parse(f.read_text(errors="ignore"))
        except (SyntaxError, ValueError):
            continue
        paras = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                d = ast.get_docstring(node)
                if not d:
                    continue
                for p in re.split(r"\n\s*\n", d):
                    p = " ".join(p.split())
                    n = len(p.split())
                    if min_words <= n <= max_words:
                        paras.append(p)
        if paras:
            by_module[rel] = paras
    return by_module


def split_by_module(by_module, seed=0, eval_frac=0.45):
    rng = random.Random(seed)
    mods = sorted(by_module)
    rng.shuffle(mods)
    cut = int(len(mods) * eval_frac)
    seen = set()
    ev, tr = [], []
    for i, m in enumerate(mods):
        for p in by_module[m]:
            if p in seen:
                continue
            seen.add(p)
            (ev if i < cut else tr).append(p)
    return tr, ev
