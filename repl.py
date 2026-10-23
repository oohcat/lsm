"""
repl.py — a structure-only REPL over Fsk(At). Run with `python -i repl.py`.

No new mathematics: every answer comes from `skew_seq` (UVZ Thms 5, 6,
Lemma 21) or `interp` (∆⊥, Lack–Street Cor. 11.4). No notation layer
either: terms are written with the existing constructors, since the
README leaves operators undecided. Nothing imports this file.

  show(f)        f : A → C, or the typing error
  hom(A, C)      every map A → C, without duplicates (list: homset)
  same(f, g)     equality, decided twice by routes that share no code
                 (boolean: equal)
  inv(f)         how far f is from invertible: retractions, sections

Terms are written with the notation layer of `skew_terms`: `>>` for
composition (diagrammatic), `@` for ⊗ on objects and maps, `A @ f` and
`f @ A` for whiskering. Every result echoes as valid Python with every
bracket shown, followed by its type, so the echo says which tree was
built: `X @ Y @ Z` is ((X @ Y) @ Z).
"""

from __future__ import annotations

import sys

from interp import equal_in_delta
from skew_seq import equal, hom_exists, homset
from skew_terms import (I, Alpha, Atom, Id, Lam, Rho, Tens, Tensor,  # noqa: F401
                        Term, Then, X, Y, Z, W)


def show(f) -> None:
    print(f"{f!r} : {f.dom!r} → {f.cod!r}")


def _echo(v) -> None:
    import builtins
    if v is None:
        return
    builtins._ = v
    if isinstance(v, Term):
        show(v)
    else:
        print(repr(v))


sys.displayhook = _echo


def hom(A, C) -> None:
    """Prints Hom(A, C). For the list itself, call `homset`."""
    maps = homset(A, C)
    print(f"Hom({A}, {C}): {len(maps)}")
    for m in maps:
        print("  ", repr(m))


def same(f, g) -> None:
    """Prints the verdict. For the boolean, call `equal`."""
    if f.dom != g.dom or f.cod != g.cod:
        print("not parallel")
        return
    a, b = equal(f, g), equal_in_delta(f, g)
    if a != b:
        # Both procedures are complete; disagreement is a bug in one of them.
        raise AssertionError(f"skew_seq says {a}, ∆⊥ says {b}")
    print("equal" if a else "distinct")


def _is_id(t, A) -> bool:
    return equal(t, Id(A))


def inv(f) -> None:
    """Retractions r (f;r = id) and sections s (s;f = id) of f.

    Decided, not sampled: Hom(cod, dom) is finite and `homset` lists all
    of it (UVZ Thm 6), and `equal` is complete (Thm 5). f is invertible
    iff some map is both.
    """
    back = homset(f.cod, f.dom)
    r = [g for g in back if _is_id(Then(f, g), f.dom)]
    s = [g for g in back if _is_id(Then(g, f), f.cod)]
    show(f)
    print(f"  maps back: {len(back)}   retractions: {len(r)}   sections: {len(s)}")
    both = [g for g in r if g in s or any(equal(g, h) for h in s)]
    if both:
        print("  invertible, inverse", both[0])
    elif r:
        print("  split mono, not invertible; retraction", r[0])
    elif s:
        print("  split epi, not invertible; section", s[0])
    elif back:
        print("  maps back exist, none cancels f")
    else:
        print("  no map back at all")
