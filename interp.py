"""
interp.py — Fsk(At) interpreted in ∆⊥, the one model that is faithful.

Sibling of `cqm/interp.py`, and the place where the two projects differ
most. `cqm`'s FdHilb functor is a strict monoidal dagger functor that is
provably *not* faithful (its README, "Functor, not yet faithful"). This
one is faithful, so it is a second, complete decision procedure for
equality, sharing no code with `skew_seq`. Agreement between the two is
evidence; disagreement is a bug in one of them.

THE MODEL
---------
∆⊥ is the category of finite non-empty ordinals and order- and
bottom-preserving functions, with ⊗ the ordinal sum and unit 1. It is
skew monoidal with α the identity, λₙ : 1+n → n the surjection σ₀
collapsing the bottom two elements, and ρₙ : n → n+1 the injection δₙ
missing the new top element. Lack & Street, *Triangulations, orientals,
and skew monoidal categories*, Adv. Math. 258 (2014), §3. Checked in the
source.

Every atom and I go to the ordinal 1, so A goes to the number of leaves
of the formula A (atoms and units both). A morphism m → n is represented
as the tuple of its values (f(0), …, f(m-1)).

WHAT IS CITED, AND WHAT IS TESTED
---------------------------------
  - Faithfulness on the one-generator case: Lack & Street, Corollary 11.4
    — Fsk admits a unique faithful structure-preserving functor into ∆⊥
    sending the generator to the unit. Their introduction calls this a
    complete solution to the word problem. Checked in the source.
  - Several atoms: Lack & Street, Theorem 12.2 describes the free skew
    monoidal category on a category X; for X discrete (a set of atoms) a
    morphism is just an Fsk morphism that respects the labels. Since
    parallel maps already have the same labels, faithfulness transfers.
    That last step is argued here, not stated in the paper; it is tested
    on a two-atom sweep in `test_interp.py`.
  - That this `interp` is a functor at all — that the axioms hold in ∆⊥
    with these choices of λ and ρ — is not taken from the paper but
    tested: every entry of `skew_terms.AXIOMS` must interpret equally.

Nothing here is used by `skew_terms` or `skew_seq`; the term layer knows
nothing about this file.
"""

from __future__ import annotations

from typing import Tuple

from skew_terms import (
    Alpha,
    Fma,
    Id,
    Lam,
    Rho,
    Tens,
    Tensor,
    Term,
    Then,
)

Map = Tuple[int, ...]


def size(A: Fma) -> int:
    """The ordinal A goes to: atoms and I are both 1, ⊗ is +."""
    if isinstance(A, Tens):
        return size(A.left) + size(A.right)
    return 1


def interp(t: Term) -> Map:
    """The ∆⊥ map of a term, as the tuple of its values.

    Recurses over the raw tree, never over a normal form, so that agreement
    with `skew_seq.equal` is something the tests check rather than
    something true by construction.
    """
    if isinstance(t, Id):
        return tuple(range(size(t.obj)))
    if isinstance(t, Then):
        f, g = interp(t.first), interp(t.second)
        return tuple(g[i] for i in f)
    if isinstance(t, Tensor):
        f, g = interp(t.left), interp(t.right)
        shift = size(t.left.cod)
        return f + tuple(shift + j for j in g)
    if isinstance(t, Lam):
        # σ₀ : 1+n → n, identifying the unit's element with the bottom.
        return (0,) + tuple(range(size(t.obj)))
    if isinstance(t, Rho):
        # δₙ : n → n+1, the inclusion missing the new top element.
        return tuple(range(size(t.obj)))
    if isinstance(t, Alpha):
        # ∆⊥ is strictly associative.
        return tuple(range(size(t.dom)))
    raise TypeError(f"interp: not a term: {t!r}")


def is_delta_bot_map(f: Map, m: int, n: int) -> bool:
    """f : m → n is order-preserving and bottom-preserving."""
    return (
        len(f) == m
        and f[0] == 0
        and all(0 <= v < n for v in f)
        and all(f[i] <= f[i + 1] for i in range(m - 1))
    )


def equal_in_delta(f: Term, g: Term) -> bool:
    """Equality in ∆⊥. By faithfulness, equality in Fsk(At) itself."""
    if f.dom != g.dom or f.cod != g.cod:
        return False
    return interp(f) == interp(g)
