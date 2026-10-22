"""
writer.py — the writer monad on (Set, ×, 1), as data for Szlachányi §7.

Not part of lsm's term layer, and nothing there imports it. It supplies the
hypotheses of Szlachányi Proposition 7.2: a monoidal category (E, ⊗, R) and
a bimonad (O, ω, ι) on it, i.e. a monad with an opmonoidal structure
O^{M,N} : O(M⊗N) → OM⊗ON, O⁰ : OR → R satisfying (65)–(71). Whether this
data meets those hypotheses is what `test_writer.py` checks; no source read
states it.

THE DATA
--------
E is (Set, ×, 1). Elements of M⊗N are pairs (m, n); R is the one-point set
{()}. The structure maps are written in Szlachányi's right-monoidal
orientation, as (65)–(71) use them:
    a_{L,M,N} : L⊗(M⊗N) → (L⊗M)⊗N
    l⁻¹_M     : M → R⊗M
    r_M       : M⊗R → M
For a monoid W, OM = W × M, with elements (w, m):
    ω_M : OOM → OM          (w, (v, m)) ↦ (w·v, m)
    ι_M : M → OM            m ↦ (e, m)
    O^{M,N}                 (w, (m, n)) ↦ ((w, m), (w, n))
    O⁰                      (w, ()) ↦ ()
O^{M,N} is the diagonal one: the log is copied to both factors. Whether
it is the only opmonoidal structure on this monad is not decided here.
Writing the log on the first factor only, ((w, m), (e, n)), is not one:
it fails (66).

Natural transformations are represented by their action on elements, and
equations are checked elementwise on finite samples, so a pass is evidence
on those samples, not a proof.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, List


@dataclass(frozen=True)
class Monoid:
    name: str
    elems: tuple  # a finite sample; for an infinite W, not closed under mul
    mul: Callable[[Any, Any], Any]
    e: Any


def free_monoid(letters: str, max_len: int) -> Monoid:
    words = tuple(w for n in range(max_len + 1)
                  for w in itertools.product(letters, repeat=n))
    return Monoid(f"free({letters})", words, lambda u, v: u + v, ())


def cyclic_group(n: int) -> Monoid:
    return Monoid(f"Z/{n}", tuple(range(n)), lambda u, v: (u + v) % n, 0)


TRIVIAL = Monoid("1", (0,), lambda u, v: 0, 0)


# ---- (Set, ×, 1): objects as finite sample lists ---------------------------

R: List[Any] = [()]


def tens(M: List[Any], N: List[Any]) -> List[Any]:
    return [(m, n) for m in M for n in N]


def a(x):
    l, (m, n) = x
    return ((l, m), n)


def l_inv(m):
    return ((), m)


def r(x):
    m, _u = x
    return m


# ---- the writer bimonad ----------------------------------------------------

class Writer:
    def __init__(self, W: Monoid):
        self.W = W

    def obj(self, M: List[Any]) -> List[Any]:
        return [(w, m) for w in self.W.elems for m in M]

    def fmap(self, f: Callable) -> Callable:
        return lambda x: (x[0], f(x[1]))

    def omega(self, x):
        w, (v, m) = x
        return (self.W.mul(w, v), m)

    def iota(self, m):
        return (self.W.e, m)

    def op2(self, x):
        w, (m, n) = x
        return ((w, m), (w, n))

    def op0(self, x):
        return ()
