"""
bialgebra.py — the monoid bialgebra GF(2)[W], as data for Szlachányi §3.

Not part of lsm's term layer, and nothing there imports it. It is the
linear counterpart of `writer.py`: where the writer monad copies the log
by the diagonal of Set, here the comultiplication copies only the log,
w ↦ w ⊗ w, and nothing copies a module.

Szlachányi §3 (on the register) builds a right-monoidal structure on right
R-modules from a right R-bialgebroid H. Here R = k = GF(2), and H = k[W]
is a k-bialgebra. That a k-bialgebra is a k-bialgebroid, with source and
target both the unit map k → H, is argued here, not taken from a source
read. Whether this data meets the bialgebra axioms is what
`test_bialgebra.py` checks.

THE DATA
--------
A vector of H is a finite set of basis words, read as their sum over
GF(2); addition is symmetric difference. A vector of H ⊗ H is a finite set
of pairs, and of H ⊗ H ⊗ H a set of triples. On basis elements:
    m(w ⊗ v) = w·v          u(1) = e
    Δ(w) = w ⊗ w            ε(w) = 1
Every map is extended linearly, so over GF(2) terms that occur twice
cancel. The basis elements are group-like, which is why Δ copies the log
and ε forgets it.

Equations are checked on basis elements and on small sums of them. For a
finite W the basis is the whole of W, so a check over it is exact by
linearity; for W free it is a sample of words, so a pass is evidence on
that sample.
"""

from __future__ import annotations

from typing import Any, Callable, FrozenSet, Iterable

from writer import Monoid

Vec = FrozenSet[Any]


def idempotent_monoid() -> Monoid:
    """{e, a} with a·a = a: a monoid that is not a group."""
    return Monoid("{e,a}, a·a=a", ("e", "a"),
                  lambda u, v: "a" if "a" in (u, v) else "e", "e")


def vec(*basis: Any) -> Vec:
    """The GF(2) sum of the given basis elements; repeats cancel."""
    out: set = set()
    for b in basis:
        out ^= {b}
    return frozenset(out)


def lin(f: Callable[[Any], Vec]) -> Callable[[Vec], Vec]:
    """Extend a map defined on basis elements linearly over GF(2)."""
    def F(x: Vec) -> Vec:
        out: set = set()
        for b in x:
            out ^= set(f(b))
        return frozenset(out)
    return F


def scalar(x: Vec) -> int:
    """A vector of k ⊗ … ⊗ k = k, i.e. its GF(2) coefficient."""
    return len(x) % 2


class MonoidBialgebra:
    def __init__(self, W: Monoid):
        self.W = W

    def basis(self) -> Iterable[Any]:
        return self.W.elems

    # algebra
    def m(self, x: Vec) -> Vec:
        return lin(lambda p: vec(self.W.mul(p[0], p[1])))(x)

    def u(self) -> Vec:
        return vec(self.W.e)

    # coalgebra
    def delta(self, x: Vec) -> Vec:
        return lin(lambda w: vec((w, w)))(x)

    def eps(self, x: Vec) -> int:
        return scalar(x)


def tensor(x: Vec, y: Vec) -> Vec:
    """x ⊗ y for vectors of H, as a vector of H ⊗ H."""
    return vec(*[(a, b) for a in x for b in y])


def f_tensor(f: Callable[[Vec], Vec], g: Callable[[Vec], Vec]) -> Callable[[Vec], Vec]:
    """f ⊗ g on H ⊗ H, applied basis pair by basis pair."""
    return lin(lambda p: tensor(f(vec(p[0])), g(vec(p[1]))))
