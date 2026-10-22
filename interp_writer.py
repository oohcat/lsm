"""
interp_writer.py — Fsk(At) interpreted in the skew structure induced by the
writer bimonad of `writer.py`.

Sibling of `interp.py`, and different from it in the way that matters: ∆⊥
is faithful, this model is not claimed to be. It is not a decision
procedure. It is a model in which the structure maps can be looked at
elementwise, chosen because it keeps λ, ρ and α non-invertible for the
right choice of W.

THE MODEL
---------
Szlachányi, the left-handed form stated after Definition 7.3 (read from the
page image, on the register): for an opmonoidal monad (O, ω, ι) on a
monoidal category (E, ⊗, R),
    M ⊙ N   := OM ⊗ N
    γ̇_{L,M,N} := a⁻¹_{OL,OM,N} ∘ ((ω_L ⊗ OM) ⊗ N) ∘ (O^{OL,M} ⊗ N)
    η̇_M     := r⁻¹_{OM} ∘ ι_M
    ε̇_M     := l_M ∘ (O⁰ ⊗ M)
is left-monoidal. That form is stated there, not proved; it follows from
Proposition 7.2 by the reversal duality after Def. 2.1. Its hypotheses for
the writer monad are what `test_writer.py` checks.

In lsm's names γ̇ is α : (A⊗B)⊗C → A⊗(B⊗C), η̇ is ρ : A → A⊗I and ε̇ is
λ : I⊗A → A. Every atom goes to a two-element set and I to {()}, so an
element of the interpretation of A⊗B is ((w, a), b).

Each map below is composed from `writer.py`'s ω, ι, O^{M,N}, O⁰ and the
cartesian a, l, r, following the formula, not written out by hand. The
elementwise form of α it produces is
    ((w, ((v, a), b)), c) ↦ ((w·v, a), ((w, b), c)).
"""

from __future__ import annotations

from typing import Any, Callable, List

from skew_terms import Alpha, Atom, Fma, Id, Lam, Rho, Tens, Tensor, Term, Then, Unit
from writer import R, Monoid, Writer, tens


# ---- inverses of the cartesian structure maps in writer.py -----------------

def a_inv(x):
    (l, m), n = x
    return (l, (m, n))


def r_inv(m):
    return (m, ())


def l(x):
    _u, m = x
    return m


def _then(*fs: Callable) -> Callable:
    def h(x):
        for f in fs:
            x = f(x)
        return x
    return h


def _tensor(f: Callable, g: Callable) -> Callable:
    return lambda x: (f(x[0]), g(x[1]))


def _id(x):
    return x


# ---- the induced left-monoidal structure ------------------------------------

class WriterSkew:
    def __init__(self, W: Monoid):
        self.W = W
        self.O = Writer(W)

    def gamma(self) -> Callable:
        O = self.O
        return _then(_tensor(O.op2, _id),
                     _tensor(_tensor(O.omega, _id), _id),
                     a_inv)

    def eta(self) -> Callable:
        return _then(self.O.iota, r_inv)

    def eps(self) -> Callable:
        return _then(_tensor(self.O.op0, _id), l)

    def elems(self, A: Fma) -> List[Any]:
        """A finite sample of the set A goes to."""
        if isinstance(A, Atom):
            return [A.name.lower() + "0", A.name.lower() + "1"]
        if isinstance(A, Unit):
            return list(R)
        return tens(self.O.obj(self.elems(A.left)), self.elems(A.right))

    def interp(self, t: Term) -> Callable:
        """The function a term goes to. Recurses over the raw tree."""
        if isinstance(t, Id):
            return _id
        if isinstance(t, Then):
            return _then(self.interp(t.first), self.interp(t.second))
        if isinstance(t, Tensor):
            # f ⊙ g = Of ⊗ g: the log rides with the left factor.
            return _tensor(self.O.fmap(self.interp(t.left)),
                           self.interp(t.right))
        if isinstance(t, Lam):
            return self.eps()
        if isinstance(t, Rho):
            return self.eta()
        if isinstance(t, Alpha):
            return self.gamma()
        raise TypeError(f"interp: not a term: {t!r}")

    def equal_on_sample(self, f: Term, g: Term) -> bool:
        if f.dom != g.dom or f.cod != g.cod:
            return False
        F, G = self.interp(f), self.interp(g)
        return all(F(x) == G(x) for x in self.elems(f.dom))
