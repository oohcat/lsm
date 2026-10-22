"""
interp_bialgebra.py — Fsk(At) interpreted in the skew structure Szlachányi
§3 builds from the monoid bialgebra GF(2)[W] of `bialgebra.py`.

The linear sibling of `interp_writer.py`. Like it, this model is not
claimed faithful and decides nothing; it is a model in which the structure
maps can be inspected. Unlike it, nothing here copies a module: the only
copying is Δ on H, and Δ copies basis elements, not vectors.

THE MODEL
---------
Szlachányi §3 (on the register, read from the page image), with R = k:
    M ∗ N := M ⊗ N ⊗ H,        elements [m, n, h]
    η_M : M → R ∗ M            m ↦ [1, m, 1_H]
    ε_M : M ∗ R → M            [m, r, h] ↦ m · ε(s(r) h)
    γ_{L,M,N} : L ∗ (M ∗ N) → (L ∗ M) ∗ N
                               [l, [m, n, g], h] ↦ [[l, m, h⁽¹⁾], n, g h⁽²⁾]
That is right-monoidal (stated there, not proved). lsm is left skew, so
the product is reversed, by the duality after Def. 2.1: lsm's A ⊗ B goes to
B ∗ A. Then lsm's α_{A,B,C} : (A⊗B)⊗C → A⊗(B⊗C) is γ_{C,B,A}, lsm's ρ is η
and lsm's λ is ε, each in the direction lsm requires.

With R = k = GF(2), s(r) = r·1_H, and every basis element of H = k[W]
group-like (h⁽¹⁾ ⊗ h⁽²⁾ = h ⊗ h, ε(h) = 1). So on basis elements, with an
element of lsm's A ⊗ B written (b, a, h) for [b, a, h] ∈ B ∗ A:
    λ_A : (a, (), h) ↦ a
    ρ_A : a ↦ ((), a, e)
    α   : (c, (b, a, g), h) ↦ ((c, b, h), a, g·h)
Every map is linear over GF(2): a vector is a finite set of basis elements,
and maps are applied basis element by basis element with cancellation.
That k-bialgebras are k-bialgebroids is argued in `bialgebra.py`.

The inner log of α's image is a suffix of the outer one, g·h ends in h,
where the writer model's is a prefix, w·v starts with w. As transcribed,
this file and `interp_writer.py` multiply logs in opposite orders. Each is
a model (the other order is the opposite monoid's); this file follows §3's
γ as printed.

Every atom goes to a two-dimensional space, I to k. Equations are checked
on every basis element of the domain, which by linearity is exact for a
finite W; for W free the basis is a sample of words.
"""

from __future__ import annotations

from typing import Any, Callable, List

from bialgebra import MonoidBialgebra, Vec, lin, vec
from skew_terms import Alpha, Atom, Fma, Id, Lam, Rho, Tens, Tensor, Term, Then, Unit
from writer import Monoid

ONE = ()  # the basis element 1 of k


class BialgebraSkew:
    def __init__(self, W: Monoid):
        self.W = W
        self.H = MonoidBialgebra(W)

    # ---- objects: a basis for the space each formula goes to ----------
    def basis(self, A: Fma) -> List[Any]:
        if isinstance(A, Atom):
            return [A.name.lower() + "0", A.name.lower() + "1"]
        if isinstance(A, Unit):
            return [ONE]
        # lsm's A ⊗ B goes to B ∗ A = B ⊗ A ⊗ H
        return [(b, a, h) for b in self.basis(A.right)
                for a in self.basis(A.left) for h in self.W.elems]

    # ---- §3's structure maps, on basis elements -----------------------
    def eps_basis(self, x) -> Vec:
        # ε_M([m, r, h]) = m · ε(s(r) h); r is ONE and ε(h) = 1.
        m, _r, h = x
        return vec(m) if self.H.eps(vec(h)) else vec()

    def eta_basis(self, m) -> Vec:
        # η_M(m) = [1, m, 1_H]
        return vec((ONE, m, self.W.e))

    def gamma_basis(self, x) -> Vec:
        # γ([l, [m, n, g], h]) = [[l, m, h⁽¹⁾], n, g h⁽²⁾], with Δh = Σ h⁽¹⁾ ⊗ h⁽²⁾.
        l, (m, n, g), h = x
        out: set = set()
        for h1, h2 in self.H.delta(vec(h)):
            for gh2 in self.H.m(vec((g, h2))):
                out ^= {((l, m, h1), n, gh2)}
        return frozenset(out)

    # ---- terms ---------------------------------------------------------
    def interp(self, t: Term) -> Callable[[Vec], Vec]:
        """The linear map a term goes to. Recurses over the raw tree."""
        if isinstance(t, Id):
            return lambda x: x
        if isinstance(t, Then):
            f, g = self.interp(t.first), self.interp(t.second)
            return lambda x: g(f(x))
        if isinstance(t, Tensor):
            # lsm's f ⊗ g goes to g ∗ f = g ⊗ f ⊗ id_H.
            f, g = self.interp(t.left), self.interp(t.right)
            return lin(lambda x: vec(*[(b2, a2, x[2])
                                       for b2 in g(vec(x[0]))
                                       for a2 in f(vec(x[1]))]))
        if isinstance(t, Lam):
            return lin(self.eps_basis)
        if isinstance(t, Rho):
            return lin(self.eta_basis)
        if isinstance(t, Alpha):
            return lin(self.gamma_basis)
        raise TypeError(f"interp: not a term: {t!r}")

    def equal_on_basis(self, f: Term, g: Term) -> bool:
        if f.dom != g.dom or f.cod != g.cod:
            return False
        F, G = self.interp(f), self.interp(g)
        return all(F(vec(b)) == G(vec(b)) for b in self.basis(f.dom))
