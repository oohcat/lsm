"""
skew_terms.py — the term layer of a left skew monoidal category.

Sibling of `cqm/strict_terms.py`, and deliberately its opposite where the
mathematics forces a difference. `cqm` is strict: objects are flat tuples,
`⊗` is concatenation, and the structure maps are identities. Here objects
are *bracketed trees* and λ, ρ, α are directed generators with no inverses,
because the whole content of skew monoidality is that those maps do not
invert. Flattening brackets is not an available simplification: it is the
thing being denied.

WHAT IS CHECKED, AND WHAT ISN'T
-------------------------------
Citation discipline follows the sibling project: a reference written down
here has been read in the source, and anything not on that list is
unchecked until it is.

Checked against Uustalu, Veltri & Zeilberger, *The Sequent Calculus of
Skew Monoidal Categories* (arXiv:2003.05213, the extended journal version):

  - §2, Figure 1: the rules of the skew monoidal *categorical calculus* —
    id, comp, ⊗, λ, ρ, α — which are exactly the generators below.
  - §2, the directions λ_A : I⊗A → A, ρ_A : A → A⊗I,
    α_{A,B,C} : (A⊗B)⊗C → A⊗(B⊗C).
  - §2, Proposition 1: a function At → C into any skew monoidal C extends
    *uniquely* to a strict monoidal functor Fsk(At) → C. This is the
    universal property that licenses a separate `interp` taking nothing
    but a dictionary on atoms. (In `cqm` the analogous claim is flagged as
    orientation rather than theorem; here it is a cited result.)
  - §2, the frontier ∂ and the fact that ∂A = ∂B is *necessary but not
    sufficient* for a map A ⇒ B to exist. See `frontier` and
    `NO_MAP_EXISTS`.
  - §2, the three non-derivable sequents and the three pairs of distinct
    parallel maps. See `NO_MAP_EXISTS` and `MUST_NOT_IDENTIFY`.

NOT checked against its own source: Szlachányi, *Skew-monoidal categories
and bialgebroids*, Adv. Math. 231 (2012). The five axioms below are taken
from UVZ's presentation of it, not from Szlachányi directly.

NOT fully checked even in UVZ: the commuting triangles (a)–(e) are
typeset as diagrams, and the orientation of (b), (c) and (d) below was
reconstructed from the standard statement of the skew monoidal axioms
rather than read off the rendered diagrams. They are well-typed (the
tests confirm that much) and they are the usual ones, but the
*orientation* of each triangle is the part to re-check against a typeset
copy before any of it is cited in the paper. This is the same class of
error as the Mac Lane VII.2-vs-XI.3 mixup recorded in `cqm`, caught
early rather than late.

WHAT THIS FILE DOES NOT DO
--------------------------
It does not decide equality of terms. Structural equality here is
*syntactic*: `Then(f, Id(A))` is a different tree from `f`. Deciding the
quotient by the equations is the job of the focused sequent calculus
(UVZ §3–§5), which goes in `skew_seq.py`. That calculus produces exactly
one canonical representative per equivalence class, so unlike
`cqm.normalise()` — sound but incomplete, missing interchange — the
skew normaliser is complete by construction. The obligations it must
discharge are stated here as data (`AXIOMS`, `MUST_NOT_IDENTIFY`,
`NO_MAP_EXISTS`) so that they are written down before the normaliser
exists to be graded against them.

CHOSEN, NOT FORCED: no generating morphisms
-------------------------------------------
There is no `Box` generator. Fsk(At) is free on a set of *objects*; its
only morphisms are built from id, comp, ⊗ and the three structure maps.
That restriction is what UVZ's coherence theorem covers, and it is what
buys decidable equality and duplicate-free homset enumeration. Adding
generating morphisms forfeits both — nothing in the cited results
extends to that case. Whether the banking model needs them is open and
belongs in the paper, not here.

CHOSEN, NOT FORCED: no dagger
-----------------------------
`cqm` has `dagger`. This file cannot. A dagger would supply
α† : A⊗(B⊗C) → (A⊗B)⊗C, and UVZ §2 states that no such map exists in
Fsk(At). A dagger on a skew monoidal category would collapse it to a
monoidal one, which is precisely the structure the banking strand argues
against. Its absence is load-bearing, not an omission.

CONVENTION
----------
Composition is diagrammatic, as in `cqm`: `Then(f, g)` is "f, then g",
which UVZ write `g ∘ f`. This is not merely inherited from the sibling
project — it is the better fit here, because UVZ's sequent-calculus cut
rules are already diagrammatic (`scut` takes S|Γ → A and A|Δ → C to
S|Γ,Δ → C), so the port in `skew_seq.py` will be a transcription rather
than a translation. Only the categorical calculus of Figure 1 is written
in ∘-order in the paper, and only `Then` is affected.

Run with `python skew_terms.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Objects (formulae)
# ---------------------------------------------------------------------------


class Fma:
    """A formula: an atom, the unit I, or a tensor of two formulae.

    Trees, not tuples. `(A@B)@C` and `A@(B@C)` are distinct objects, and
    the only thing connecting them is `Alpha`, in one direction.
    """

    def __matmul__(self, other: "Fma") -> "Tens":
        return Tens(self, other)


@dataclass(frozen=True, repr=False)
class Atom(Fma):
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, repr=False)
class Unit(Fma):
    def __repr__(self) -> str:
        return "I"


I = Unit()


@dataclass(frozen=True, repr=False)
class Tens(Fma):
    left: Fma
    right: Fma

    def __repr__(self) -> str:
        return f"({self.left}⊗{self.right})"


def frontier(A: Fma) -> Tuple[str, ...]:
    """∂A: the underlying list of atoms, with I contributing nothing.

    UVZ §2. This is exactly the data a naive "objects are strings"
    model would retain, and `NO_MAP_EXISTS` is the record of why that
    retention is not enough: ∂A = ∂B is necessary for a map A ⇒ B and
    demonstrably not sufficient.
    """
    if isinstance(A, Atom):
        return (A.name,)
    if isinstance(A, Unit):
        return ()
    if isinstance(A, Tens):
        return frontier(A.left) + frontier(A.right)
    raise TypeError(f"not a formula: {A!r}")


# ---------------------------------------------------------------------------
# Terms (derivations of the categorical calculus, UVZ Figure 1)
# ---------------------------------------------------------------------------


class Term:
    """A morphism of Fsk(At), as a syntax tree. Contains no numbers.

    `dom` and `cod` are computed from the tree, never stored, so a term
    cannot carry a type that disagrees with its own structure.
    """

    @property
    def dom(self) -> Fma:
        raise NotImplementedError

    @property
    def cod(self) -> Fma:
        raise NotImplementedError


@dataclass(frozen=True, repr=False)
class Id(Term):
    obj: Fma

    @property
    def dom(self) -> Fma:
        return self.obj

    @property
    def cod(self) -> Fma:
        return self.obj

    def __repr__(self) -> str:
        return f"id[{self.obj}]"


@dataclass(frozen=True, repr=False)
class Then(Term):
    """Diagrammatic composition: first, then second. UVZ's `comp`."""

    first: Term
    second: Term

    def __post_init__(self) -> None:
        if self.first.cod != self.second.dom:
            raise TypeError(
                f"cannot compose: {self.first} ends at {self.first.cod}, "
                f"{self.second} starts at {self.second.dom}"
            )

    @property
    def dom(self) -> Fma:
        return self.first.dom

    @property
    def cod(self) -> Fma:
        return self.second.cod

    def __repr__(self) -> str:
        return f"({self.first} ; {self.second})"


@dataclass(frozen=True, repr=False)
class Tensor(Term):
    """f ⊗ g. Note ⊗ is still an honest bifunctor on C × C.

    Skew monoidality weakens *bracketing*, not interchange. The law
    (f⊗g);(h⊗k) = (f;h)⊗(g;k) holds on the nose here, and is listed in
    `AXIOMS` as an obligation on the normaliser. A structure in which
    interchange fails is premonoidal, not skew monoidal, and is a
    different weakening entirely.
    """

    left: Term
    right: Term

    @property
    def dom(self) -> Fma:
        return Tens(self.left.dom, self.right.dom)

    @property
    def cod(self) -> Fma:
        return Tens(self.left.cod, self.right.cod)

    def __repr__(self) -> str:
        return f"({self.left} ⊗ {self.right})"


@dataclass(frozen=True, repr=False)
class Lam(Term):
    """λ_A : I⊗A → A. One direction only; λ⁻¹ is not constructible."""

    obj: Fma

    @property
    def dom(self) -> Fma:
        return Tens(I, self.obj)

    @property
    def cod(self) -> Fma:
        return self.obj

    def __repr__(self) -> str:
        return f"λ[{self.obj}]"


@dataclass(frozen=True, repr=False)
class Rho(Term):
    """ρ_A : A → A⊗I. One direction only; ρ⁻¹ is not constructible."""

    obj: Fma

    @property
    def dom(self) -> Fma:
        return self.obj

    @property
    def cod(self) -> Fma:
        return Tens(self.obj, I)

    def __repr__(self) -> str:
        return f"ρ[{self.obj}]"


@dataclass(frozen=True, repr=False)
class Alpha(Term):
    """α_{A,B,C} : (A⊗B)⊗C → A⊗(B⊗C). One direction only.

    This is the generator the whole banking claim rests on. There is no
    constructor for its inverse, and by UVZ §2 no composite of the
    available generators provides one.
    """

    a: Fma
    b: Fma
    c: Fma

    @property
    def dom(self) -> Fma:
        return Tens(Tens(self.a, self.b), self.c)

    @property
    def cod(self) -> Fma:
        return Tens(self.a, Tens(self.b, self.c))

    def __repr__(self) -> str:
        return f"α[{self.a},{self.b},{self.c}]"


def is_parallel(f: Term, g: Term) -> bool:
    """Same domain and same codomain. Not an assertion that f = g."""
    return f.dom == g.dom and f.cod == g.cod


# ---------------------------------------------------------------------------
# The equational theory, as data
# ---------------------------------------------------------------------------
#
# Stated here, ahead of any normaliser, so that the normaliser is graded
# against obligations written down before it existed. Each entry is
# (name, lhs, rhs) over the sample objects below; the normaliser in
# `skew_seq.py` must identify every pair in `AXIOMS` and must *not*
# identify any pair in `MUST_NOT_IDENTIFY`.

X, Y, Z, W = Atom("X"), Atom("Y"), Atom("Z"), Atom("W")


def _axioms():
    A, B, C, D = X, Y, Z, W
    out = []

    # (a)  ρ_I then λ_I = id_I.
    out.append(("a: unit triangle", Then(Rho(I), Lam(I)), Id(I)))

    # (b)  (ρ_A ⊗ id_B) then α then (id_A ⊗ λ_B) = id_{A⊗B}.
    #      The axiom that makes ρ and λ "cancel" without either inverting.
    out.append((
        "b: middle triangle",
        Then(
            Tensor(Rho(A), Id(B)),
            Then(Alpha(A, I, B), Tensor(Id(A), Lam(B))),
        ),
        Id(Tens(A, B)),
    ))

    # (c)  α_{I,A,B} then λ_{A⊗B} = λ_A ⊗ id_B.
    out.append((
        "c: λ triangle",
        Then(Alpha(I, A, B), Lam(Tens(A, B))),
        Tensor(Lam(A), Id(B)),
    ))

    # (d)  ρ_{A⊗B} then α_{A,B,I} = id_A ⊗ ρ_B.
    out.append((
        "d: ρ triangle",
        Then(Rho(Tens(A, B)), Alpha(A, B, I)),
        Tensor(Id(A), Rho(B)),
    ))

    # (e)  Pentagon, directed.
    out.append((
        "e: pentagon",
        Then(Alpha(Tens(A, B), C, D), Alpha(A, B, Tens(C, D))),
        Then(
            Tensor(Alpha(A, B, C), Id(D)),
            Then(Alpha(A, Tens(B, C), D), Tensor(Id(A), Alpha(B, C, D))),
        ),
    ))

    # Category laws.
    f = Lam(A)  # I⊗A → A, used as a stand-in arrow
    out.append(("unit left", Then(Id(f.dom), f), f))
    out.append(("unit right", Then(f, Id(f.cod)), f))

    # Functoriality of ⊗ — interchange. Holds strictly; see `Tensor`.
    g, h = Rho(A), Lam(B)
    out.append((
        "⊗ interchange",
        Tensor(Then(f, g), h),
        Then(Tensor(f, Id(h.dom)), Tensor(g, h)),
    ))
    out.append(("⊗ preserves id", Tensor(Id(A), Id(B)), Id(Tens(A, B))))

    # Naturality of λ, ρ, α, witnessed with structural arrows since there
    # are no generating morphisms to witness them with.
    k = Lam(A)  # I⊗A → A
    out.append((
        "λ natural",
        Then(Tensor(Id(I), k), Lam(A)),
        Then(Lam(Tens(I, A)), k),
    ))
    out.append((
        "ρ natural",
        Then(k, Rho(A)),
        Then(Rho(Tens(I, A)), Tensor(k, Id(I))),
    ))
    return out


AXIOMS = _axioms()


# Pairs that are parallel but genuinely distinct in Fsk(At). UVZ §2.
# These are the "must not fire" cases: a normaliser that identifies any
# of them has quotiented too hard and is modelling a monoidal category,
# not a skew monoidal one. Each of these *does* hold in any monoidal
# category, which is exactly why they are the discriminating tests.
MUST_NOT_IDENTIFY = [
    ("ρ_I ∘ λ_I ≠ id", Then(Lam(I), Rho(I)), Id(Tens(I, I))),
    (
        "(ρ⊗λ) ∘ α ≠ id",
        Then(Alpha(X, I, Y), Tensor(Rho(X), Lam(Y))),
        Id(Tens(Tens(X, I), Y)),
    ),
    (
        "α ∘ (ρ⊗λ) ≠ id",
        Then(Tensor(Rho(X), Lam(Y)), Alpha(X, I, Y)),
        Id(Tens(X, Tens(I, Y))),
    ),
]


# Object pairs with equal frontier and *no* map between them at all.
# UVZ §2. These are the obligations for the decision procedure in
# `skew_seq.py`, and jointly they are the proof that a strings-with-a-
# preorder model cannot work: the frontier is equal in every case.
NO_MAP_EXISTS = [
    ("no λ⁻¹", X, Tens(I, X)),
    ("no ρ⁻¹", Tens(X, I), X),
    ("no α⁻¹", Tens(X, Tens(Y, Z)), Tens(Tens(X, Y), Z)),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _test_objects_are_trees():
    # Fails under: objects implemented as flat tuples with concatenation
    # for ⊗, i.e. the `cqm` representation. That representation makes
    # these two objects equal and leaves nothing for α to act on.
    assert (X @ Y) @ Z != X @ (Y @ Z)
    assert frontier((X @ Y) @ Z) == frontier(X @ (Y @ Z)) == ("X", "Y", "Z")
    assert frontier(I) == ()
    assert frontier((X @ I) @ (I @ Y)) == ("X", "Y")


def _test_typing_rejects_bad_composition():
    # Fails under: Then dropping its __post_init__ check.
    try:
        Then(Lam(X), Lam(Y))
    except TypeError:
        pass
    else:
        raise AssertionError("Then accepted a type-incorrect composition")


def _test_structure_map_directions():
    # Fails under: any transcription error in dom/cod. These directions
    # are the definition of *left* skew; reversing ρ or λ gives a
    # different (right skew) structure that the banking axioms do not claim.
    assert Lam(X).dom == I @ X and Lam(X).cod == X
    assert Rho(X).dom == X and Rho(X).cod == X @ I
    assert Alpha(X, Y, Z).dom == (X @ Y) @ Z
    assert Alpha(X, Y, Z).cod == X @ (Y @ Z)


def _test_no_inverse_generators_exist():
    # Fails under: adding an inverse constructor, or a dagger. Stated as
    # a test rather than a comment because "you cannot build this" is the
    # central structural claim and deserves to be checkable.
    names = {c.__name__ for c in Term.__subclasses__()}
    assert names == {"Id", "Then", "Tensor", "Lam", "Rho", "Alpha"}, names
    assert not hasattr(Term, "dagger")


def _test_frontier_is_invariant():
    # Every generator preserves ∂, so every term does. A term whose
    # frontier changed would be a typing bug upstream.
    # Fails under: Tensor swapping left/right in dom or cod.
    for _, lhs, rhs in AXIOMS + MUST_NOT_IDENTIFY:
        for t in (lhs, rhs):
            assert frontier(t.dom) == frontier(t.cod), t


def _test_axiom_pairs_are_parallel():
    # An equation between non-parallel terms is malformed, and would
    # silently pass a normaliser test that only compared normal forms.
    # Fails under: a mis-oriented triangle in _axioms().
    for name, lhs, rhs in AXIOMS:
        assert is_parallel(lhs, rhs), (name, lhs.dom, lhs.cod, rhs.dom, rhs.cod)


def _test_distinct_pairs_are_parallel():
    # These must be parallel or they are not a coherence failure at all —
    # two maps with different types being unequal is uninteresting.
    # Fails under: MUST_NOT_IDENTIFY entries built with wrong objects.
    for name, lhs, rhs in MUST_NOT_IDENTIFY:
        assert is_parallel(lhs, rhs), name


def _test_frontier_is_not_sufficient():
    # The crux. Every pair in NO_MAP_EXISTS has equal frontier and no
    # connecting map, so the frontier — which is precisely what a
    # "free skew monoidal category on strings" would retain — cannot
    # decide existence of a morphism.
    # Fails under: nothing in this file; this is a claim about Fsk(At)
    # that `skew_seq.py` must discharge. Recorded here so the obligation
    # is written down before the decision procedure exists.
    for name, a, b in NO_MAP_EXISTS:
        assert frontier(a) == frontier(b), name
        assert a != b, name


def _test_syntactic_equality_is_not_semantic():
    # Documents the current gap deliberately, so this test starts failing
    # — correctly — the moment skew_seq.py lands and equality moves to
    # the focused normal form.
    f = Lam(X)
    assert Then(f, Id(X)) != f
    assert Then(Id(f.dom), f) != f


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
    print(f"{len(AXIOMS)} equations the normaliser must respect")
    print(f"{len(MUST_NOT_IDENTIFY)} parallel pairs it must keep distinct")
    print(f"{len(NO_MAP_EXISTS)} homsets it must report empty")


if __name__ == "__main__":
    _run()
