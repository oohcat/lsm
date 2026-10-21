"""
skew_seq.py — the focused sequent calculus for Fsk(At), and the decision
procedures it yields.

Sibling of `skew_terms.py`, which holds the objects, the terms, and the
obligations this file must discharge. Nothing here imports NumPy; this is
the analogue of `cqm/strict_terms.py`'s `normalise()`, not of `interp.py`.

WHY A SEQUENT CALCULUS AND NOT A REWRITE SYSTEM
-----------------------------------------------
`cqm.normalise()` rewrites terms toward a canonical layer sequence, and
is sound but incomplete: it has no interchange rule, so two equal terms
can normalise apart. The obvious move here would be to write the same
kind of rewriter for the skew equations. UVZ show why that is the wrong
shape of solution and what the right one is: directing the equations on
cut-free derivations left-to-right is weakly confluent and strongly
normalising, and the resulting normal forms then turn out to admit a
*direct description* — the focused subsystem of Figure 8. So the
normal forms are characterised rather than searched for, and the
procedure is complete by construction instead of by accumulating rules
until the counterexamples stop arriving.

The consequence is that skew equality is decidable *and* complete, which
is strictly better than the strict project manages. That is not because
skew is easier; it is because this presentation was solved in the
literature and `cqm`'s was not looked up.

WHAT IS CHECKED
---------------
Everything below is transcribed from Uustalu, Veltri & Zeilberger,
*The Sequent Calculus of Skew Monoidal Categories* (arXiv:2003.05213),
read in the source:

  - §3, Figure 4: the sequent calculus (id, shift, scut, ccut, IL, ⊗L,
    IR, ⊗R) and the stoup discipline — left rules apply only to the
    formula in the stoup.
  - §3, the interpretation ⟦S | Γ⟧ of an antecedent as the left-nested
    tensor ((S⊗A₁)⊗A₂)…⊗Aₙ, with the empty stoup read as I.
  - §3, Lemmata 1–4 (the ⟦f | Γ⟧, ψ and φ constructions) and Theorem 2
    (soundness). Transcribed as `bracket`, `psi`, `phi`, `sound`.
  - §5, Figure 8: the focused subsystem, with phase annotation. In an
    L-sequent the stoup is unrestricted; in an R-sequent it is forced to
    be irreducible (empty or atomic).
  - §5, Lemma 17: admissibility of id, ⊗R, scut and ccut for L-phase
    sequents, by mutual lexicographic induction. Transcribed as
    `id_L`, `tensR_L`, `scut`, `scut_R`, `ccut`, `ccut_R`.
  - §5, Lemma 21 and Theorem 6: root-first search terminates and
    enumerates each homset without duplicates. Transcribed as
    `focderivs` / `homset`.
  - §5, Theorem 5 (coherence: equality). Transcribed as `equal`.

Also checked: Uustalu, *Coherence for skew-monoidal categories*, MSFP 2014,
§3, Proposition 6 — every object has exactly one map into its normal form
X₁⊗(X₂⊗(…⊗(Xₙ⊗I))), and dually exactly one map out of its reverse normal
form ((I⊗X₁)⊗…)⊗Xₙ. A sufficient condition for uniqueness only; Uustalu
notes that Lack–Street give the necessary and sufficient one, which is
what `interp.py` uses. Tested in `_test_uustalu_normal_forms`, an
independent prediction nothing in this file was built to satisfy.

Formerly NOT checked, now closed: the orientation of triangles (b), (c)
and (d) in `skew_terms.AXIOMS`. They were reconstructed rather than read,
and `_test_axioms_hold` was written to fail loudly if a reconstruction was
wrong. It passed, and the orientations have since been read in three
sources (see the `skew_terms` docstring).

A NOTE ON THE DERIVABILITY AUDIT
--------------------------------
`skew_terms.DERIVABILITY_AUDIT` asks, for each structure map at a
bracketed object, whether naming it as a primitive smuggles in a rule —
the failure `cqm`'s composite `Cup` turned out to exhibit.

This calculus dissolves the question rather than answering it. There are
no λ, ρ or α rules here at all: the six rules are id, shift, IL, ⊗L, IR
and ⊗R, and every structure map at every object is *derived* (UVZ's
derivations (1)–(3)). So no instance of λ, ρ or α can smuggle anything
in, at any object, because none of them is a primitive to begin with.

What remains checkable, and is checked, is the sharper question the
audit was really after: whether each such instance is the *unique* map
of its type. Where the homset is a singleton the map is forced and
carries no choice; where it is not, the choice is real and must be made
explicitly. `_test_derivability_audit` reports both, and the result is
not uniform — see the note there.

Run with `python skew_seq.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from skew_terms import (
    AXIOMS,
    DERIVABILITY_AUDIT,
    MUST_NOT_DERIVE_FROM_B_AND_E,
    MUST_NOT_IDENTIFY,
    NO_MAP_EITHER_WAY,
    NO_MAP_EXISTS,
    Alpha,
    Atom,
    Fma,
    I,
    Id,
    Lam,
    Rho,
    Tens,
    Tensor,
    Term,
    Then,
    Unit,
    X,
    Y,
    Z,
    frontier,
)

Ctx = Tuple[Fma, ...]
Stoup = Optional[Fma]  # None is UVZ's empty stoup "-"


# ---------------------------------------------------------------------------
# Focused derivations (UVZ Figure 8)
# ---------------------------------------------------------------------------
#
# Each node computes its own sequent from its children rather than storing
# one, so a derivation cannot carry a sequent that disagrees with its own
# structure. This is the same discipline as `Term.dom`/`Term.cod`.
#
# Phase is a property, not a field: R-phase nodes are exactly IdF, IRF and
# TensRF, and `Switch` is the only way down from L to R.


class Foc:
    """A focused derivation of `stoup | ctx ⟶_phase succ`."""

    @property
    def stoup(self) -> Stoup:
        raise NotImplementedError

    @property
    def ctx(self) -> Ctx:
        raise NotImplementedError

    @property
    def succ(self) -> Fma:
        raise NotImplementedError

    @property
    def phase(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, repr=False)
class IdF(Foc):
    """id^foc : X | ⟶_R X, at an *atom* only."""

    atom: Atom

    @property
    def stoup(self) -> Stoup:
        return self.atom

    @property
    def ctx(self) -> Ctx:
        return ()

    @property
    def succ(self) -> Fma:
        return self.atom

    @property
    def phase(self) -> str:
        return "R"

    def __repr__(self) -> str:
        return f"id^foc[{self.atom}]"


@dataclass(frozen=True, repr=False)
class IRF(Foc):
    """IR^foc : - | ⟶_R I."""

    @property
    def stoup(self) -> Stoup:
        return None

    @property
    def ctx(self) -> Ctx:
        return ()

    @property
    def succ(self) -> Fma:
        return I

    @property
    def phase(self) -> str:
        return "R"

    def __repr__(self) -> str:
        return "IR^foc"


@dataclass(frozen=True, repr=False)
class TensRF(Foc):
    """⊗R^foc : from T|Γ₁ ⟶_R C₁ and -|Γ₂ ⟶_L C₂, infer T|Γ₁,Γ₂ ⟶_R C₁⊗C₂.

    The asymmetry is the point and is easy to get backwards: the left
    premise stays in R-phase, the right premise drops to L-phase and has
    a forced-empty stoup. That forced-empty stoup is exactly what blocks
    λ⁻¹ — see the `NO_MAP_EXISTS` discussion in `skew_terms`.
    """

    left: Foc
    right: Foc

    def __post_init__(self) -> None:
        if self.left.phase != "R":
            raise TypeError("⊗R^foc: left premise must be in R-phase")
        if self.right.phase != "L":
            raise TypeError("⊗R^foc: right premise must be in L-phase")
        if self.right.stoup is not None:
            raise TypeError("⊗R^foc: right premise must have empty stoup")

    @property
    def stoup(self) -> Stoup:
        return self.left.stoup

    @property
    def ctx(self) -> Ctx:
        return self.left.ctx + self.right.ctx

    @property
    def succ(self) -> Fma:
        return Tens(self.left.succ, self.right.succ)

    @property
    def phase(self) -> str:
        return "R"

    def __repr__(self) -> str:
        return f"⊗R^foc({self.left}, {self.right})"


@dataclass(frozen=True, repr=False)
class Switch(Foc):
    """switch : R-phase to L-phase. Conclusion equals premise."""

    body: Foc

    def __post_init__(self) -> None:
        if self.body.phase != "R":
            raise TypeError("switch: premise must be in R-phase")

    @property
    def stoup(self) -> Stoup:
        return self.body.stoup

    @property
    def ctx(self) -> Ctx:
        return self.body.ctx

    @property
    def succ(self) -> Fma:
        return self.body.succ

    @property
    def phase(self) -> str:
        return "L"

    def __repr__(self) -> str:
        return f"switch({self.body})"


@dataclass(frozen=True, repr=False)
class Shift(Foc):
    """shift : from A|Γ ⟶_L C, infer -|A,Γ ⟶_L C."""

    body: Foc

    def __post_init__(self) -> None:
        if self.body.phase != "L" or self.body.stoup is None:
            raise TypeError("shift: premise must be L-phase with a stoup")

    @property
    def stoup(self) -> Stoup:
        return None

    @property
    def ctx(self) -> Ctx:
        return (self.body.stoup,) + self.body.ctx

    @property
    def succ(self) -> Fma:
        return self.body.succ

    @property
    def phase(self) -> str:
        return "L"

    def __repr__(self) -> str:
        return f"shift({self.body})"


@dataclass(frozen=True, repr=False)
class ILF(Foc):
    """IL : from -|Γ ⟶_L C, infer I|Γ ⟶_L C."""

    body: Foc

    def __post_init__(self) -> None:
        if self.body.phase != "L" or self.body.stoup is not None:
            raise TypeError("IL: premise must be L-phase with empty stoup")

    @property
    def stoup(self) -> Stoup:
        return I

    @property
    def ctx(self) -> Ctx:
        return self.body.ctx

    @property
    def succ(self) -> Fma:
        return self.body.succ

    @property
    def phase(self) -> str:
        return "L"

    def __repr__(self) -> str:
        return f"IL({self.body})"


@dataclass(frozen=True, repr=False)
class TensLF(Foc):
    """⊗L : from A|B,Γ ⟶_L C, infer A⊗B|Γ ⟶_L C."""

    body: Foc

    def __post_init__(self) -> None:
        if self.body.phase != "L" or self.body.stoup is None:
            raise TypeError("⊗L: premise must be L-phase with a stoup")
        if not self.body.ctx:
            raise TypeError("⊗L: premise context must be non-empty")

    @property
    def stoup(self) -> Stoup:
        return Tens(self.body.stoup, self.body.ctx[0])

    @property
    def ctx(self) -> Ctx:
        return self.body.ctx[1:]

    @property
    def succ(self) -> Fma:
        return self.body.succ

    @property
    def phase(self) -> str:
        return "L"

    def __repr__(self) -> str:
        return f"⊗L({self.body})"


# ---------------------------------------------------------------------------
# Admissible rules (UVZ Lemma 17)
# ---------------------------------------------------------------------------
#
# id, ⊗R, scut and ccut are not rules of the focused calculus. They are
# *admissible*: given focused derivations of the premises, these build a
# focused derivation of the conclusion. This is what makes `cmplt` below
# produce focused derivations directly, so no separate `focus` pass is
# needed.


def id_L(A: Fma) -> Foc:
    """A | ⟶_L A, by induction on A."""
    if isinstance(A, Atom):
        return Switch(IdF(A))
    if isinstance(A, Unit):
        return ILF(Switch(IRF()))
    if isinstance(A, Tens):
        return TensLF(tensR_L(id_L(A.left), Shift(id_L(A.right))))
    raise TypeError(f"not a formula: {A!r}")


def tensR_L(f: Foc, g: Foc) -> Foc:
    """From S|Γ ⟶_L A and -|Δ ⟶_L B, build S|Γ,Δ ⟶_L A⊗B.

    By induction on the *first* premise, pushing through the invertible
    left rules until it is a `switch`, at which point the genuine ⊗R^foc
    fires.
    """
    if isinstance(f, ILF):
        return ILF(tensR_L(f.body, g))
    if isinstance(f, TensLF):
        return TensLF(tensR_L(f.body, g))
    if isinstance(f, Shift):
        return Shift(tensR_L(f.body, g))
    if isinstance(f, Switch):
        return Switch(TensRF(f.body, g))
    raise TypeError(f"tensR_L: not an L-phase derivation: {f!r}")


def scut(f: Foc, g: Foc) -> Foc:
    """From S|Γ ⟶_L A and A|Δ ⟶_L C, build S|Γ,Δ ⟶_L C.

    Diagrammatic order: this is "f then g", UVZ's `g ∘ f`.
    """
    if isinstance(f, Shift):
        return Shift(scut(f.body, g))
    if isinstance(f, ILF):
        return ILF(scut(f.body, g))
    if isinstance(f, TensLF):
        return TensLF(scut(f.body, g))
    if isinstance(f, Switch):
        return scut_R(f.body, g)
    raise TypeError(f"scut: not an L-phase derivation: {f!r}")


def scut_R(f: Foc, g: Foc) -> Foc:
    """From T|Γ ⟶_R A and A|Δ ⟶_L C, build T|Γ,Δ ⟶_L C."""
    if isinstance(f, IdF):
        return g
    if isinstance(f, IRF):
        # A = I forces g to start with IL.
        assert isinstance(g, ILF), f"scut_R: expected IL, got {g!r}"
        return g.body
    if isinstance(f, TensRF):
        # A = A₁⊗A₂ forces g to start with ⊗L.
        assert isinstance(g, TensLF), f"scut_R: expected ⊗L, got {g!r}"
        return scut_R(f.left, ccut(f.right, g.body, 0))
    raise TypeError(f"scut_R: not an R-phase derivation: {f!r}")


def ccut(f: Foc, g: Foc, at: int) -> Foc:
    """From -|Γ ⟶_L A and S|Δ₀,A,Δ₁ ⟶_L C, build S|Δ₀,Γ,Δ₁ ⟶_L C.

    `at` is len(Δ₀): the position of A in g's context. Carrying the split
    as an index rather than as two tuples keeps the ⊗L case honest —
    that case *grows* Δ₀ on the left, and an index makes the off-by-one
    visible instead of hiding it in a tuple concatenation.
    """
    if isinstance(g, Shift):
        if at == 0:
            # A was the formula shift moved into the stoup.
            return scut(f, g.body)
        return Shift(ccut(f, g.body, at - 1))
    if isinstance(g, ILF):
        return ILF(ccut(f, g.body, at))
    if isinstance(g, TensLF):
        # ⊗L moved B into the front of the context, so A sits one later.
        return TensLF(ccut(f, g.body, at + 1))
    if isinstance(g, Switch):
        return Switch(ccut_R(f, g.body, at))
    raise TypeError(f"ccut: not an L-phase derivation: {g!r}")


def ccut_R(f: Foc, g: Foc, at: int) -> Foc:
    """From -|Γ ⟶_L A and T|Δ₀,A,Δ₁ ⟶_R C, build T|Δ₀,Γ,Δ₁ ⟶_R C."""
    if isinstance(g, TensRF):
        n = len(g.left.ctx)
        if at < n:
            return TensRF(ccut_R(f, g.left, at), g.right)
        return TensRF(g.left, ccut(f, g.right, at - n))
    # id^foc and IR^foc both have empty contexts, so A cannot occur.
    raise TypeError(f"ccut_R: impossible case {g!r}")


# ---------------------------------------------------------------------------
# Completeness: terms to focused derivations (UVZ Theorem 1, then Lemma 17)
# ---------------------------------------------------------------------------


def cmplt(f: Term) -> Foc:
    """Translate a categorical-calculus term to a focused derivation.

    The structure maps are *derived*, not primitive: λ, ρ and α each
    become a small derivation tree (UVZ derivations (1)–(3) of §3,
    generalised from atoms to arbitrary formulae). This is the concrete
    sense in which the composite-instance question of
    `skew_terms.DERIVABILITY_AUDIT` dissolves.
    """
    if isinstance(f, Id):
        return id_L(f.obj)
    if isinstance(f, Then):
        return scut(cmplt(f.first), cmplt(f.second))
    if isinstance(f, Tensor):
        return TensLF(tensR_L(cmplt(f.left), Shift(cmplt(f.right))))
    if isinstance(f, Lam):
        # λ_A : I⊗A ⟶ A   —   ⊗L, IL, shift, id
        return TensLF(ILF(Shift(id_L(f.obj))))
    if isinstance(f, Rho):
        # ρ_A : A ⟶ A⊗I   —   ⊗R(id, IR)
        return tensR_L(id_L(f.obj), Switch(IRF()))
    if isinstance(f, Alpha):
        # α : (A⊗B)⊗C ⟶ A⊗(B⊗C)   —   ⊗L, ⊗L, then ⊗R(id, shift ⊗R(id, shift id))
        inner = tensR_L(id_L(f.b), Shift(id_L(f.c)))
        return TensLF(TensLF(tensR_L(id_L(f.a), Shift(inner))))
    raise TypeError(f"cmplt: not a term: {f!r}")


# ---------------------------------------------------------------------------
# Soundness: focused derivations back to terms (UVZ Lemmata 1-4, Theorem 2)
# ---------------------------------------------------------------------------


def ant(S: Stoup, ctx: Ctx) -> Fma:
    """⟦S | Γ⟧ — the left-nested tensor, empty stoup read as I."""
    A = I if S is None else S
    for C in ctx:
        A = Tens(A, C)
    return A


def bracket(f: Term, ctx: Ctx) -> Term:
    """⟦f | Γ⟧ : ⟦A|Γ⟧ ⇒ ⟦B|Γ⟧ for f : A ⇒ B. UVZ Lemma 2."""
    for C in ctx:
        f = Tensor(f, Id(C))
    return f


def psi(A: Fma, B: Fma, ctx: Ctx) -> Term:
    """ψ_{A,B,Γ} : ⟦A⊗B|Γ⟧ ⇒ A ⊗ ⟦B|Γ⟧. UVZ Lemma 3."""
    if not ctx:
        return Id(Tens(A, B))
    C, rest = ctx[0], ctx[1:]
    return Then(bracket(Alpha(A, B, C), rest), psi(A, Tens(B, C), rest))


def phi(S: Stoup, gamma: Ctx, delta: Ctx) -> Term:
    """φ_{S,Γ,Δ} : ⟦S|Γ,Δ⟧ ⇒ ⟦S|Γ⟧ ⊗ ⟦-|Δ⟧. UVZ Lemma 4."""
    A = I if S is None else S
    while gamma:
        A, gamma = Tens(A, gamma[0]), gamma[1:]
    return Then(bracket(Rho(A), delta), psi(A, I, delta))


def sound(d: Foc) -> Term:
    """Translate a focused derivation to a term ⟦S|Γ⟧ ⇒ C. UVZ Theorem 2."""
    if isinstance(d, IdF):
        return Id(d.atom)
    if isinstance(d, IRF):
        return Id(I)
    if isinstance(d, (Switch, ILF, TensLF)):
        # For IL and ⊗L the interpretation of premise and conclusion are
        # literally equal; switch does not change the sequent at all.
        return sound(d.body)
    if isinstance(d, Shift):
        return Then(bracket(Lam(d.body.stoup), d.body.ctx), sound(d.body))
    if isinstance(d, TensRF):
        return Then(
            phi(d.left.stoup, d.left.ctx, d.right.ctx),
            Tensor(sound(d.left), sound(d.right)),
        )
    raise TypeError(f"sound: not a derivation: {d!r}")


# ---------------------------------------------------------------------------
# Enumeration and decision (UVZ Lemma 21, Theorems 5 and 6)
# ---------------------------------------------------------------------------


def focderivs(S: Stoup, ctx: Ctx, C: Fma, phase: str = "L") -> List[Foc]:
    """Every focused derivation of `S | ctx ⟶_phase C`, each exactly once.

    Root-first proof search, read straight off Figure 8. Terminates by
    UVZ's lexicographic rank on (occurrences of I and ⊗; stoup empty or
    not, with singleton < empty; phase, with R < L).
    """
    if phase == "L":
        if isinstance(S, Unit):
            return [ILF(d) for d in focderivs(None, ctx, C, "L")]
        if isinstance(S, Tens):
            return [
                TensLF(d)
                for d in focderivs(S.left, (S.right,) + ctx, C, "L")
            ]
        # Stoup is irreducible: empty or atomic.
        out: List[Foc] = []
        if S is None and ctx:
            out += [Shift(d) for d in focderivs(ctx[0], ctx[1:], C, "L")]
        out += [Switch(d) for d in focderivs(S, ctx, C, "R")]
        return out

    out = []
    if isinstance(C, Tens):
        for i in range(len(ctx) + 1):
            lefts = focderivs(S, ctx[:i], C.left, "R")
            if not lefts:
                continue
            rights = focderivs(None, ctx[i:], C.right, "L")
            for a in lefts:
                for b in rights:
                    out.append(TensRF(a, b))
    if isinstance(C, Atom) and S == C and not ctx:
        out.append(IdF(C))
    if isinstance(C, Unit) and S is None and not ctx:
        out.append(IRF())
    return out


def homset(A: Fma, C: Fma) -> List[Term]:
    """Every map A ⇒ C in Fsk(At), without duplicates. UVZ Theorem 6."""
    return [sound(d) for d in focderivs(A, (), C)]


def hom_exists(A: Fma, C: Fma) -> bool:
    return bool(focderivs(A, (), C))


def equal(f: Term, g: Term) -> bool:
    """Decide f ≐ g in Fsk(At). UVZ Theorem 5.

    `cmplt` already lands in the focused calculus, where each equivalence
    class has exactly one representative, so structural equality of the
    resulting derivation trees is the decision.
    """
    if f.dom != g.dom or f.cod != g.cod:
        return False
    return cmplt(f) == cmplt(g)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _test_derived_structure_maps_typecheck():
    # Fails under: any transcription error in cmplt's λ/ρ/α derivations,
    # or in the Foc nodes' computed sequents. The derivations are built
    # by hand from UVZ (1)-(3) and this is what catches a wrong one.
    for t in (Lam(X), Rho(X), Alpha(X, Y, Z), Lam(Tens(X, Y)), Rho(I)):
        d = cmplt(t)
        assert d.phase == "L", t
        assert d.stoup == t.dom, (t, d.stoup)
        assert d.ctx == (), t
        assert d.succ == t.cod, (t, d.succ)


def _test_sound_inverts_cmplt_on_types():
    # sound ∘ cmplt must at least land back on the same homset. A weaker
    # check than Lemma 9 (sound(cmplt f) ≐ f) but it is the part that can
    # be checked without already trusting `equal`.
    # Fails under: a mis-typed φ or ψ in the soundness translation.
    for _, lhs, rhs in AXIOMS:
        for t in (lhs, rhs):
            back = sound(cmplt(t))
            assert back.dom == t.dom and back.cod == t.cod, t


def _test_sound_cmplt_roundtrip_is_identity_on_classes():
    # UVZ Lemma 9: sound(cmplt f) ≐ f. Checked via `equal` rather than
    # structurally, which is the correct statement — the round trip is
    # identity on equivalence classes, not on trees.
    # Fails under: a soundness translation that lands in the wrong class,
    # e.g. φ built with λ where ρ belongs.
    for _, lhs, rhs in AXIOMS:
        for t in (lhs, rhs):
            assert equal(sound(cmplt(t)), t), t


def _test_axioms_hold():
    # The five Szlachányi axioms, the category laws, and interchange must
    # all be identified by the decision procedure.
    #
    # This was also how the reconstructed orientation of triangles (b),
    # (c) and (d) was first checked, before the sources were read: a
    # wrong orientation is a well-typed, parallel pair of terms that is
    # simply *not* an equation of Fsk, so it would fail here.
    # Fails under: a mis-oriented triangle, or a normaliser that quotients
    # too little.
    for name, lhs, rhs in AXIOMS:
        assert equal(lhs, rhs), f"axiom does not hold: {name}"


def _test_must_not_identify():
    # The three parallel-but-distinct pairs of UVZ §2. Each holds in any
    # monoidal category, so a procedure that identifies them is deciding
    # monoidal equality and the whole skew claim is void.
    # Fails under: quotienting too hard — e.g. treating α as invertible,
    # or letting scut collapse a ⊗R against a mismatched left rule.
    for name, lhs, rhs in MUST_NOT_IDENTIFY:
        assert not equal(lhs, rhs), f"wrongly identified: {name}"


def _test_no_map_exists():
    # The frontier argument, discharged. Each pair has equal ∂ and an
    # empty homset, so the string a naive model would retain cannot
    # decide existence of a morphism.
    # Fails under: dropping the stoup restriction — an unrestricted ⊗L
    # makes all three derivable, which is precisely the monoidal calculus.
    for name, a, b in NO_MAP_EXISTS:
        assert frontier(a) == frontier(b), name
        assert not hom_exists(a, b), f"map should not exist: {name}"
        # ...and the other direction *does* exist, or the test is vacuous.
        assert hom_exists(b, a), f"expected the forward map: {name}"


def _test_no_map_either_way():
    # Uustalu §3 shows the first direction is empty; the second is a
    # finding of this procedure, recorded after a guess that it existed
    # failed. To stay non-vacuous without a forward map, both objects are
    # shown to have *some* map out — into their common normal form, which
    # Uustalu Prop. 6 guarantees — so the empty homsets are not an artefact
    # of a search that finds nothing from these objects at all.
    # Fails under: a search that loses derivations, e.g. dropping the
    # shift rule, through the non-vacuity check (mutation-tested). That it
    # also catches a dropped stoup restriction is argued, not tested: the
    # unrestricted calculus derives α⁻¹, and with it a map one way.
    for name, a, b in NO_MAP_EITHER_WAY:
        assert frontier(a) == frontier(b), name
        assert not hom_exists(a, b), f"{name}: forward map exists"
        assert not hom_exists(b, a), f"{name}: backward map exists"
        nf = _normal_form(frontier(a))
        assert hom_exists(a, nf) and hom_exists(b, nf), f"{name}: vacuous"


def _normal_form(atoms) -> Fma:
    """Uustalu's normal form: X₁⊗(X₂⊗(…⊗(Xₙ⊗I)))."""
    A: Fma = I
    for a in reversed(atoms):
        A = Tens(Atom(a), A)
    return A


def _reverse_normal_form(atoms) -> Fma:
    """Uustalu's reverse normal form: ((I⊗X₁)⊗X₂)…⊗Xₙ."""
    A: Fma = I
    for a in atoms:
        A = Tens(A, Atom(a))
    return A


def _test_uustalu_normal_forms():
    # Uustalu Prop. 6 and its dual: for every A, exactly one map
    # A ⇒ nf(A) and exactly one map rnf(A) ⇒ A. Nothing in this file was
    # written to make that true; it is a prediction from a different proof
    # (normalising rewrites, not focusing) about the same category.
    # Fails under: a focderivs that fabricates derivations (count > 1), or
    # one that misses the canonical normalising map (count 0).
    fs = _formulae(2, [X, Y], with_unit=True)
    for A in fs:
        atoms = frontier(A)
        assert len(homset(A, _normal_form(atoms))) == 1, ("into nf", A)
        assert len(homset(_reverse_normal_form(atoms), A)) == 1, ("out of rnf", A)


def _test_enumeration_is_duplicate_free():
    # UVZ Theorem 6 promises each homset without duplicates. If two
    # derivations coincide the promise is broken; if two *terms* coincide
    # while the derivations differ, `sound` has collapsed distinct classes.
    # Fails under: a focderivs that tries the same context split twice.
    for A, C in [
        (Tens(Tens(X, I), Y), Tens(Tens(X, I), Y)),
        (Tens(I, I), Tens(I, I)),
        (Tens(Tens(X, Y), Z), Tens(X, Tens(Y, Z))),
        (Tens(X, Tens(I, Y)), Tens(Tens(X, I), Y)),
    ]:
        ds = focderivs(A, (), C)
        assert len(set(ds)) == len(ds), (A, C, "duplicate derivations")
        ts = [sound(d) for d in ds]
        assert len(set(map(repr, ts))) == len(ts), (A, C, "collapsed terms")


def _test_homsets_are_not_thin():
    # Fsk is not a preorder: I⊗I has two endomaps. This is the fact that
    # kills any "strings with a preorder on them" model, and it is the
    # one place where skew departs from the Tamari order — which *is*
    # thin, because it has no unit.
    # Fails under: any model that records only source and target.
    assert len(homset(Tens(I, I), Tens(I, I))) == 2
    assert len(homset(Tens(Tens(X, I), Y), Tens(Tens(X, I), Y))) == 2
    # ...while the unit-free part stays thin, as the Tamari order says.
    assert len(homset(Tens(Tens(X, Y), Z), Tens(X, Tens(Y, Z)))) == 1


def _test_kelly_axioms_are_not_vacuous():
    # skew_terms.MUST_NOT_DERIVE_FROM_B_AND_E records that (a), (c), (d)
    # do not follow from (b) and (e) here, unlike the monoidal case.
    # This suite CANNOT discharge that: independence needs a model where
    # (b) and (e) hold and (a) fails, which is a countermodel task, not a
    # decision-procedure task. What is checked here is only the weaker
    # claim that each is a non-trivial equation — both sides are distinct
    # trees that the procedure genuinely has to work to identify.
    # Recorded as a partial check so the gap is visible rather than
    # papered over by a passing test.
    by_name = {n: (l, r) for n, l, r in AXIOMS}
    for name in MUST_NOT_DERIVE_FROM_B_AND_E:
        lhs, rhs = by_name[name]
        assert lhs != rhs, name
        assert cmplt(lhs) == cmplt(rhs), name


def _test_derivability_audit():
    # The audit owed by skew_terms, discharged. Two findings.
    #
    # First: nothing can be smuggled in at any object, because the
    # calculus has no structure-map primitives at all — λ, ρ and α are
    # derived from the six rules whether their arguments are atomic or
    # composite. The `cqm` failure mode has no foothold here.
    #
    # Second: the result is not uniform. Most instances are the unique map
    # of their type, but where a unit sits under a bracket of the domain
    # the homset can be larger, and naming the instance is a real choice.
    #
    # Two wrong generalisations have been made here and both are kept on
    # record. An earlier version asserted ρ_I : I ⇒ I⊗I was ambiguous
    # because axiom (a) is about ρ_I; it is a singleton. The correction
    # then over-generalised to "ambiguity lives only in the endo-homsets,
    # never at the structure maps", and asserted every audited instance
    # was unique. That is also false: λ_{I⊗X} : I⊗(I⊗X) ⇒ I⊗X has a second
    # map, id⊗λ_X (Uustalu, MSFP 2014, §3), and α_{X,I,Y} has α followed
    # by the idempotent of Lack–Street §2. The sizes are now recorded per
    # instance in DERIVABILITY_AUDIT rather than asserted uniformly.
    #
    # Fails under: a homset enumeration that misses maps, which would
    # make instances look spuriously forced; or one that invents them,
    # which would make forced instances look like choices.
    report = []
    for name, term, size in DERIVABILITY_AUDIT:
        maps = homset(term.dom, term.cod)
        assert maps, f"structure map not derivable: {name}"
        assert any(equal(m, term) for m in maps), f"not found: {name}"
        assert len(maps) == size, f"{name}: homset has {len(maps)}, recorded {size}"
        report.append((name, size))
    assert any(s == 1 for _, s in report) and any(s > 1 for _, s in report), \
        "audit no longer exercises both outcomes"
    _test_derivability_audit.report = report


def _formulae(depth: int, atoms, with_unit: bool) -> List[Fma]:
    """All formulae up to a tensor depth, for the sweeps below."""
    out: List[Fma] = list(atoms) + ([I] if with_unit else [])
    for _ in range(depth):
        out = list(dict.fromkeys(out + [Tens(a, b) for a in out for b in out]))
    return out


def _test_tamari_thinness_in_the_unit_free_fragment():
    # The strongest independent prediction available, because it is a
    # claim about this implementation that UVZ make and that nothing in
    # the construction above forces: drop the unit and Fsk becomes the
    # free skew SEMIgroup category, where there is at most one map
    # between any two objects — the Tamari order on bracketed words.
    #
    # It also locates the whole difficulty. Multiplicity in Fsk comes
    # from the unit and from nothing else, so a banking model owes an
    # account of what I *is* before any of the interesting structure is
    # even in play.
    # Fails under: a scut or ccut that fabricates derivations, which
    # would show up here as a second map where the Tamari order has one.
    uf = _formulae(2, [X, Y, Z], with_unit=False)
    assert len(uf) > 100, "sweep too small to be worth running"
    for A in uf:
        for C in uf:
            if frontier(A) != frontier(C):
                continue
            n = len(homset(A, C))
            assert n <= 1, f"unit-free homset not thin: {A} ⇒ {C} has {n}"


def _test_unit_is_what_creates_multiplicity():
    # The converse half, and a "must not fire" check on the test above:
    # if thinness held *with* the unit too, the previous test would be
    # passing vacuously and this calculus would be the Tamari order
    # rather than Fsk.
    # Fails under: any quotient that collapses the unit maps.
    wu = _formulae(2, [X], with_unit=True)
    thick = [
        (A, C)
        for A in wu
        for C in wu
        if frontier(A) == frontier(C) and len(homset(A, C)) > 1
    ]
    assert thick, "unit did not create multiplicity: this is not Fsk"


def _test_frontier_is_necessary():
    # UVZ §2: ∂A = ∂B is necessary for a map A ⇒ B. `NO_MAP_EXISTS`
    # covers the failure of sufficiency; this is the other direction,
    # swept rather than spot-checked.
    # Fails under: a ⊗R context split that loses or duplicates a formula.
    wu = _formulae(2, [X], with_unit=True)
    for A in wu:
        for C in wu:
            if frontier(A) != frontier(C):
                assert not hom_exists(A, C), f"map crosses frontiers: {A} ⇒ {C}"


def _test_equal_is_a_congruence():
    # Being an equivalence relation is not enough: `equal` must respect
    # composition and ⊗, or it is not deciding equality in a category.
    #
    # The witnesses have to be *distinct trees that are equal*, and the
    # only supply of those is AXIOMS. An earlier version of this test
    # built a list of assorted terms and looked for equal pairs among
    # them; the pairs it found were entries that happened to repeat, so
    # `t` and `u` were the same tree and every congruence instance was
    # `equal(Then(t,g), Then(t,g))`. It passed on 480 such instances
    # while checking nothing. The `checked` counter below exists because
    # that is exactly the failure a congruence test invites.
    # Fails under: an scut sensitive to the shape of its argument's
    # derivation rather than to its class.
    probes: List[Term] = [
        Id(X), Lam(X), Rho(X), Lam(I), Rho(I),
        Alpha(X, Y, Z), Alpha(X, I, Y), Id(Tens(X, I)),
    ]
    checked = 0
    for name, lhs, rhs in AXIOMS:
        assert lhs != rhs and equal(lhs, rhs), name
        for g in probes:
            if lhs.cod == g.dom:
                checked += 1
                assert equal(Then(lhs, g), Then(rhs, g)), (name, "post", g)
            if g.cod == lhs.dom:
                checked += 1
                assert equal(Then(g, lhs), Then(g, rhs)), (name, "pre", g)
            checked += 1
            assert equal(Tensor(lhs, g), Tensor(rhs, g)), (name, "⊗R", g)
            assert equal(Tensor(g, lhs), Tensor(g, rhs)), (name, "⊗L", g)
    assert checked > 50, f"congruence check too thin: {checked}"


def _test_stoup_discipline_is_what_blocks_inverses():
    # A "must not fire" test in the strongest form available: show that
    # the *only* thing standing between this calculus and the monoidal
    # one is the stoup restriction. With ⊗L allowed anywhere in the
    # antecedent, α⁻¹ is derivable (UVZ §3 shows the derivation). Here
    # the closest legal approach must fail.
    # Fails under: a Shift or TensLF that does not enforce its premise's
    # stoup, which would silently readmit the unrestricted left rule.
    try:
        TensLF(Switch(IdF(X)))  # premise has no context to draw B from
    except TypeError:
        pass
    else:
        raise AssertionError("⊗L accepted a premise with empty context")
    try:
        ILF(Shift(id_L(X)))  # premise's stoup is not empty after shift
    except TypeError:
        raise AssertionError("IL wrongly rejected a legal premise")


def _test_tensR_rejects_a_stoup_in_its_right_premise():
    # This guard is unreachable from every call path above — instrumented
    # across the whole suite, it fires zero times, because `focderivs`
    # always recurses with an empty stoup on the right and every other
    # construction site feeds it a `Shift` or a `Switch(IRF())`.
    #
    # That makes it look like `cqm`'s one surviving mutant, the redundant
    # wire check in `yank()`. It is not the same case, and the difference
    # is worth stating. `yank()`'s check was forced by a check elsewhere;
    # this one is forced by nothing — it is an invariant that every
    # caller happens to maintain. Drop it and `TensRF(IdF(X), id_L(Y))`
    # becomes well-formed and claims `X | ⟶ X⊗Y`, a map whose frontier
    # does not even match. So it is a load-bearing guard that no other
    # test exercises, which is a test gap and not an equivalent mutant.
    # Fails under: removing the empty-stoup check on ⊗R^foc.
    try:
        TensRF(IdF(X), id_L(Y))
    except TypeError:
        pass
    else:
        raise AssertionError("⊗R^foc accepted a right premise with a stoup")


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")

    report = _test_derivability_audit.report
    print(f"\nderivability audit — {len(report)} structure-map instances:")
    for name, size in report:
        print(f"  {'forced' if size == 1 else f'choice ({size} maps)':16} {name}")

    print("\nhomset sizes — multiplicity appears only where a unit sits")
    print("under a bracket:")
    for A, C in [
        (Tens(I, I), Tens(I, I)),
        (Tens(Tens(X, I), Y), Tens(Tens(X, I), Y)),
        (Tens(X, Tens(I, Y)), Tens(X, Tens(I, Y))),
        (Tens(Tens(X, Y), Z), Tens(Tens(X, Y), Z)),
        (Tens(Tens(X, Y), Z), Tens(X, Tens(Y, Z))),
        (Tens(X, Tens(Y, Z)), Tens(Tens(X, Y), Z)),
    ]:
        print(f"  {len(homset(A, C))}  {A} ⇒ {C}")


if __name__ == "__main__":
    _run()
