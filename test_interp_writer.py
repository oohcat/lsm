"""
test_interp_writer.py — lsm's obligations checked in the writer model.
Plain asserts, no framework. Run with `python test_interp_writer.py`.

Three choices of W, each chosen to witness a different piece:
  - free on {p, r}: λ, ρ and α all non-invertible;
  - Z/2: λ, ρ non-invertible, α invertible at atom components;
  - 1: the strict case, which belongs with `cqm`.
Equations are checked elementwise on samples (words up to length 2 for the
free monoid), so a pass is evidence on those samples, not a proof. For Z/2
and 1 the samples are the whole sets, so the bijectivity checks there are
exact.
"""

from __future__ import annotations

from interp_writer import WriterSkew
from skew_terms import AXIOMS, MUST_NOT_IDENTIFY, Alpha, Atom, I, Lam, Rho, Tens
from writer import TRIVIAL, cyclic_group, free_monoid

FREE, Z2 = free_monoid("pr", 2), cyclic_group(2)
X, Y, Z = Atom("X"), Atom("Y"), Atom("Z")


def _test_structure_maps_elementwise():
    # The composites from Szlachányi's formulas, against their elementwise
    # form in the module docstring.
    # Fails under: γ̇ built with O^{OL,M} writing the log on the first
    # factor only, η̇ using a non-unit element, λ sent to the identity, or
    # γ̇ with ω as v·w (all mutation-tested). The last is the only catcher
    # besides the prefix test: ω as v·w is the model for the opposite
    # monoid, so every axiom still holds.
    S = WriterSkew(FREE)
    al = S.interp(Alpha(X, Y, Z))
    x = ((("p",), ((("r",), "x0"), "y0")), "z0")   # ((w, ((v, a), b)), c)
    assert al(x) == ((("p", "r"), "x0"), ((("p",), "y0"), "z0"))
    assert S.interp(Rho(X))("x0") == (((), "x0"), ())
    assert S.interp(Lam(X))(((("p",), ()), "x0")) == "x0"


def _test_axioms_hold():
    # The model half: every equation of the theory holds, for each W.
    # Fails under: Tensor dropping the log of its left factor (caught here
    # only), η̇ using a non-unit element, or γ̇ with the inner log taken
    # from v (all mutation-tested). Not under γ̇ with ω as v·w: that is the
    # opposite monoid's model, and the axioms hold in it.
    for W in (FREE, Z2, TRIVIAL):
        S = WriterSkew(W)
        for name, lhs, rhs in AXIOMS:
            assert S.equal_on_sample(lhs, rhs), (W.name, name)


def _test_skew_at_the_unit():
    # MUST_NOT_IDENTIFY in the model: every pair stays distinct once W ≠ 1,
    # including ε₀ = λ_I ; ρ_I ≠ id on I⊗I. For W = 1 all four collapse,
    # which is why the strict case is not an illustration of skewness.
    # Fails under: λ sent to the identity (mutation-tested). No mutant of
    # α, ρ or ⊗ tried was caught here; those are caught by the axioms.
    for W in (FREE, Z2):
        S = WriterSkew(W)
        for name, lhs, rhs in MUST_NOT_IDENTIFY:
            assert not S.equal_on_sample(lhs, rhs), (W.name, name)
    S1 = WriterSkew(TRIVIAL)
    for name, lhs, rhs in MUST_NOT_IDENTIFY:
        assert S1.equal_on_sample(lhs, rhs), ("1", name)


def _test_lambda_rho_not_invertible():
    # λ_X is not injective and ρ_X is not surjective once W ≠ 1, so
    # neither has an inverse in the model.
    # Fails under: λ sent to the identity (mutation-tested), or W = 1
    # (argued).
    for W in (FREE, Z2):
        S = WriterSkew(W)
        lam = S.interp(Lam(X))
        imgs = [lam(x) for x in S.elems(Tens(I, X))]
        assert len(set(imgs)) < len(imgs), (W.name, "λ injective")
        rho = S.interp(Rho(X))
        hit = {rho(x) for x in S.elems(X)}
        assert any(y not in hit for y in S.elems(Tens(X, I))), (W.name, "ρ onto")


def _test_alpha_one_way_for_free_W():
    # For W free, every image of α_{X,Y,Z} has its inner log a prefix of
    # its outer log, and ((e, x), ((p, y), z)) violates that, so α is not
    # surjective. Injective on the sample. The prefix property is checked
    # on the sample and argued in general: the image is ((w·v, a), ((w, b), c)).
    # Fails under: γ̇ with the inner log taken from v, γ̇ with ω as v·w, or
    # O^{OL,M} writing the log on the first factor only (all
    # mutation-tested).
    S = WriterSkew(FREE)
    al = S.interp(Alpha(X, Y, Z))
    dom = S.elems(Tens(Tens(X, Y), Z))
    imgs = [al(x) for x in dom]
    assert len(set(imgs)) == len(imgs), "α not injective on sample"
    for (u, _a), ((w, _b), _c) in imgs:
        assert u[: len(w)] == w, ("inner log not a prefix", u, w)
    witness = (((), "x0"), ((("p",), "y0"), "z0"))
    assert not witness[0][0][: len(witness[1][0][0])] == witness[1][0][0]


def _test_alpha_invertible_for_group_W():
    # For W = Z/2 the samples are whole sets, so this is exact: α_{X,Y,Z}
    # is a bijection, while λ and ρ stay non-invertible (previous test).
    # Which W keep α one-way at atoms is a property of W: invertible iff W
    # is a group (argued, not tested beyond these two cases).
    # Fails under: O^{OL,M} writing the log on the first factor only
    # (mutation-tested). Not under γ̇ with the inner log taken from v,
    # which is still a bijection when W is a group.
    S = WriterSkew(Z2)
    al = S.interp(Alpha(X, Y, Z))
    imgs = {al(x) for x in S.elems(Tens(Tens(X, Y), Z))}
    cod = S.elems(Tens(X, Tens(Y, Z)))
    assert len(imgs) == len(S.elems(Tens(Tens(X, Y), Z))) == len(cod)
    assert imgs == set(cod)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    _run()
