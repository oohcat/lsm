"""
test_interp_bialgebra.py — lsm's obligations checked in the linear model.
Plain asserts, no framework. Run with `python test_interp_bialgebra.py`.

Four choices of W:
  - free on {p, r}: λ, ρ and α all non-invertible;
  - Z/2: λ, ρ non-invertible, α invertible at atom components;
  - {e, a} with a·a = a: not a group, so α not invertible at atoms;
  - 1: the strict case, which belongs with `cqm`.
Checks run on every basis element of the domain. For finite W that is
exact by linearity; for W free it is a sample of words.
"""

from __future__ import annotations

from bialgebra import idempotent_monoid, vec
from interp_bialgebra import ONE, BialgebraSkew
from skew_terms import AXIOMS, MUST_NOT_IDENTIFY, Alpha, Atom, I, Lam, Rho, Tens
from writer import TRIVIAL, cyclic_group, free_monoid

FREE, Z2, IDEM = free_monoid("pr", 2), cyclic_group(2), idempotent_monoid()
X, Y, Z = Atom("X"), Atom("Y"), Atom("Z")


def _test_structure_maps_are_section_3():
    # The transcription of §3's η, ε, γ after reversal, against their
    # basis form in the module docstring.
    # Fails under: γ swapping the inner and outer logs, γ dropping g, γ
    # multiplying h·g, η using a non-unit element, ε vanishing off e, or λ
    # sent to the identity (all mutation-tested). γ with h·g is caught here
    # and by the suffix test only: it is a valid model (the opposite
    # monoid's), but not §3's formula.
    S = BialgebraSkew(FREE)
    al = S.interp(Alpha(X, Y, Z))
    x = ("z0", ("y0", "x0", ("r",)), ("p",))      # (c, (b, a, g), h)
    assert al(vec(x)) == vec((("z0", "y0", ("p",)), "x0", ("r", "p")))
    assert S.interp(Rho(X))(vec("x0")) == vec((ONE, "x0", ()))
    assert S.interp(Lam(X))(vec(("x0", ONE, ("p",)))) == vec("x0")


def _test_axioms_hold():
    # The model half: every equation of the theory holds, for each W.
    # Fails under: ⊗ on maps dropping the log (caught here only), γ
    # swapping the inner and outer logs, γ dropping g, η using a non-unit
    # element, ε vanishing off e, or λ sent to the identity (all
    # mutation-tested). Not under γ with h·g, the opposite monoid's model.
    for W in (FREE, Z2, IDEM, TRIVIAL):
        S = BialgebraSkew(W)
        for name, lhs, rhs in AXIOMS:
            assert S.equal_on_basis(lhs, rhs), (W.name, name)


def _test_skew_at_the_unit():
    # MUST_NOT_IDENTIFY: every pair stays distinct once W ≠ 1, and all
    # collapse for W = 1.
    # Fails under: λ sent to the identity, which is also caught by the
    # axioms (mutation-tested).
    for W in (FREE, Z2, IDEM):
        S = BialgebraSkew(W)
        for name, lhs, rhs in MUST_NOT_IDENTIFY:
            assert not S.equal_on_basis(lhs, rhs), (W.name, name)
    S1 = BialgebraSkew(TRIVIAL)
    for name, lhs, rhs in MUST_NOT_IDENTIFY:
        assert S1.equal_on_basis(lhs, rhs), ("1", name)


def _test_lambda_rho_not_invertible():
    # λ_X has a nonzero kernel and ρ_X misses a basis element, once W ≠ 1.
    # Over GF(2): (x0, 1, e) + (x0, 1, w) goes to x0 + x0 = 0.
    # Fails under: λ sent to the identity, or ε vanishing off e (both
    # mutation-tested), or W = 1 (argued).
    for W in (FREE, Z2, IDEM):
        S = BialgebraSkew(W)
        lam = S.interp(Lam(X))
        w = next(v for v in W.elems if v != W.e)
        assert lam(vec(("x0", ONE, W.e), ("x0", ONE, w))) == vec(), W.name
        rho = S.interp(Rho(X))
        image = set().union(*(rho(vec(b)) for b in S.basis(X)))
        assert any(b not in image for b in S.basis(Tens(X, I))), W.name


def _alpha_permutes_basis(S):
    # α_{X,Y,Z} sends each basis element to one basis element (h is
    # group-like), so it is invertible iff that assignment is a bijection
    # of bases. Exact for finite W.
    al = S.interp(Alpha(X, Y, Z))
    images = [al(vec(b)) for b in S.basis(Tens(Tens(X, Y), Z))]
    assert all(len(v) == 1 for v in images)
    flat = [next(iter(v)) for v in images]
    return len(set(flat)) == len(flat) and set(flat) == set(S.basis(Tens(X, Tens(Y, Z))))


def _test_alpha_invertible_iff_group():
    # Szlachányi §3: γ is invertible exactly when the bialgebroid is Hopf.
    # Here, exactly for finite W: α at atoms is invertible for Z/2 and not
    # for {e, a}. That k[W] is Hopf iff W is a group is not cited; these
    # two cases are what is checked.
    # Fails under: γ dropping g, which is no longer injective even for
    # Z/2 (mutation-tested). Not under γ swapping the inner and outer logs:
    # that still permutes the basis for Z/2 and still fails for {e, a}.
    assert _alpha_permutes_basis(BialgebraSkew(Z2))
    assert not _alpha_permutes_basis(BialgebraSkew(IDEM))


def _test_alpha_one_way_for_free_W():
    # For W free, every image of α_{X,Y,Z} has the inner (b) log a suffix
    # of the outer (a) log: g·h ends with h. ((z, y, p), x, e) violates
    # that, so it is not in the image; since α sends basis elements to
    # distinct basis elements, it is not in the span of the image either.
    # Fails under: γ swapping the inner and outer logs, or γ multiplying
    # h·g, which makes the inner log a prefix rather than a suffix (both
    # mutation-tested).
    S = BialgebraSkew(FREE)
    al = S.interp(Alpha(X, Y, Z))
    for b in S.basis(Tens(Tens(X, Y), Z)):
        ((_c, _b, inner), _a, outer), = al(vec(b))
        assert outer[len(outer) - len(inner):] == inner, (inner, outer)
    witness = (("z0", "y0", ("p",)), "x0", ())
    assert not (witness[2][len(witness[2]) - len(witness[0][2]):] == witness[0][2])


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    _run()
