"""
test_interp.py — the ∆⊥ model against the term layer and the sequent
calculus. Plain asserts, no framework, as in `cqm`. Run with
`python test_interp.py`.
"""

from __future__ import annotations

from interp import equal_in_delta, interp, is_delta_bot_map, size
from skew_seq import _formulae, equal, homset
from skew_terms import (
    AXIOMS,
    MUST_NOT_IDENTIFY,
    Alpha,
    Atom,
    I,
    Id,
    Lam,
    Rho,
    Tens,
    Tensor,
    Then,
    X,
    Y,
)


def _test_structure_maps_are_lack_street():
    # Lack & Street §3: λ is σ₀, ρ is δₙ, α is the identity.
    # Fails under: λ collapsing the top pair instead of the bottom, or ρ
    # missing the bottom element (not bottom-preserving, so not in ∆⊥).
    assert interp(Lam(X)) == (0, 0)
    assert interp(Lam(Tens(X, Y))) == (0, 0, 1)
    assert interp(Rho(X)) == (0,)
    assert interp(Rho(Tens(X, Y))) == (0, 1)
    assert interp(Alpha(X, Y, Tens(X, I))) == (0, 1, 2, 3)


def _test_axioms_hold_in_delta():
    # The functor half: every equation of the theory must hold in ∆⊥, or
    # `interp` is not a structure-preserving functor and nothing else here
    # means anything.
    # Fails under: a wrong λ or ρ, or Tensor shifting by the domain width
    # instead of the codomain width.
    for name, lhs, rhs in AXIOMS:
        assert interp(lhs) == interp(rhs), (name, interp(lhs), interp(rhs))


def _test_must_not_identify_separates():
    # Must-not-fire, in the model. Each pair is distinct in Fsk; a
    # faithful functor must keep it distinct. Lack & Street §11 use exactly
    # this to show ε₀ = λ_I then ρ_I is not the identity.
    # Fails under: λ collapsing the top pair instead of the bottom
    # (mutation-tested). Not under ρ as δ₀: that mutant is caught by the
    # axiom and type tests instead, and leaves these pairs distinct.
    for name, lhs, rhs in MUST_NOT_IDENTIFY:
        assert interp(lhs) != interp(rhs), name


def _test_images_are_delta_bot_maps():
    # Every image is a legal ∆⊥ morphism of the right type.
    # Fails under: any off-by-one in Tensor's shift, which produces values
    # outside the codomain or a non-monotone tuple.
    for A in _formulae(2, [X], with_unit=True):
        for C in _formulae(2, [X], with_unit=True):
            for h in homset(A, C):
                assert is_delta_bot_map(interp(h), size(A), size(C)), (A, C, h)


def _test_faithful_on_two_atoms():
    # Lack & Street Cor. 11.4 with Thm 12.2: distinct maps of Fsk(At)
    # have distinct images. Swept over every homset between two-atom
    # formulae of depth 2; each homset is enumerated without duplicates
    # by `skew_seq.homset` (UVZ Thm 6), so distinct entries are distinct
    # maps and must stay distinct here.
    # Fails under: λ sent to the constant-0 map, or a homset enumeration
    # that returns a duplicate entry (both mutation-tested).
    fs = _formulae(2, [X, Y], with_unit=True)
    checked = 0
    for A in fs:
        for C in fs:
            hs = homset(A, C)
            if len(hs) < 2:
                continue
            images = [interp(h) for h in hs]
            assert len(set(images)) == len(images), ("not faithful", A, C)
            checked += 1
    assert checked > 20, f"faithfulness sweep too thin: {checked}"


def _test_two_routes_agree():
    # The cross-check the split exists for: `equal` (focused derivations)
    # and `equal_in_delta` (functions between ordinals) must agree on
    # every parallel pair, in both directions. Composites are used so the
    # pairs are not already in either procedure's normal form.
    # Fails under: a bug in either procedure, which shows up here as a
    # disagreement rather than as a silently wrong answer.
    fs = _formulae(2, [X], with_unit=True)
    checked = 0
    for A in fs:
        for C in fs:
            hs = homset(A, C)
            if not hs:
                continue
            for B in fs:
                for f in homset(A, B)[:2]:
                    for g in homset(B, C)[:2]:
                        t = Then(f, g)
                        for h in hs:
                            assert equal(t, h) == equal_in_delta(t, h), (t, h)
                            checked += 1
    assert checked > 500, f"two-route check too thin: {checked}"


def _test_lack_street_worked_example():
    # Lack & Street §11: (ρ⊗1)⊗1 ; α⊗1 ; α ; 1⊗(λ⊗1) equals α_{W,XI,Y},
    # checked there in ∆⊥ as δ₁ then σ₁. Here by both routes.
    W = Atom("W")
    t = Then(
        Tensor(Tensor(Rho(W), Id(Tens(X, I))), Id(Y)),
        Then(
            Tensor(Alpha(W, I, Tens(X, I)), Id(Y)),
            Then(
                Alpha(W, Tens(I, Tens(X, I)), Y),
                Tensor(Id(W), Tensor(Lam(Tens(X, I)), Id(Y))),
            ),
        ),
    )
    a = Alpha(W, Tens(X, I), Y)
    assert equal(t, a) and equal_in_delta(t, a)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    _run()
