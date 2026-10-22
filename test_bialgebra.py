"""
test_bialgebra.py — GF(2)[W] meets the bialgebra axioms. Plain asserts,
no framework. Run with `python test_bialgebra.py`.

Checked for three monoids: free on {p, r} (words up to length 2), Z/2, and
{e, a} with a·a = a, which is not a group. Every equation is checked on all
basis elements and on all sums of two of them, so that cancellation over
GF(2) is exercised. For the two finite monoids that is exact by linearity.
"""

from __future__ import annotations

import itertools

from bialgebra import (MonoidBialgebra, f_tensor, idempotent_monoid, lin,
                       scalar, tensor, vec)
from writer import cyclic_group, free_monoid

MONOIDS = [free_monoid("pr", 2), cyclic_group(2), idempotent_monoid()]


def _vectors(H):
    B = list(H.basis())
    return [vec(b) for b in B] + [vec(a, b) for a, b in itertools.combinations(B, 2)]


def _flat_left(x):   # ((a, b), c) -> (a, b, c)
    return lin(lambda t: vec((t[0][0], t[0][1], t[1])))(x)


def _flat_right(x):  # (a, (b, c)) -> (a, b, c)
    return lin(lambda t: vec((t[0], t[1][0], t[1][1])))(x)


def _id(x):
    return x


def _mul2(H, s, t):
    """Product in H ⊗ H: (a⊗b)(c⊗d) = m(a⊗c) ⊗ m(b⊗d), extended bilinearly.
    Built from H.m, not from W, so it multiplies the way H does."""
    return lin(lambda q: tensor(H.m(vec((q[0][0], q[1][0]))),
                                H.m(vec((q[0][1], q[1][1])))))(
        vec(*[(p, r) for p in s for r in t]))


def _test_algebra_laws():
    # m(m ⊗ id) = m(id ⊗ m) and m(u ⊗ x) = x = m(x ⊗ u).
    # Fails under: u as a non-unit basis element (mutation-tested, caught
    # here only). Not under m as v·w: that is k[W^op], also an algebra.
    for W in MONOIDS:
        H = MonoidBialgebra(W)
        vs = _vectors(H)
        for x, y, z in itertools.product(vs[:6], repeat=3):
            assert H.m(tensor(H.m(tensor(x, y)), z)) == \
                H.m(tensor(x, H.m(tensor(y, z)))), (W.name, "assoc", x, y, z)
        for x in vs:
            assert H.m(tensor(H.u(), x)) == x == H.m(tensor(x, H.u())), (W.name, "unit", x)


def _test_coalgebra_laws():
    # Coassociativity, (Δ ⊗ id)Δ = (id ⊗ Δ)Δ after re-bracketing, and the
    # counit laws (ε ⊗ id)Δ = id = (id ⊗ ε)Δ.
    # Fails under: Δ(w) = w ⊗ e, Δ(w) = w⊗w + e⊗e, or ε vanishing off e
    # (all mutation-tested).
    for W in MONOIDS:
        H = MonoidBialgebra(W)
        for x in _vectors(H):
            d = H.delta(x)
            assert _flat_left(f_tensor(H.delta, _id)(d)) == \
                _flat_right(f_tensor(_id, H.delta)(d)), (W.name, "coassoc", x)
            left = lin(lambda p: vec(p[1]) if H.eps(vec(p[0])) else vec())(d)
            right = lin(lambda p: vec(p[0]) if H.eps(vec(p[1])) else vec())(d)
            assert left == x == right, (W.name, "counit", x)


def _test_delta_and_eps_are_algebra_maps():
    # Δ(xy) = Δ(x)Δ(y), Δ(1) = 1 ⊗ 1, ε(xy) = ε(x)ε(y), ε(1) = 1.
    # Fails under: ε vanishing off e, Δ(w) = w⊗w + e⊗e, or lin taking
    # unions (all mutation-tested). Not under Δ(w) = w ⊗ e, which is still
    # multiplicative; the counit laws catch that one. Not under m as v·w,
    # the bialgebra of the opposite monoid.
    for W in MONOIDS:
        H = MonoidBialgebra(W)
        vs = _vectors(H)
        for x, y in itertools.product(vs, repeat=2):
            assert H.delta(H.m(tensor(x, y))) == _mul2(H, H.delta(x), H.delta(y)), \
                (W.name, "Δ mult", x, y)
            assert H.eps(H.m(tensor(x, y))) == H.eps(x) * H.eps(y) % 2, \
                (W.name, "ε mult", x, y)
        assert H.delta(H.u()) == tensor(H.u(), H.u())
        assert H.eps(H.u()) == 1


def _test_delta_copies_the_log_not_vectors():
    # The point of the linear model. Δ copies basis elements, the log, but
    # not vectors: for x = e + a, Δ(x) = e⊗e + a⊗a, which is not x ⊗ x.
    # Over GF(2) also (e + a)² = 0 in GF(2)[Z/2], so sums really cancel.
    # Fails under: lin taking unions instead of symmetric differences, or
    # Δ(w) = w ⊗ e (both mutation-tested).
    H = MonoidBialgebra(cyclic_group(2))
    x = vec(0, 1)
    assert H.delta(x) == vec((0, 0), (1, 1))
    assert H.delta(x) != tensor(x, x)
    assert H.m(tensor(x, x)) == vec()
    assert scalar(vec("anything", "anything")) == 0


def _test_samples_are_not_vacuous():
    # Fails under: a free sample of words up to length 0 (argued, not
    # mutation-tested).
    H = MonoidBialgebra(MONOIDS[0])
    assert len(_vectors(H)) > 20
    assert any(len(w) > 0 for w in H.basis())


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    _run()
