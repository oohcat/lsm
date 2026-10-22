"""
test_writer.py — the writer monad meets the hypotheses of Szlachányi
Proposition 7.2. Plain asserts, no framework. Run with
`python test_writer.py`.

Equation numbers are Szlachányi's, arXiv:1201.4981v2 §7, read from the page
images. Each is checked elementwise on samples for three monoids: free on
two letters (words up to length 2), Z/2, and the trivial monoid.
"""

from __future__ import annotations

from writer import R, TRIVIAL, Writer, a, cyclic_group, free_monoid, l_inv, r, tens

MONOIDS = [free_monoid("pr", 2), cyclic_group(2), TRIVIAL]
L, M, N = ["l0", "l1"], ["m0", "m1"], ["n0", "n1"]


def _id(x):
    return x


def _tensor(f, g):
    return lambda x: (f(x[0]), g(x[1]))


def _then(*fs):
    def h(x):
        for f in fs:
            x = f(x)
        return x
    return h


def _agree(f, g, dom, label):
    for x in dom:
        assert f(x) == g(x), (label, x, f(x), g(x))


def _test_monad_laws():
    # ω ∘ Oω = ω ∘ ωO, and ω ∘ ιO = id = ω ∘ Oι.
    # Fails under: ι(m) = (p, m) for a non-unit p, or ω multiplying
    # w·w·v (mutation-tested). Not under ω as v·w: that is the writer
    # monad of the opposite monoid, a legitimate monad.
    for W in MONOIDS:
        O = Writer(W)
        OOOM = O.obj(O.obj(O.obj(M)))
        _agree(_then(O.fmap(O.omega), O.omega),
               _then(O.omega, O.omega), OOOM, (W.name, "assoc"))
        _agree(_then(O.iota, O.omega), _id, O.obj(M), (W.name, "unit left"))
        _agree(_then(O.fmap(O.iota), O.omega), _id, O.obj(M),
               (W.name, "unit right"))


def _test_opmonoidal_65_67():
    # (65) a ∘ (OL ⊗ O^{M,N}) ∘ O^{L,M⊗N} = (O^{L,M} ⊗ ON) ∘ O^{L⊗M,N} ∘ Oa
    # (66) (O⁰ ⊗ ON) ∘ O^{R,N} ∘ Ol⁻¹ = l⁻¹_{ON}
    # (67) r_{OM} ∘ (OM ⊗ O⁰) ∘ O^{M,R} = Or_M
    # Fails under: O^{M,N} writing w·w on the second factor (caught by
    # (65)), or writing the log on the first factor only (caught by (66)).
    # Both mutation-tested.
    for W in MONOIDS:
        O = Writer(W)
        _agree(_then(O.op2, _tensor(_id, O.op2), a),
               _then(O.fmap(a), O.op2, _tensor(O.op2, _id)),
               O.obj(tens(L, tens(M, N))), (W.name, 65))
        _agree(_then(O.fmap(l_inv), O.op2, _tensor(O.op0, _id)),
               l_inv, O.obj(N), (W.name, 66))
        _agree(_then(O.op2, _tensor(_id, O.op0), r),
               O.fmap(r), O.obj(tens(M, R)), (W.name, 67))


def _test_opmonoidal_68_71():
    # (68) (ω ⊗ ω) ∘ O^{OM,ON} ∘ OO^{M,N} = O^{M,N} ∘ ω_{M⊗N}
    # (69) O⁰ ∘ ω_R = O⁰ ∘ OO⁰
    # (70) O^{M,N} ∘ ι_{M⊗N} = ι_M ⊗ ι_N
    # (71) O⁰ ∘ ι_R = id_R
    # Fails under: O^{M,N} writing w·w on the second factor (mutation-tested,
    # caught by (68)). (69) and (71) hold for any O⁰ into the one-point set,
    # so no mutant of O⁰ can fail them; they are checked for completeness.
    for W in MONOIDS:
        O = Writer(W)
        _agree(_then(O.fmap(O.op2), O.op2, _tensor(O.omega, O.omega)),
               _then(O.omega, O.op2),
               O.obj(O.obj(tens(M, N))), (W.name, 68))
        _agree(_then(O.omega, O.op0), _then(O.fmap(O.op0), O.op0),
               O.obj(O.obj(R)), (W.name, 69))
        _agree(_then(O.iota, O.op2), _tensor(O.iota, O.iota),
               tens(M, N), (W.name, 70))
        _agree(_then(O.iota, O.op0), _id, R, (W.name, 71))


def _test_samples_are_not_vacuous():
    # The free sample must contain a word with no inverse in the sample's
    # closure, or nothing downstream distinguishes it from a group.
    # Fails under: a free sample of words up to length 0 (argued, not
    # mutation-tested).
    W = MONOIDS[0]
    assert any(len(w) > 0 for w in W.elems)
    assert len(Writer(W).obj(Writer(W).obj(tens(M, N)))) > 100


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    _run()
