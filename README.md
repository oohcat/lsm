# lsm

A left skew monoidal implementation. Banking with categories, not
categories with banking.

That ordering of words is the whole thesis, as it is for `cqm`. The
usual approach takes banking as it is practised (accounts, ledgers,
payment rails, API schemas) and *notices* that some of its structure
looks categorical. This project inverts that: it starts from the
categorical structure itself (objects, morphisms, composition, tensor,
unit, and the structure maps α, λ, ρ, none of them required to be
invertible) and treats *that* as the theory. Banking shows up only as
models of the categorical terms, not as the substrate the terms are
built from. If banking really is left skew monoidal, the category should
be primary and any banking system one model among possible others, not
the thing being modelled.

## Direction

The definition is the only authority on what lsm contains. Banking
supplies models. Each illustration starts from a piece of the structure
("what makes λ go one way only?") and looks for a banking instance that
witnesses it, not the reverse. An illustration earns its place only if
it exercises skewness: it names the structure map it witnesses and shows
that the map has no inverse in the model. One that would work equally
well in the strict setting belongs with `cqm`.

When a banking situation seems to need something the definition does not
give (a partial tensor, linearity, a fixed order of checks), that belongs
to a particular model, or it shows the model is not skew monoidal. It is
never a reason to change lsm.

The near-term tool is a REPL: write a term, see its type, and find out
whether it can be written at all. Computing rewrites and deciding
equivalence are not priorities. Where they are needed they are done by
hand, and the REPL's job is to keep every hand-made step well-typed. If
left skew monoidal proves to be the right category for banking, the work
extends to rewriting and equivalence in the way DisCoPy does for the
monoidal case. Until then it does not.

`cqm` is the stepping stone. It worked out the term/interpretation split
and the REPL notation for the free strict monoidal category, in the
familiar setting of quantum mechanics. This project takes the method and
the notation, and none of the strict core.

One caveat governs everything downstream. Writing every known banking
example as a term here would show that left skew monoidal is expressive
enough. It would not show that banking *is* left skew monoidal. Any
category with at least as much structure would also express them, and a
corpus that never uses α⁻¹ only shows that nobody drew one. The claim has
content only through what it forbids and what it identifies: maps that
must not exist, and composites that must or must not be equal.

## What is written down, and what isn't

The rules are `cqm`'s, with one addition.

**Falsifiability before a test is written.** Every test names, in its
comment, an implementation that would make it fail. Where that claim has
been checked by mutation it says so; where it is only argued it says that.

**No comment that restates the code.** A comment states what the reader
cannot deduce from the code: verified versus assumed, chosen versus
forced, what a test would catch.

**Citation discipline.** A theorem number or section reference is written
down only after it has been read in the source. The register of what has
been checked is under [Citations](#citations). Anything not on it is
unchecked.

**Increments the maths dictates.** Each commit carries one change and one
justification, a cited result, named in the commit message. Undoing it is
one `git revert`, with no list of side effects to remember. Changes are
additive where possible, and nothing goes in ahead of a source that
licenses it, however useful it looks. The first source to look in is the
touchstone, UVZ (see [Citations](#citations)).

## Scope

**Left skew, in these directions:** λ_A : I⊗A → A, ρ_A : A → A⊗I,
α_{A,B,C} : (A⊗B)⊗C → A⊗(B⊗C). None has an inverse, and no inverse can
be built from the generators (UVZ §2). Right skew reverses all three (Szlachányi Def. 2.1) and
is left skew on the opposite category (Uustalu §2), so "left" is a claim
about which way the modelled morphisms point.

**Objects are bracketed trees,** not words. `(X⊗Y)⊗Z` and `X⊗(Y⊗Z)` are
different objects joined only by α, in one direction. Flattening brackets,
which is how `cqm` makes its core strict, is exactly what is denied here.

**No dagger.** A dagger would supply α† : A⊗(B⊗C) → (A⊗B)⊗C, which does not
exist (UVZ §2).

**No swap.** Terms are planar.

**No generating morphisms yet.** The free skew monoidal category here is
free on a set of atoms. Its only morphisms are built from identities,
composition, ⊗ and the three structure maps. See
[Not decided](#not-decided) for what the literature allows beyond that.

**Composition is diagrammatic.** `Then(f, g)` is "f, then g", which UVZ
and Mac Lane write `g ∘ f`. It is also how UVZ's cut rules read, so the
sequent calculus is transcribed rather than translated.

## The design

As in `cqm`, a morphism is a term: a tree over `Id`, `Then`, `Tensor`,
`Lam`, `Rho` and `Alpha`, with no numbers in it. Domains and codomains are
computed from the tree, never stored, so a term cannot carry a type its
structure contradicts. `Then` refuses to compose terms whose types do not
meet. That typing is what the REPL will rest on.

Interpretation is a separate file. The universal property that licenses
it is UVZ Proposition 1: any assignment of atoms in a skew monoidal
category extends uniquely to a strict monoidal functor out of the free
one.

## State

Six suites, plain `assert`, no framework. Each runs directly.

**`skew_terms.py`** (10 tests). Objects, terms and typing, and the
obligations stated as data before any procedure existed to meet them:

- `AXIOMS`: the five axioms, the category laws, interchange and
  naturality. Eleven equations.
- `MUST_NOT_IDENTIFY`: four parallel pairs that are distinct. Three are
  Lack–Street's idempotents ε₀, εˡ, εʳ, each against the identity. The
  fourth is Uustalu's λ_{I⊗X} ≠ id⊗λ_X.
- `NO_MAP_EXISTS`: three homsets that are empty, with equal frontiers and
  a map in the other direction.
- `NO_MAP_EITHER_WAY`: one pair with no map in either direction, from
  Uustalu §3.
- `DERIVABILITY_AUDIT`: nineteen structure-map instances, each with the
  size of its homset. Fourteen are forced by their type and five are not.
- `MUST_NOT_DERIVE_FROM_B_AND_E`: axioms (a), (c), (d), which do not
  follow from (b) and (e) here as they do in the monoidal case. Stated,
  not established; see [Not decided](#not-decided).

**`skew_seq.py`** (18 tests). The focused sequent calculus of UVZ §5,
with the decision procedures it yields: `equal` (Theorem 5), `homset`
(Theorem 6, every map without duplicates) and `hom_exists` (Lemma 21).
Every obligation above is discharged here. Two tests are predictions
from other proofs that nothing in the file was built to meet: Tamari
thinness of the unit-free fragment (Zeilberger, Thm 1.17), and exactly
one map into each normal form and out of each reverse normal form
(Uustalu, Prop. 6).

**`interp.py`, `test_interp.py`** (7 tests). Terms interpreted in ∆⊥,
finite non-empty ordinals with order- and bottom-preserving maps
(Lack–Street §3). That functor is faithful (Lack–Street Cor. 11.4), so
`equal_in_delta` is a second complete decision procedure that shares no
code with `skew_seq`. The tests check that the axioms hold in ∆⊥,
faithfulness on a two-atom sweep, and that the two procedures agree on
every parallel pair tried. Nothing in the term layer imports this file.

So for terms built only from structure maps, equality is decided twice,
completely, by routes that share nothing. That is stronger than `cqm`
manages, whose `equal` is sound but incomplete and whose FdHilb functor
is not faithful. It is not because skew is easier. This presentation was
solved in the literature.

**`writer.py`, `test_writer.py`** (4 tests). Data for a model, not yet a
model: the writer monad OM = W × M on (Set, ×, 1), with the diagonal
opmonoidal structure. The tests check the monad laws and Szlachányi's
(65)–(71), the hypotheses of his Proposition 7.2, elementwise on samples
for W free on two letters, Z/2 and 1. A pass is evidence on those
samples, not a proof. No source read states that this monad meets the
hypotheses; that is what the suite is for. Nothing in the term layer
imports these files.

**`interp_writer.py`, `test_interp_writer.py`** (6 tests). Terms
interpreted in the skew structure Szlachányi's left form (after Def. 7.3)
builds from that monad: A⊗B goes to (W × A) × B. Each structure map is
composed from the formula, not written out by hand. Unlike ∆⊥ this model
is not claimed faithful, so it decides nothing; it is a model in which
the structure maps can be inspected. Three choices of W witness three
different things:

- *W free:* every axiom holds, every `MUST_NOT_IDENTIFY` pair stays
  distinct, λ and ρ have no inverse, and α_{X,Y,Z} is injective but not
  onto. Its image keeps the inner log a prefix of the outer one, and
  ((e, x), ((p, y), z)) is not hit.
- *W = Z/2:* λ and ρ still have no inverse and every pair stays
  distinct, but α_{X,Y,Z} is a bijection. These samples are whole sets,
  so this is exact. That α at atom components is invertible exactly when W
  is a group is argued, not tested beyond these two cases.
- *W = 1:* every pair collapses. This is the strict case, which belongs
  with `cqm`.

As with the writer suite, a pass on the free monoid is evidence on
samples, not a proof.

**`bialgebra.py`, `test_bialgebra.py`** (5 tests). Data for a linear
model, not yet a model: the monoid bialgebra GF(2)[W], with Δ(w) = w ⊗ w
and ε(w) = 1, the counterpart of the writer monad in which only the log
is copied. Szlachányi §3 builds a skew structure from a bialgebroid; that
a k-bialgebra is one, with source and target the unit map, is argued, not
cited. The tests check the bialgebra axioms on every basis element and on
every sum of two, for W free on two letters, Z/2 and {e, a} with a·a = a,
which is not a group. For the finite monoids this is exact by linearity.
One test records what makes the model linear: Δ(e + a) is e⊗e + a⊗a, not
(e + a) ⊗ (e + a), so Δ copies basis elements but no vector. The skew
structure §3 then supplies is not built here. Nothing in the term layer
imports these files.

**Mutation.** There is no `mutate.py` yet. One-off checks have been run:
eight mutants against the first two suites when they were written, and
six against the three term-layer and ∆⊥ suites, with every test run on
its own under each. Every mutant was caught. Two test comments named the
wrong catcher and were corrected to what was observed. Five mutants of
the writer data were run against `test_writer.py`. Four were caught. One
passed, as expected: ω as v·w, which is the writer monad of the opposite
monoid. One of the four caught had been believed valid: writing the log
on the first factor only fails (66), and `writer.py` says so. Six
mutants of the interpretation were run against `test_interp_writer.py`,
and all six were caught. Building γ̇ with ω as v·w passes every axiom,
since it is the opposite monoid's model; only the elementwise test and
the prefix test catch it. Three test comments named the wrong catcher
and were corrected to what was observed. Six mutants of the bialgebra
data were run against `test_bialgebra.py`. Five were caught. One passed,
as expected: m as v·w, the bialgebra of the opposite monoid. On the first
run it was caught, because the test's product on H ⊗ H multiplied in W
rather than through H's own m; the test was corrected, and one comment
that named the wrong catcher was corrected with it.

## Not decided

Recorded so they are not rediscovered by accident, and kept out of the
code until a source licenses them.

**Generating morphisms (`Box`).** A REPL for writing down models needs
generating morphisms. The literature splits them in two:

- *Typed atom → atom, forming a category X:* covered. Lack–Street
  Thm 12.2: a morphism of the free skew monoidal category on X is a
  morphism φ of Fsk together with one X-morphism per atom wire,
  Xᵢ → Y_{φ(i)}.
- *Typed by formulae,* such as `pay : Account ⊗ Amount → Receipt ⊗ Account`:
  not covered by any source read. Bourke–Lack's skew multicategories are
  the natural home for many-input, one-output generators, but give no
  free construction on them.

Banking models would probably want the second kind. That is a fact about
those models, not a requirement on lsm: by the increment rule, only the
first can go in now. Typing alone would survive the second kind;
the decision procedures would not, and would have to refuse such terms.

**The notation layer.** `cqm`'s operators (`>>`, `@`, whiskering), its
Python-valid `repr` and its type-printing echo. They need only a
category and a functor ⊗, which a skew monoidal category has. On this
side the operators must keep the user's bracketing, and there is no
`.dag`.

**Independence of (a), (c), (d).** Stated by UVZ §2 and Uustalu §2.
Kelly's monoidal derivation (Thms 6–7) cancels against isomorphisms,
which is the step skewness removes, but no source read gives a
countermodel. ∆⊥ cannot be one: all five axioms hold in it.

**Parked threads.** Raised in discussion and deliberately not planned:

- *Swaps.* UML `par` on a shared participant interleaves events, which
  needs a swap or a partial commutation. On disjoint participants it is
  just ⊗.
- *Activations as wires.* What a sequence diagram commits to is the
  activation bar, not the lifeline. Nested bars form a stack, which fits
  the stoup. An idle stretch of lifeline is a candidate meaning for I. A
  conjecture only.
- *The corpus.* BIAN publishes sequence diagrams the REPL should be able
  to recreate, subject to the caveat under [Direction](#direction).
- *The Cat-Banking quotient.* As a functor out of these terms it would
  supply an empirical equivalence to test the equations against. It can
  only be lax monoidal.
- *Rewriting and equivalence for terms with boxes.*
- *A `mutate.py` port.*

## Cross-strand notes

From `cqm`: the term/interpretation split, an interpretation fixed by its
values on generators, two independent routes to the same answer, the
falsifiability filter, and mutation testing. Method and tooling cross
freely; subject matter does not. `cqm` stays strict, and nothing here
parametrises its core over which theory it is in.

## Citations

**The touchstone is UVZ.** Heunen & Vicary play this role for `cqm`:
everything traces there, and what cannot be traced there does not go in.
Here the term layer, the sequent calculus and both decision procedures
are transcriptions of Uustalu, Veltri & Zeilberger. Its scope is the
project's boundary: it covers the free skew monoidal category on a set
of atoms, and nothing with generating morphisms. The other sources below
are admitted for what UVZ does not cover: ∆⊥ and boxes typed atom → atom
(Lack–Street), the axioms at their origin (Szlachányi), and independent
cross-checks. Where they and UVZ overlap, UVZ is the reference cited.

Checked against the source, all attached to this project:

- **Uustalu, Veltri & Zeilberger**, *The sequent calculus of skew monoidal
  categories*, arXiv:2003.05213v1, the revised and extended version of
  the MFPS 2018 paper. §2: Figure 1, the directions, Proposition 1, the
  frontier, the non-existence and non-equality examples, laws (a)–(e).
  §3: Figure 4, the antecedent interpretation, Lemmata 1–4, Theorem 2.
  §5: Figure 8, Lemma 17, Lemma 21, Theorems 5 and 6.
- **Szlachányi**, *Skew-monoidal categories and bialgebroids*, Adv. Math.
  231 (2012), read as arXiv:1201.4981v2. Definition 2.1 and the left/right
  duality following it.
  §7: Proposition 7.2 (bimonad-induced right-monoidal structure, (81)–(84)), and the left-handed form stated after Definition 7.3 (M ⊙ N := OM ⊗ N), read from the page images. The left form is stated there, not proved; it follows by the reversal duality after Def. 2.1.
  §3: the right-monoidal structure on right R-modules induced by a right
  R-bialgebroid H, M ∗ N := M ⊗_{R₁} (N ⊗_{R₂} H), with η, ε and γ as
  given there (30), read from the page image. That this is right-monoidal
  is stated there ("easy to verify, using the bialgebroid axioms"), not
  proved. The remark following it, also read: γ is, up to isomorphism,
  the Galois map g ⊗ h ↦ h⁽¹⁾ ⊗ gh⁽²⁾, so the bialgebroid is a Hopf
  algebroid precisely when γ is invertible; "Hopf algebroid" is taken in
  the sense of Schauenburg, not read. The structure is right-monoidal; it
  is used here in left form by the reversal duality after Def. 2.1.
- **Lack & Street**, *Triangulations, orientals, and skew monoidal
  categories*, Adv. Math. 258 (2014), read as arXiv:1302.4488v1, whose
  numbering is cited. §2, (2.4)–(2.8) and the idempotents ε₀, εˡ, εʳ; §3,
  ∆⊥ and its structure maps; §11, Corollary 11.4 and the worked example;
  §12, Theorem 12.2.
- **Bourke & Lack**, *Free skew monoidal categories*, arXiv:1708.06087v3.
  §1–2, clubbability; Propositions 7.3–7.7, the axioms in this
  orientation.
- **Bourke & Lack**, *Skew monoidal categories and skew multicategories*,
  J. Algebra 506 (2018), read as arXiv:1708.06088v2. Theorem 6.1, tight
  and loose multimaps.
- **Uustalu**, *Coherence for skew-monoidal categories*, MSFP 2014, EPTCS
  153, pp. 68–77. §2, the axioms and handedness; §3, Proposition 6 and
  the examples.
- **Zeilberger**, *A sequent calculus for a semi-associative law*, LMCS
  15(1):9 (2019). Theorems 1.2, 1.6, 1.17; §1.6.
- **Kelly**, *On MacLane's conditions for coherence of natural
  associativities, commutativities, etc.*, J. Algebra 1 (1964), 397–402.
  Theorems 6 and 7. The diagrams did not survive text extraction; the
  numbered conditions were identified from the wording of the proofs.

Cited secondhand, not read: Lack–Street Theorem 12.2 rests on Kelly's
theory of clubs (*An abstract approach to coherence*, LNM 281, 1972),
which is not attached. *Braided skew monoidal categories*, TAC 35, would
be the first thing to read if swaps are taken up; not yet read, and its
authorship not checked.

Lack & Street, Skew monoidales, skew warpings and quantum categories, TAC 26 (2012) — the general skew-warping construction. Bruguières, Lack & Virelizier, Prop. 2.6, the source of Szlachányi Prop. 7.1.

Schauenburg, *Duals and doubles of quantum groupoids*, Contemp. Math. 267
(2000), 273–299 — the sense of "Hopf algebroid" in Szlachányi §3's
invertibility remark.

Heunen & Vicary, *Categories for Quantum Theory*, is attached for the
`cqm` strand. Nothing here is cited to it.
