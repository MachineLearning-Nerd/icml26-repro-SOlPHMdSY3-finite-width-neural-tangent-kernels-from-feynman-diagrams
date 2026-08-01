# Claim 1 — VERIFIED

## Exact source contract

Section 4 rules (i)–(v) define external lines, cubic interactions, Gaussian
propagators, six selection rules, quartic `D/F/A/B` vertices, quadratic
mean-correction vertices, and the multiplication/summation translation. Theorem 1
(`theoremone`) says these rules uniquely determine the `D/F/A/B` layer recursions
at order `1/n`; equation `eq:F` is the explicit algebraic target. This verdict is
limited to that finite-order MLP claim, not the paper's broader all-orders theorem.

Source: arXiv `2508.11522v4`, Section 4 anchors `S4`/`S4.SS1`; HTML SHA-256
`62cc33fa20ebd0d07a64a0386b96318a3ae6341880c031689c139d6c8ffe409a`.

## Executable evidence

[Verifier and independent reconstruction](../code/claim1_verifier.py) enumerates
allowed external signatures, direct/propagated `F` terms, and the `D/F/A/B`
quartic rule. All six checks passed: unique external signatures, two admissible
`F` channel partitions, unique direct/propagated `F`, coverage of `D/F/A/B`,
translation to direct and recursive algebra, and rejection of an invalid unpaired
color.

The invalid-color case is the negative control: it must be rejected, so a verifier
that accepts every graph cannot pass. The cumulative [raw JSON](../data/cumulative_run.json)
contains the complete rule table, checker output, source anchors, and booleans.

## Limits

This is an independently executable reconstruction of the published finite set of
rules and the worked `F` recursion. It is not a machine-checked proof of every
all-orders statement in the paper.
