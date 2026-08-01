# Claim 5 — VERIFIED

![Depth stability](../reports/reproduction/images/claim5_depth_stability.svg)

## Exact source contract

Section 6 Figure 1: a bias-free ReLU MLP of width 200 is evaluated from depth 1
through 30 using means over 1,000 initializations. At critical `C_W=2`, diagonal
and distinct-input NTKs scale linearly; lower/higher `C_W` give exponential decay
and growth. Appendix D gives `Theta_aa^(l)=||x_a||^2 l/n_0`; Appendix C permits
output-channel trace averaging. This is a finite empirical architecture claim,
not a universal theorem.

## Results

[Paper-scale verifier](../code/claim5_stability.py) used the exact width, depths,
source inputs, and 1,000 networks per low/critical/high regime, with deterministic
seeds and two Hutchinson probes. At `C_W=2`:

- predicted diagonal slope: 0.669128;
- observed through-origin slope: 0.679499 ± 0.012189 (`z=0.851`);
- linear regression `R²=0.999900`;
- all 30/30 depths lie in the 99% pointwise intervals;
- maximum relative deviation is 2.519%.

The low-control log-slope is -2.02515 and the high-control log-slope is +0.74685;
the high curve is explicitly rejected by the critical contract. The
[independent checker](../code/claim5_independent_checker.py) reconstructs the
critical formula separately. Full depth-wise means, standard errors, seeds, and
runtimes: [raw JSON](../data/cumulative_run.json).

## Limits

Hutchinson trace estimation is unbiased but adds variance; reported SEs include
network variation for the chosen two-probe estimator. Only the paper's stated
inputs and three initialization regimes are assessed.
