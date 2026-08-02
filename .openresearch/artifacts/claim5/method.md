# Claim 5 method

The verifier implements the paper's bias-free standard parameterization directly. For each of 1,000 independent networks per `C_W`, all 30 width-200 preactivation layers are evaluated for the paper's exact inputs.

The paper-permitted output-channel trace-average NTK is estimated with four independent Rademacher parameter-space probes per network. A JAX JVP gives an unbiased Hutchinson estimate of the full parameter-Jacobian inner product, averaged over all 200 output channels. Ensemble means, sample standard deviations, standard errors, regressions, and fixed pass thresholds are emitted as JSON.

An independent width-3, depth-3 checker compares JAX's Jacobian to a NumPy central finite difference and compares the Hutchinson estimator to the exact Jacobian trace. The `C_W=4` curve is a negative control and must fail the critical contract.
