# Claim 4 limitations and deviations

The analytic values are recovered from the paper's vector-PDF curve because the v4 article does not tabulate them. The author code commit is audited separately, but the current cumulative environment does not add its GPU-oriented `jax[cuda12-local]` dependency. The coordinate consistency check bounds digitization spread; this is weaker than independently re-solving every numerical Gaussian integral.

The estimator uses two unbiased Hutchinson probes rather than materializing full Jacobians. The experiment covers four representative paper widths above 20 rather than all 11 plotted widths. A descendant separately tests the judge's conflated 5-million-sample ReLU statement.
