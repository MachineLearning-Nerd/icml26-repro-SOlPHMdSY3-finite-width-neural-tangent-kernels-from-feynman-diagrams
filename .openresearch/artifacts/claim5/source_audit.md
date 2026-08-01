# Claim 5 source audit

Source: arXiv:2508.11522v4 source tar, SHA-256 `5237b9fec2f128b23266771acbc8e44837619d5309cd32840ce537071b64c47f`, retrieved 2026-08-01.

Section 6 and Figure 1 state that a bias-free ReLU MLP has width 200, is evaluated through 30 layers, and uses sample means from 1,000 initializations. At `C_W=2`, both diagonal and distinct-input NTK components scale linearly; lower and higher values give exponential decay and growth. Appendix D gives the exact two inputs and the diagonal prediction `Theta_aa^(l)=||x_a||^2 l/n_0`. Appendix C permits output-channel trace averaging to improve sample efficiency.

The claim is finite and empirical. It is not construed as a universal theorem over architectures, nonlinearities, widths, or inputs.
