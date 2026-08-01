# Claim 5 limitations and deviations

The architecture, width, depth, inputs, number of network initializations, and initialization regimes are paper-scale. The only estimator substitution is unbiased Hutchinson parameter probing instead of materializing the complete Jacobian. Four probes are used per network and their uncertainty contributes to the reported ensemble standard errors. Appendix C's output-channel trace average is used rather than an arbitrary fixed channel.

The run does not reproduce the paper's additional NNGP, V, A, B, D, or F stability plots; they are not part of the imported Claim 5 contract.
