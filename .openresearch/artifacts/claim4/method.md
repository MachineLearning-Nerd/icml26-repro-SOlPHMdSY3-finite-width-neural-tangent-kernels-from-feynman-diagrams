# Claim 4 method

The reproduction uses the paper's exact GeLU, architecture, weight variance, inputs, double precision, and output-channel trace average. Four independently seeded ensembles at widths 32, 56, 100, and 220 each contain exactly 100,000 networks. Two Rademacher parameter probes per network give an unbiased trace-average NTK estimator; uncertainty includes network and probe variation.

The prediction is extracted before sampling from the paper's vector PDF coordinates and checked for constant `1/n` coefficients across all 11 source widths. The verifier compares fresh means to this prediction and to the infinite-width negative control. One hundred raw block means of 1,000 networks are emitted per width.
