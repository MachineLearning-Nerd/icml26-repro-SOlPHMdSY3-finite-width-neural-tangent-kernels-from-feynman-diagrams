# Claim 4 source audit

The imported judge statement conflates two distinct v4 experiments. Section 6 Figure 2 and Appendix D use a four-layer **GeLU** MLP, `C_W=1.98305826`, widths up to 220, 100,000 NTK samples, and report close agreement above width 20 with the solved first-order recursion. Figure 3 separately uses a four-layer **ReLU** MLP and 5,000,000 samples to show absence of diagonal corrections.

The source-PDF vector curves are immutable pre-existing predictions, not fit to reproduction samples. SHA-256: diagonal `865845aaa9c0f203e9fc041c7d5841c1548bc5d511abe55a8c9f9eef782683d6`; off-diagonal `6623a42cf5703a8374603e190beef9f6e267a9a1dd3f13a536ec94e7ba0fceb5`. Their 11 points decode to widths 6, 11, 19, 32, 56, 83, 100, 150, 172, 200, 220 and recover constant `1/n` NTK coefficients near `+1.4144` (diagonal) and `-0.32457` (off-diagonal).
