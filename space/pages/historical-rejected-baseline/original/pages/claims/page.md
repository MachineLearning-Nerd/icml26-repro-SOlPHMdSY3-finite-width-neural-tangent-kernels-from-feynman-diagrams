# Claims


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ea03fb5f4399", "created_at": "2026-07-31T11:08:23+00:00", "title": "Claims to reproduce"}
-->
## Claims to reproduce

1. Feynman-diagram graphical rules are introduced that simplify layer-wise recursion relations for finite-width Neural Tangent Kernel (NTK) statistics at order 1/n (Section 4).
2. A recursion relation for the first-order (1/n) finite-width correction to the NTK mean is derived using five Feynman diagrams containing quadratic and quartic vertices (Section 5.1).
3. For scale-invariant activation functions (ReLU and LeakyReLU), the infinite-width NTK diagonal Theta(x,x) has no finite-width corrections (Section 5.3).
4. Experiments on four-layer ReLU MLPs sampled over 5x10^6 initializations, with hidden-layer widths n>=20, confirm the finite-width kernel corrections predicted by the recursion relations (Figure 2, Section 6).
5. Gradient-stability experiments on 200-hidden-unit ReLU MLPs with up to 30 layers, averaged over 1000 initializations, show linear scaling with depth at the critical initialization C_W=2 (Figure 1, Section 6).
