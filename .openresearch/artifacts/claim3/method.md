# Method

Positive homogeneity on the real line reduces every admissible activation to slopes `a_plus` and `a_minus` on the two half-lines. The verifier computes the exact Gaussian expectations `E[sigma(Z)^2]=A K` and `E[sigma'(Z)^2]=A`, proves all non-propagating diagram sources vanish under differentiation with respect to `K`, and emits the induction schema for every correction order. Independent Gauss-Hermite quadrature checks ReLU, LeakyReLU (`alpha=0.1`) and identity at three variances. GeLU is the intended negative control because its derivative-squared expectation varies with `K`.
