# Method

`reproduction.diagram_rules` encodes the primitive outer-vertex insertion slots and correction-vertex signatures. It enumerates compatible diagrams without storing the expected five IDs in the enumerator. The verifier checks completeness, uniqueness, vertex counts, and coefficients. A separately implemented checker audits the serialized certificate without importing the rule module. The mutation control removes `F` and must exit nonzero.

This baseline checks the finite order-`1/n` five-diagram construction. It does not yet certify every theorem in Section 4 or numerically evaluate the Gaussian expectations.
