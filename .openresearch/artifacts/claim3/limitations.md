# Limitations and deviations

The proof is for the exact paper assumptions and finite depth. It says nothing about biases, non-Gaussian initialization, convolutional layers, off-diagonal NTKs, individual finite samples, or infinite-depth/width joint limits. The Monte Carlo experiment with 5 million initializations is a separate empirical contract and is not replaced by quadrature.
