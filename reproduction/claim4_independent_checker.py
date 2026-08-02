"""Independent exact-Jacobian check for the Claim 4 GeLU estimator."""

import jax
import jax.numpy as jnp
import numpy as np

from reproduction.claim4_gelu import forward, init_params


def check():
    width = 3
    params = init_params(jax.random.PRNGKey(4404), width)
    leaves, tree = jax.tree_util.tree_flatten(params)
    shapes = [leaf.shape for leaf in leaves]
    sizes = [leaf.size for leaf in leaves]
    flat = jnp.concatenate([leaf.ravel() for leaf in leaves])

    def unflatten(vector):
        values = []
        offset = 0
        for shape, size in zip(shapes, sizes, strict=True):
            values.append(vector[offset : offset + size].reshape(shape))
            offset += size
        return jax.tree_util.tree_unflatten(tree, values)

    tiny_inputs = jnp.array([[0.3, -0.8, 0.2, 1.1], [-0.7, 0.4, 0.9, -0.1]])
    jacobian = np.asarray(
        jax.jacrev(lambda value: forward(unflatten(value), tiny_inputs).reshape(-1))(
            flat
        )
    ).reshape(2, width, -1)
    exact = np.array(
        (
            np.mean(np.sum(jacobian[0] ** 2, axis=-1)),
            np.mean(np.sum(jacobian[0] * jacobian[1], axis=-1)),
        )
    )
    rng = np.random.default_rng(4405)
    probes = rng.choice((-1.0, 1.0), size=(8192, flat.size))
    directional = np.einsum("iwp,qp->qiw", jacobian, probes)
    estimates = np.stack(
        (
            np.mean(directional[:, 0] ** 2, axis=-1),
            np.mean(directional[:, 0] * directional[:, 1], axis=-1),
        ),
        axis=-1,
    )
    mean = estimates.mean(axis=0)
    standard_error = estimates.std(axis=0, ddof=1) / np.sqrt(estimates.shape[0])
    z = np.abs(mean - exact) / standard_error
    checks = {
        "hutchinson_matches_exact_jacobian_within_4_standard_errors": bool(
            np.max(z) <= 4.0
        ),
        "exact_diagonal_positive": bool(exact[0] > 0),
    }
    return {
        "tiny_width": width,
        "parameter_count": int(flat.size),
        "probes": probes.shape[0],
        "exact_trace_average": exact.tolist(),
        "hutchinson_trace_average": mean.tolist(),
        "hutchinson_standard_error": standard_error.tolist(),
        "max_z": float(np.max(z)),
        "checks": checks,
        "passed": all(checks.values()),
    }
