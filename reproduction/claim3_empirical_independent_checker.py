"""Independent calibration for the five-million-sample NTK estimator."""

import jax
import jax.numpy as jnp
import numpy as np

from reproduction.claim3_paper_scale import (
    C_W,
    DEPTH,
    INPUTS,
    LEAK,
    forward,
    infinite_ntk,
    init_params,
)


def check_activation(params, alpha, seed):
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

    jacobian = np.asarray(
        jax.jacrev(lambda vector: forward(unflatten(vector), jnp.asarray(INPUTS), alpha).reshape(-1))(
            flat
        )
    ).reshape(2, 2, -1)
    exact = np.array(
        (
            np.mean(np.sum(jacobian[0] ** 2, axis=-1)),
            np.mean(np.sum(jacobian[0] * jacobian[1], axis=-1)),
        )
    )
    rng = np.random.default_rng(seed)
    probes = rng.choice((-1.0, 1.0), size=(8192, flat.size))
    directional = np.einsum("iop,qp->qio", jacobian, probes)
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
    return {
        "exact_trace_average": exact.tolist(),
        "hutchinson_trace_average": mean.tolist(),
        "hutchinson_standard_error": standard_error.tolist(),
        "max_z": float(np.max(z)),
        "passed": bool(np.max(z) <= 4.0),
    }


def check():
    width = 3
    params = init_params(jax.random.PRNGKey(3301), width)
    relu = check_activation(params, 0.0, 3302)
    leaky = check_activation(params, LEAK, 3303)
    input_variance = float(np.mean(INPUTS[0] ** 2))
    relu_closed_form = DEPTH * input_variance
    leaky_multiplier = C_W * (1.0 + LEAK**2) / 2.0
    leaky_closed_form = DEPTH * leaky_multiplier ** (DEPTH - 1) * input_variance
    recurrence_values = {
        "ReLU": float(infinite_ntk(0.0)[0, 0]),
        "LeakyReLU(alpha=0.1)": float(infinite_ntk(LEAK)[0, 0]),
    }
    checks = {
        "relu_hutchinson_matches_exact_jacobian": relu["passed"],
        "leakyrelu_hutchinson_matches_exact_jacobian": leaky["passed"],
        "relu_recurrence_matches_independent_closed_form": bool(
            abs(recurrence_values["ReLU"] - relu_closed_form) < 1e-12
        ),
        "leakyrelu_recurrence_matches_independent_closed_form": bool(
            abs(recurrence_values["LeakyReLU(alpha=0.1)"] - leaky_closed_form) < 1e-12
        ),
    }
    return {
        "tiny_width": width,
        "hutchinson_probes": 8192,
        "relu": relu,
        "leakyrelu": leaky,
        "independent_closed_form_diagonal": {
            "ReLU": relu_closed_form,
            "LeakyReLU(alpha=0.1)": leaky_closed_form,
        },
        "recurrence_diagonal": recurrence_values,
        "checks": checks,
        "passed": all(checks.values()),
    }
