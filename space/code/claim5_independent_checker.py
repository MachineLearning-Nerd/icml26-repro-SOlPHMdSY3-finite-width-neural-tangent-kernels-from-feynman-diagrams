"""Independent finite-difference and exact-Jacobian check of Claim 5 estimator."""

import jax
import jax.numpy as jnp
import numpy as np

from reproduction.claim5_stability import forward_all_layers, init_params


def numpy_forward(params, inputs):
    z = inputs
    outputs = []
    for layer, weights in enumerate(params):
        activations = z if layer == 0 else np.maximum(z, 0.0)
        z = activations @ weights.T / np.sqrt(activations.shape[-1])
        outputs.append(z.copy())
    return np.stack(outputs)


def flatten(params):
    shapes = [value.shape for value in params]
    flat = np.concatenate([np.asarray(value).ravel() for value in params])

    def unflatten(vector):
        result = []
        offset = 0
        for shape in shapes:
            size = int(np.prod(shape))
            result.append(vector[offset : offset + size].reshape(shape))
            offset += size
        return tuple(result)

    return flat, unflatten


def check():
    width = 3
    depth = 3
    inputs = np.array([[0.7, -1.1], [-0.4, 0.9]], dtype=np.float64)
    params = tuple(
        np.asarray(value, dtype=np.float64)
        for value in init_params(jax.random.PRNGKey(451), width, depth, 2.0)
    )
    flat, unflatten = flatten(params)

    def jax_from_flat(vector):
        jax_params = tuple(jnp.asarray(value) for value in unflatten(vector))
        return forward_all_layers(jax_params, jnp.asarray(inputs)).reshape(-1)

    jacobian = np.asarray(jax.jacrev(jax_from_flat)(jnp.asarray(flat)))
    step = 1e-4
    finite_difference = np.empty_like(jacobian)
    for parameter in range(flat.size):
        direction = np.zeros_like(flat)
        direction[parameter] = step
        plus = numpy_forward(unflatten(flat + direction), inputs).reshape(-1)
        minus = numpy_forward(unflatten(flat - direction), inputs).reshape(-1)
        finite_difference[:, parameter] = (plus - minus) / (2.0 * step)

    exact = jacobian.reshape(depth, 2, width, -1)
    exact_theta = np.stack(
        (
            np.mean(np.sum(exact[:, 0] * exact[:, 0], axis=-1), axis=-1),
            np.mean(np.sum(exact[:, 0] * exact[:, 1], axis=-1), axis=-1),
            np.mean(np.sum(exact[:, 1] * exact[:, 1], axis=-1), axis=-1),
        ),
        axis=-1,
    )

    rng = np.random.default_rng(912)
    probes = rng.choice((-1.0, 1.0), size=(8192, flat.size))
    directional = np.einsum("diwp,qp->qdiw", exact, probes)
    estimates = np.stack(
        (
            np.mean(directional[:, :, 0] ** 2, axis=-1),
            np.mean(directional[:, :, 0] * directional[:, :, 1], axis=-1),
            np.mean(directional[:, :, 1] ** 2, axis=-1),
        ),
        axis=-1,
    )
    estimate_mean = estimates.mean(axis=0)
    estimate_se = estimates.std(axis=0, ddof=1) / np.sqrt(estimates.shape[0])
    z = np.abs(estimate_mean - exact_theta) / np.maximum(estimate_se, 1e-12)

    checks = {
        "jax_jacobian_matches_independent_finite_difference": (
            float(np.max(np.abs(jacobian - finite_difference))) < 2e-3
        ),
        "hutchinson_mean_matches_exact_trace_within_4_standard_errors": (
            float(np.max(z)) <= 4.0
        ),
    }
    return {
        "tiny_network": {"width": width, "depth": depth, "parameters": flat.size},
        "finite_difference_step": step,
        "max_jacobian_absolute_error": float(
            np.max(np.abs(jacobian - finite_difference))
        ),
        "hutchinson_probes": probes.shape[0],
        "exact_trace_average": exact_theta.tolist(),
        "hutchinson_trace_average": estimate_mean.tolist(),
        "hutchinson_standard_error": estimate_se.tolist(),
        "max_hutchinson_z": float(np.max(z)),
        "checks": checks,
        "passed": all(checks.values()),
    }
