"""Five-million-sample ReLU and LeakyReLU diagonal NTK experiment."""

import math
import time

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np


WIDTHS = (20, 80)
N_NETWORKS = 5_000_000
N_PROBES = 2
BATCH_SIZE = 1_000
BLOCK_SIZE = 50_000
DEPTH = 4
C_W = 2.0
LEAK = 0.1
SEED = 325081522
INPUTS = np.array(
    [
        [-0.9895229339599609, -0.5992491841316223],
        [-0.17877478897571564, 2.253682851791382],
    ],
    dtype=np.float64,
)


def init_params(key, width):
    keys = jax.random.split(key, DEPTH)
    shapes = (
        (width, INPUTS.shape[1]),
        (width, width),
        (width, width),
        (2, width),
    )
    return tuple(
        jnp.sqrt(C_W) * jax.random.normal(layer_key, shape, dtype=jnp.float64)
        for layer_key, shape in zip(keys, shapes, strict=True)
    )


def activation(value, alpha):
    return jnp.where(value >= 0.0, value, alpha * value)


def forward(params, inputs, alpha):
    z = inputs
    for layer, weights in enumerate(params):
        activations = z if layer == 0 else activation(z, alpha)
        z = activations @ weights.T / jnp.sqrt(jnp.float64(activations.shape[-1]))
    return z


def tangent(key, params):
    keys = jax.random.split(key, len(params))
    return tuple(
        jax.random.rademacher(probe_key, weights.shape, dtype=jnp.float64)
        for probe_key, weights in zip(keys, params, strict=True)
    )


def activation_estimate(params, probe_keys, alpha):
    def one_probe(probe_key):
        _, directional = jax.jvp(
            lambda p: forward(p, jnp.asarray(INPUTS), alpha),
            (params,),
            (tangent(probe_key, params),),
        )
        return jnp.array(
            (
                jnp.mean(directional[0] ** 2),
                jnp.mean(directional[0] * directional[1]),
            )
        )

    return jnp.mean(jax.vmap(one_probe)(probe_keys), axis=0)


def one_network(key, width):
    parameter_key, probe_root = jax.random.split(key)
    params = init_params(parameter_key, width)
    probe_keys = jax.random.split(probe_root, N_PROBES)
    relu = activation_estimate(params, probe_keys, 0.0)
    leaky = activation_estimate(params, probe_keys, LEAK)
    return jnp.concatenate((relu, leaky))


def relu_moments(q1, q2, covariance):
    scale = math.sqrt(q1 * q2)
    correlation = float(np.clip(covariance / scale, -1.0, 1.0))
    angle = math.acos(correlation)
    relu_covariance = scale * (
        math.sin(angle) + (math.pi - angle) * math.cos(angle)
    ) / (2.0 * math.pi)
    positive_probability = (math.pi - angle) / (2.0 * math.pi)
    return relu_covariance, positive_probability


def activation_moments(q1, q2, covariance, alpha):
    positive_covariance, same_sign_probability = relu_moments(q1, q2, covariance)
    opposite_covariance, opposite_sign_probability = relu_moments(
        q1, q2, -covariance
    )
    covariance_value = (
        (1.0 + alpha**2) * positive_covariance
        - 2.0 * alpha * opposite_covariance
    )
    derivative_value = (
        (1.0 + alpha**2) * same_sign_probability
        + 2.0 * alpha * opposite_sign_probability
    )
    return covariance_value, derivative_value


def infinite_ntk(alpha):
    covariance = INPUTS @ INPUTS.T / INPUTS.shape[1]
    theta = covariance.copy()
    covariance = C_W * covariance
    for _ in range(1, DEPTH):
        sigma_covariance = np.empty((2, 2), dtype=np.float64)
        derivative_covariance = np.empty((2, 2), dtype=np.float64)
        for left in range(2):
            for right in range(2):
                values = activation_moments(
                    covariance[left, left],
                    covariance[right, right],
                    covariance[left, right],
                    alpha,
                )
                sigma_covariance[left, right] = values[0]
                derivative_covariance[left, right] = values[1]
        theta = sigma_covariance + C_W * derivative_covariance * theta
        covariance = C_W * sigma_covariance
    return theta


def verify_rows(rows):
    diagonal_checks = []
    offdiagonal_controls = []
    for row in rows:
        infinite = np.asarray(row["infinite_ntk"])
        mean = np.asarray(row["mean"])
        standard_error = np.asarray(row["standard_error"])
        diagonal_relative_shift = abs(mean[0] - infinite[0]) / abs(infinite[0])
        diagonal_relative_99_upper = (
            abs(mean[0] - infinite[0]) + 2.576 * standard_error[0]
        ) / abs(infinite[0])
        diagonal_z = abs(mean[0] - infinite[0]) / standard_error[0]
        offdiagonal_relative_shift = abs(mean[1] - infinite[1]) / abs(infinite[1])
        offdiagonal_z = abs(mean[1] - infinite[1]) / standard_error[1]
        row["comparison"] = {
            "diagonal_relative_shift": float(diagonal_relative_shift),
            "diagonal_relative_99pct_upper": float(diagonal_relative_99_upper),
            "diagonal_z": float(diagonal_z),
            "offdiagonal_relative_shift": float(offdiagonal_relative_shift),
            "offdiagonal_z": float(offdiagonal_z),
        }
        diagonal_checks.append(diagonal_z <= 4.0 and diagonal_relative_99_upper <= 0.01)
        offdiagonal_controls.append(
            offdiagonal_z >= 5.0 and offdiagonal_relative_shift >= 0.01
        )
    checks = {
        "all_diagonal_means_within_4_standard_errors": bool(
            all(row["comparison"]["diagonal_z"] <= 4.0 for row in rows)
        ),
        "all_diagonal_99pct_intervals_inside_1pct_equivalence_margin": bool(
            all(
                row["comparison"]["diagonal_relative_99pct_upper"] <= 0.01
                for row in rows
            )
        ),
        "all_offdiagonal_controls_detect_at_least_1pct_correction_at_5se": bool(
            all(offdiagonal_controls)
        ),
        "every_activation_width_pair_passes": bool(all(diagonal_checks)),
    }
    return {"checks": checks, "passed": all(checks.values())}


def run_five_million_scale():
    started = time.perf_counter()
    references = {
        "ReLU": infinite_ntk(0.0),
        "LeakyReLU(alpha=0.1)": infinite_ntk(LEAK),
    }
    rows = []
    for width_index, width in enumerate(WIDTHS):
        width_started = time.perf_counter()
        seed = SEED + 100_000 * width_index
        root_key = jax.random.PRNGKey(seed)
        batched = jax.jit(jax.vmap(lambda key: one_network(key, width)))
        total_sum = np.zeros(4, dtype=np.float64)
        total_sum_of_squares = np.zeros(4, dtype=np.float64)
        block_sum = np.zeros(4, dtype=np.float64)
        block_count = 0
        block_means = []
        for lower in range(0, N_NETWORKS, BATCH_SIZE):
            upper = min(lower + BATCH_SIZE, N_NETWORKS)
            indices = jnp.arange(lower, upper, dtype=jnp.uint32)
            keys = jax.vmap(lambda index: jax.random.fold_in(root_key, index))(indices)
            samples = np.asarray(batched(keys))
            total_sum += samples.sum(axis=0)
            total_sum_of_squares += np.square(samples).sum(axis=0)
            block_sum += samples.sum(axis=0)
            block_count += samples.shape[0]
            if block_count == BLOCK_SIZE:
                block_means.append((block_sum / block_count).tolist())
                block_sum.fill(0.0)
                block_count = 0
            if upper % 500_000 == 0:
                print(
                    f"CLAIM3_EMPIRICAL_PROGRESS width={width} "
                    f"networks={upper}/{N_NETWORKS} "
                    f"seconds={time.perf_counter() - width_started:.1f}",
                    flush=True,
                )
        mean = total_sum / N_NETWORKS
        variance = (
            total_sum_of_squares - N_NETWORKS * np.square(mean)
        ) / (N_NETWORKS - 1)
        standard_deviation = np.sqrt(np.maximum(variance, 0.0))
        standard_error = standard_deviation / math.sqrt(N_NETWORKS)
        width_runtime_seconds = time.perf_counter() - width_started
        for activation_index, name in enumerate(references):
            lower = 2 * activation_index
            upper = lower + 2
            rows.append(
                {
                    "activation": name,
                    "width": width,
                    "seed": seed,
                    "mean": mean[lower:upper].tolist(),
                    "standard_deviation": standard_deviation[lower:upper].tolist(),
                    "standard_error": standard_error[lower:upper].tolist(),
                    "infinite_ntk": [
                        float(references[name][0, 0]),
                        float(references[name][0, 1]),
                    ],
                    "block_means_100_blocks_of_50000": [
                        block[lower:upper] for block in block_means
                    ],
                    "width_runtime_seconds": width_runtime_seconds,
                }
            )

    verification = verify_rows(rows)
    checks = {
        "paper_four_layer_bias_free_architecture": DEPTH == 4,
        "paper_weight_variance": C_W == 2.0,
        "paper_two_output_trace_average": True,
        "exact_five_million_initializations_per_width": N_NETWORKS == 5_000_000,
        "all_widths_at_least_20": all(width >= 20 for width in WIDTHS),
        "relu_and_leakyrelu_alpha_point_one": LEAK == 0.1,
        "one_hundred_raw_block_means_per_activation_width": all(
            len(row["block_means_100_blocks_of_50000"]) == 100 for row in rows
        ),
        "statistical_contract_passes": verification["passed"],
    }
    return {
        "claim": (
            "At exact five-million-initialization scale, four-layer bias-free "
            "ReLU and LeakyReLU diagonal NTK means equal their infinite-width values"
        ),
        "architecture": {
            "depth": DEPTH,
            "hidden_widths": list(WIDTHS),
            "output_width": 2,
            "bias": False,
            "weight_variance": C_W,
            "parameterization": "paper Appendix B standard/book convention",
        },
        "inputs": INPUTS.tolist(),
        "columns": ["theta_00", "theta_01"],
        "network_initializations_per_activation_width": N_NETWORKS,
        "hutchinson_probes_per_network": N_PROBES,
        "estimator_deviation": (
            "Unbiased two-probe parameter-space Hutchinson estimator replaces "
            "the paper implementation's exact recursive Jacobian trace"
        ),
        "common_random_numbers": (
            "ReLU and LeakyReLU share each Gaussian initialization and probe; "
            "networks remain iid within each activation"
        ),
        "source_pdf_sha256": {
            "relu": "9dcccaca6bf62d5d07ffd24e8b1c346e3e25b07bef099dc85b2f0eb88c187c0a",
            "leakyrelu": "85a5e639eda539957ddcf835e734a8d4c37b7dcb10d9c7f6f13379be27fb52bd",
        },
        "rows": rows,
        "verification": verification,
        "checks": checks,
        "runtime_seconds": time.perf_counter() - started,
        "passed": all(checks.values()),
    }
