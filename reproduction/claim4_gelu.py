"""Source-faithful GeLU finite-width NTK correction experiment."""

import time

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np


WIDTHS = (32, 56, 100, 220)
N_NETWORKS = 100_000
N_PROBES = 2
BATCH_SIZE = 64
DEPTH = 4
C_W = 1.98305826
SEED = 4250811522
INPUTS = np.array(
    [
        [0.6540248765053858, -1.3592788739383235, -0.3791179316445409, 0.34191025226280697],
        [-0.6786403121150224, -0.12161213502515096, 0.21716372787141658, -1.4709063616118982],
    ],
    dtype=np.float64,
)

SOURCE_WIDTHS = np.array((6, 11, 19, 32, 56, 83, 100, 150, 172, 200, 220))
SOURCE_X = np.array(
    (36.032046, 53.429326, 69.122143, 84.086155, 100.151210, 111.447688,
     116.796176, 128.433546, 132.363607, 136.693335, 139.428313)
)
SOURCE_DIAGONAL_Y = np.array(
    (122.018077, 80.378236, 59.336931, 47.585930, 40.223433, 37.030013,
     35.905459, 34.071692, 33.601495, 33.154808, 32.907955)
)
SOURCE_OFFDIAGONAL_Y = np.array(
    (30.408075, 72.047917, 93.089222, 104.840222, 112.202720, 115.396140,
     116.520693, 118.354461, 118.824658, 119.271345, 119.522116)
)


def source_predictions():
    diagonal_scale = 0.05 / (50.124993 - 30.694112)
    diagonal_infinite_y = 30.408075
    diagonal_infinite = 1.75 + (diagonal_infinite_y - 30.694112) * diagonal_scale
    diagonal_coefficients = (
        SOURCE_WIDTHS * (SOURCE_DIAGONAL_Y - diagonal_infinite_y) * diagonal_scale
    )

    offdiagonal_scale = 0.01 / (59.603376 - 42.668456)
    offdiagonal_infinite_y = 122.018077
    offdiagonal_infinite = 0.13 + (
        offdiagonal_infinite_y - 42.668456
    ) * offdiagonal_scale
    offdiagonal_coefficients = (
        SOURCE_WIDTHS
        * (SOURCE_OFFDIAGONAL_Y - offdiagonal_infinite_y)
        * offdiagonal_scale
    )
    coefficients = np.array(
        (np.median(diagonal_coefficients), np.median(offdiagonal_coefficients))
    )
    infinite = np.array((diagonal_infinite, offdiagonal_infinite))
    decoded_widths = 10 ** (
        1 + (SOURCE_X - 50.694348) / (116.796176 - 50.694348)
    )
    coefficient_spread = np.array(
        (
            np.max(np.abs(diagonal_coefficients - coefficients[0])),
            np.max(np.abs(offdiagonal_coefficients - coefficients[1])),
        )
    )
    return {
        "pdf_sha256": {
            "diagonal": "865845aaa9c0f203e9fc041c7d5841c1548bc5d511abe55a8c9f9eef782683d6",
            "offdiagonal": "6623a42cf5703a8374603e190beef9f6e267a9a1dd3f13a536ec94e7ba0fceb5",
        },
        "widths": SOURCE_WIDTHS.tolist(),
        "decoded_widths": decoded_widths.tolist(),
        "infinite_ntk": infinite.tolist(),
        "first_order_coefficients": coefficients.tolist(),
        "coefficient_values_from_all_11_vertices": np.stack(
            (diagonal_coefficients, offdiagonal_coefficients), axis=-1
        ).tolist(),
        "max_coefficient_digitization_spread": coefficient_spread.tolist(),
    }


def gelu_exact(value):
    return 0.5 * value * (1.0 + jax.lax.erf(value / jnp.sqrt(2.0)))


def init_params(key, width):
    keys = jax.random.split(key, DEPTH)
    shapes = [(width, INPUTS.shape[1])] + [(width, width)] * (DEPTH - 1)
    return tuple(
        jnp.sqrt(C_W) * jax.random.normal(layer_key, shape, dtype=jnp.float64)
        for layer_key, shape in zip(keys, shapes, strict=True)
    )


def forward(params, inputs):
    z = inputs
    for layer, weights in enumerate(params):
        activations = z if layer == 0 else gelu_exact(z)
        z = activations @ weights.T / jnp.sqrt(jnp.float64(activations.shape[-1]))
    return z


def tangent(key, params):
    keys = jax.random.split(key, len(params))
    return tuple(
        jax.random.rademacher(probe_key, weights.shape, dtype=jnp.float64)
        for probe_key, weights in zip(keys, params, strict=True)
    )


def one_network(key, width):
    parameter_key, probe_root = jax.random.split(key)
    params = init_params(parameter_key, width)
    probe_keys = jax.random.split(probe_root, N_PROBES)

    def one_probe(probe_key):
        _, directional = jax.jvp(
            lambda p: forward(p, jnp.asarray(INPUTS)),
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


def evaluate(means, standard_errors, predictions, infinite):
    signal = np.abs(predictions - infinite)
    residual = np.abs(means - predictions)
    infinite_residual = np.abs(means - infinite)
    ratios = residual / signal
    checks = {
        "all_points_closer_to_first_order_than_infinite_width": bool(
            np.all(residual < infinite_residual)
        ),
        "all_residuals_at_most_half_of_first_order_signal": bool(
            np.all(ratios <= 0.5)
        ),
        "median_residual_at_most_quarter_of_first_order_signal": bool(
            np.median(ratios) <= 0.25
        ),
        "all_ensemble_means_finite": bool(np.all(np.isfinite(means))),
        "all_standard_errors_positive": bool(np.all(standard_errors > 0)),
    }
    return {
        "absolute_residual": residual.tolist(),
        "absolute_residual_to_infinite_width": infinite_residual.tolist(),
        "residual_over_first_order_signal": ratios.tolist(),
        "median_residual_over_first_order_signal": float(np.median(ratios)),
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_gelu_correction():
    started = time.perf_counter()
    source = source_predictions()
    infinite = np.asarray(source["infinite_ntk"])
    coefficients = np.asarray(source["first_order_coefficients"])
    rows = []
    for width_index, width in enumerate(WIDTHS):
        width_started = time.perf_counter()
        seed = SEED + 100_000 * width_index
        keys = jax.random.split(jax.random.PRNGKey(seed), N_NETWORKS)
        batched = jax.jit(jax.vmap(lambda key: one_network(key, width)))
        batches = []
        for lower in range(0, N_NETWORKS, BATCH_SIZE):
            upper = min(lower + BATCH_SIZE, N_NETWORKS)
            batches.append(np.asarray(batched(keys[lower:upper])))
            if upper % 10_000 == 0:
                print(
                    f"CLAIM4_PROGRESS width={width} networks={upper}/{N_NETWORKS} "
                    f"seconds={time.perf_counter() - width_started:.1f}",
                    flush=True,
                )
        samples = np.concatenate(batches, axis=0)
        mean = samples.mean(axis=0)
        standard_deviation = samples.std(axis=0, ddof=1)
        standard_error = standard_deviation / np.sqrt(N_NETWORKS)
        block_means = samples.reshape(100, N_NETWORKS // 100, 2).mean(axis=1)
        prediction = infinite + coefficients / width
        rows.append(
            {
                "width": width,
                "seed": seed,
                "mean": mean.tolist(),
                "standard_deviation": standard_deviation.tolist(),
                "standard_error": standard_error.tolist(),
                "source_first_order_prediction": prediction.tolist(),
                "source_infinite_width_prediction": infinite.tolist(),
                "block_means_100_blocks_of_1000": block_means.tolist(),
                "runtime_seconds": time.perf_counter() - width_started,
            }
        )

    means = np.asarray([row["mean"] for row in rows])
    standard_errors = np.asarray([row["standard_error"] for row in rows])
    predictions = np.asarray([row["source_first_order_prediction"] for row in rows])
    infinite_grid = np.broadcast_to(infinite, means.shape)
    verification = evaluate(means, standard_errors, predictions, infinite_grid)
    infinite_control = evaluate(
        means,
        standard_errors,
        infinite_grid,
        predictions,
    )
    checks = {
        "paper_sample_count_each_width": N_NETWORKS == 100_000,
        "paper_four_layer_gelu_architecture": DEPTH == 4,
        "all_test_widths_above_20": all(width > 20 for width in WIDTHS),
        "source_curve_widths_decode_to_integers": bool(
            np.max(
                np.abs(
                    np.asarray(source["decoded_widths"])
                    - np.asarray(source["widths"])
                )
            )
            < 0.03
        ),
        "source_curve_is_constant_first_order_coefficient": bool(
            np.max(source["max_coefficient_digitization_spread"]) < 0.002
        ),
        "fresh_data_passes_first_order_contract": verification["passed"],
        "infinite_width_negative_control_fails": not infinite_control["passed"],
    }
    return {
        "claim": (
            "Fresh four-layer GeLU NTK means at widths above 20 agree more closely "
            "with the paper's first-order recursion curve than with infinite width"
        ),
        "architecture": {
            "activation": "exact GeLU",
            "bias": False,
            "depth": DEPTH,
            "weight_variance": C_W,
            "parameterization": "paper Appendix B raw Gaussian weights",
        },
        "inputs": INPUTS.tolist(),
        "columns": ["theta_00", "theta_01"],
        "network_initializations_per_width": N_NETWORKS,
        "hutchinson_probes_per_network": N_PROBES,
        "source_prediction": source,
        "rows": rows,
        "verification": verification,
        "negative_control_infinite_width_substitution": infinite_control,
        "checks": checks,
        "runtime_seconds": time.perf_counter() - started,
        "passed": all(checks.values()),
    }
