"""Paper-scale CPU reproduction of the critical-depth NTK experiment."""

import json
import time

import jax

jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np


WIDTH = 200
DEPTH = 30
N_NETWORKS = 1000
N_PROBES = 4
BATCH_SIZE = 4
SEED = 250811522
INPUTS = np.array(
    [
        [-0.9895229339599609, -0.5992491841316223],
        [-0.17877478897571564, 2.253682851791382],
    ],
    dtype=np.float32,
)
REGIMES = {"low": 0.25, "critical": 2.0, "high": 4.0}


def init_params(key, width: int, depth: int, c_w: float):
    keys = jax.random.split(key, depth)
    shapes = [(width, INPUTS.shape[1])] + [(width, width)] * (depth - 1)
    return tuple(
        jnp.sqrt(jnp.float32(c_w))
        * jax.random.normal(layer_key, shape, dtype=jnp.float32)
        for layer_key, shape in zip(keys, shapes, strict=True)
    )


def forward_all_layers(params, inputs):
    z = inputs
    outputs = []
    for layer, weights in enumerate(params):
        activations = z if layer == 0 else jax.nn.relu(z)
        z = activations @ weights.T / jnp.sqrt(jnp.float32(activations.shape[-1]))
        outputs.append(z)
    return jnp.stack(outputs)


def rademacher_tangent(key, params):
    keys = jax.random.split(key, len(params))
    return tuple(
        jax.random.rademacher(tangent_key, weights.shape, dtype=jnp.float32)
        for tangent_key, weights in zip(keys, params, strict=True)
    )


def one_network(key, c_w: float, width: int, depth: int, probes: int):
    parameter_key, probe_root = jax.random.split(key)
    params = init_params(parameter_key, width, depth, c_w)
    probe_keys = jax.random.split(probe_root, probes)

    def one_probe(probe_key):
        tangent = rademacher_tangent(probe_key, params)
        _, directional = jax.jvp(
            lambda p: forward_all_layers(p, jnp.asarray(INPUTS)),
            (params,),
            (tangent,),
        )
        theta_00 = jnp.mean(directional[:, 0] ** 2, axis=-1)
        theta_01 = jnp.mean(directional[:, 0] * directional[:, 1], axis=-1)
        theta_11 = jnp.mean(directional[:, 1] ** 2, axis=-1)
        return jnp.stack((theta_00, theta_01, theta_11), axis=-1)

    return jnp.mean(jax.vmap(one_probe)(probe_keys), axis=0)


def regression(x, y):
    design = np.column_stack((np.ones_like(x), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = intercept + slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot else 1.0,
    }


def summarize(samples):
    n = samples.shape[0]
    mean = samples.mean(axis=0)
    standard_deviation = samples.std(axis=0, ddof=1)
    standard_error = standard_deviation / np.sqrt(n)
    layers = np.arange(1, samples.shape[1] + 1, dtype=np.float64)
    slopes = np.einsum("l,nlc->nc", layers, samples) / np.dot(layers, layers)
    return {
        "mean": mean.tolist(),
        "standard_deviation": standard_deviation.tolist(),
        "standard_error": standard_error.tolist(),
        "per_network_through_origin_slopes": slopes.tolist(),
        "through_origin_slope_mean": slopes.mean(axis=0).tolist(),
        "through_origin_slope_standard_error": (
            slopes.std(axis=0, ddof=1) / np.sqrt(n)
        ).tolist(),
    }


def critical_checks(summary, low_summary, high_summary):
    layers = np.arange(1, DEPTH + 1, dtype=np.float64)
    observed = np.asarray(summary["mean"], dtype=np.float64)
    standard_error = np.asarray(summary["standard_error"], dtype=np.float64)
    expected_slope = float(np.dot(INPUTS[0], INPUTS[0]) / INPUTS.shape[1])
    expected = expected_slope * layers
    residual = observed[:, 0] - expected
    safe_se = np.maximum(standard_error[:, 0], np.finfo(np.float64).eps)
    slope_through_origin = float(summary["through_origin_slope_mean"][0])
    slope_se = float(summary["through_origin_slope_standard_error"][0])
    slope_z = abs(slope_through_origin - expected_slope) / max(slope_se, np.finfo(float).eps)
    critical_diag_regression = regression(layers, observed[:, 0])
    critical_offdiag_regression = regression(layers[9:], observed[9:, 1])

    high = np.maximum(np.asarray(high_summary["mean"], dtype=np.float64)[:, 0], 1e-30)
    high_log_regression = regression(layers[9:], np.log(high[9:]))
    low = np.maximum(np.asarray(low_summary["mean"], dtype=np.float64)[:, 0], 1e-30)
    low_log_regression = regression(layers[9:], np.log(low[9:]))
    high_expected_residual = high - expected
    high_se = np.maximum(
        np.asarray(high_summary["standard_error"], dtype=np.float64)[:, 0],
        np.finfo(np.float64).eps,
    )
    high_falsely_critical = bool(
        abs(float(np.dot(layers, high, ) / np.dot(layers, layers)) - expected_slope)
        / max(float(np.sqrt(np.sum((layers * high_se) ** 2)) / np.dot(layers, layers)), np.finfo(float).eps)
        <= 3.0
        and np.max(np.abs(high_expected_residual) / np.maximum(expected, 1e-30)) <= 0.10
    )

    checks = {
        "exact_paper_scale": WIDTH == 200 and DEPTH == 30 and N_NETWORKS == 1000,
        "critical_slope_within_3_standard_errors": slope_z <= 3.0,
        "critical_max_relative_deviation_at_most_10_percent": (
            float(np.max(np.abs(residual) / expected)) <= 0.10
        ),
        "critical_at_least_27_of_30_points_within_99pct_pointwise_ci": (
            int(np.sum(np.abs(residual) <= 2.576 * safe_se)) >= 27
        ),
        "critical_diagonal_linear_r_squared_at_least_0_995": (
            critical_diag_regression["r_squared"] >= 0.995
        ),
        "critical_offdiagonal_asymptotic_linear_r_squared_at_least_0_98": (
            critical_offdiag_regression["r_squared"] >= 0.98
        ),
        "high_variance_control_has_positive_exponential_log_slope": (
            high_log_regression["slope"] >= 0.20
        ),
        "low_variance_control_has_negative_exponential_log_slope": (
            low_log_regression["slope"] <= -0.20
        ),
        "high_variance_curve_rejected_by_critical_contract": not high_falsely_critical,
    }
    return {
        "expected_critical_diagonal_slope": expected_slope,
        "observed_critical_diagonal_slope_through_origin": slope_through_origin,
        "observed_slope_standard_error": slope_se,
        "observed_slope_z": float(slope_z),
        "critical_max_relative_deviation": float(np.max(np.abs(residual) / expected)),
        "critical_points_within_99pct_pointwise_ci": int(
            np.sum(np.abs(residual) <= 2.576 * safe_se)
        ),
        "critical_diagonal_regression": critical_diag_regression,
        "critical_offdiagonal_regression_depths_10_to_30": critical_offdiag_regression,
        "high_diagonal_log_regression_depths_10_to_30": high_log_regression,
        "low_diagonal_log_regression_depths_10_to_30": low_log_regression,
        "negative_control_high_falsely_accepted_as_critical": high_falsely_critical,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_paper_scale():
    started = time.perf_counter()
    regime_summaries = {}
    regime_seeds = {}
    for regime_index, (name, c_w) in enumerate(REGIMES.items()):
        seed = SEED + 100_000 * regime_index
        regime_seeds[name] = seed
        keys = jax.random.split(jax.random.PRNGKey(seed), N_NETWORKS)
        batched = jax.jit(
            jax.vmap(
                lambda network_key: one_network(
                    network_key, c_w, WIDTH, DEPTH, N_PROBES
                )
            )
        )
        batches = []
        regime_started = time.perf_counter()
        for lower in range(0, N_NETWORKS, BATCH_SIZE):
            upper = min(lower + BATCH_SIZE, N_NETWORKS)
            batches.append(np.asarray(batched(keys[lower:upper])))
            if upper % 100 == 0:
                print(
                    f"CLAIM5_PROGRESS regime={name} networks={upper}/{N_NETWORKS} "
                    f"seconds={time.perf_counter() - regime_started:.1f}",
                    flush=True,
                )
        samples = np.concatenate(batches, axis=0).astype(np.float64)
        regime_summaries[name] = summarize(samples)
        regime_summaries[name]["runtime_seconds"] = time.perf_counter() - regime_started

    verification = critical_checks(
        regime_summaries["critical"],
        regime_summaries["low"],
        regime_summaries["high"],
    )
    return {
        "claim": (
            "At C_W=2, a bias-free width-200 ReLU MLP has linearly scaling "
            "mean NTK through depth 30; away from C_W=2 it is exponentially unstable"
        ),
        "estimator": (
            "Appendix C output-channel trace average with unbiased Rademacher "
            "Hutchinson parameter-space probes"
        ),
        "inputs": INPUTS.tolist(),
        "width": WIDTH,
        "depth": DEPTH,
        "network_initializations_per_regime": N_NETWORKS,
        "hutchinson_probes_per_network": N_PROBES,
        "regimes": REGIMES,
        "seeds": regime_seeds,
        "columns": ["theta_00", "theta_01", "theta_11"],
        "layers": list(range(1, DEPTH + 1)),
        "summaries": regime_summaries,
        "verification": verification,
        "runtime_seconds": time.perf_counter() - started,
        "passed": verification["passed"],
    }


def main():
    result = run_paper_scale()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
