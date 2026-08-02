import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Finite-width NTKs, from diagrams to five million networks

        ![Headline evidence](https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-repro-SOlPHMdSY3-finite-width-neural-tangent-kernels-from-feynman-diagrams/main/reports/reproduction/images/claim3_exact_scale.svg)

        The infinite-width NTK is the leading term of a `1/n` expansion. This
        notebook explains the already-computed evidence; it does **not** rerun the
        expensive experiments.
        """
    )
    return


@app.cell
def _():
    diagonal = [0.0397, 0.0565, 0.0120, 0.0327]
    offdiagonal = [6.421, 8.299, 1.344, 1.926]
    labels = ["ReLU n=20", "Leaky n=20", "ReLU n=80", "Leaky n=80"]
    return diagonal, labels, offdiagonal


@app.cell
def _(diagonal, labels, mo, offdiagonal):
    mo.md(
        "## Why the headline matters\n\n"
        "The theorem concerns the ensemble-mean **diagonal**, not every kernel "
        "entry. The table keeps that distinction visible.\n\n"
        "| Cell | diagonal shift | off-diagonal control |\n|---|---:|---:|\n"
        + "\n".join(
            f"| {label} | {diag:.4f}% | {off:.3f}% |"
            for label, diag, off in zip(labels, diagonal, offdiagonal)
        )
        + "\n\nEvery diagonal 99% bound was below the precommitted 1% margin; "
        "every control exceeded 1% at more than 5 standard errors."
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The five-diagram recursion

        ![Five diagrams](https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-repro-SOlPHMdSY3-finite-width-neural-tangent-kernels-from-feynman-diagrams/main/reports/reproduction/images/claim2_five_diagrams.svg)

        The order-`1/n` NTK-mean correction has five admissible terms: quadratic
        `K1` and `Theta1`, plus quartic `V`, `D`, and `F`. The executable verifier
        checks topology, coefficients, and an independent algebraic sum; deleting
        one vertex makes its negative control fail.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Two source-scale empirical checks

        ![GeLU correction](https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-repro-SOlPHMdSY3-finite-width-neural-tangent-kernels-from-feynman-diagrams/main/reports/reproduction/images/claim4_gelu_correction.svg)

        Four fresh GeLU measurements, each averaging 100,000 networks, follow the
        source Figure 2 first-order correction rather than the infinite-width line.

        ![Depth stability](https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-repro-SOlPHMdSY3-finite-width-neural-tangent-kernels-from-feynman-diagrams/main/reports/reproduction/images/claim5_depth_stability.svg)

        At width 200 and `C_W=2`, the depth-1-to-30 slope is
        `0.679499 ± 0.012189` against `0.669128` predicted. Low/high controls decay
        and grow exponentially.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Reproduce or inspect

        Formal command:

        ```bash
        uv sync --frozen --all-extras && uv run --frozen python -m reproduction.run_all
        ```

        The command is expensive: it includes a five-million-network sweep and
        should run on the campaign's HF `cpu-upgrade` target. For the full evidence,
        read the [visual report](https://github.com/MachineLearning-Nerd/icml26-repro-SOlPHMdSY3-finite-width-neural-tangent-kernels-from-feynman-diagrams/blob/main/reports/reproduction/report.md)
        and its linked raw JSON, checkers, controls, assumptions, and limitations.
        """
    )
    return


if __name__ == "__main__":
    app.run()
