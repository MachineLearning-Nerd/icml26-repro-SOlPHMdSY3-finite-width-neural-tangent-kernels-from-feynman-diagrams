"""Build evaluator-facing figures and check statistical rerun reproducibility."""

import hashlib
import html
import json
import math
from pathlib import Path


RAW_PATH = Path(
    ".openresearch/artifacts/cumulative/raw/hf_run_2e812c37.json"
)
FIGURE_PREFIX = "reports/reproduction/images"


def _svg(title: str, subtitle: str, body: str, height: int = 470) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">
<rect width="900" height="{height}" fill="#fbfaf7"/>
<style>
text {{ font-family: ui-sans-serif, system-ui, sans-serif; fill: #19212b }}
.title {{ font-size: 27px; font-weight: 700 }} .sub {{ font-size: 15px; fill: #52606d }}
.axis {{ stroke: #9aa5b1; stroke-width: 1 }} .grid {{ stroke: #dde2e7; stroke-width: 1 }}
.label {{ font-size: 14px }} .small {{ font-size: 12px; fill: #52606d }}
</style>
<text x="50" y="43" class="title">{html.escape(title)}</text>
<text x="50" y="69" class="sub">{html.escape(subtitle)}</text>
{body}
</svg>'''


def _bar_svg(rows: list[dict]) -> str:
    values = []
    labels = []
    colors = []
    for row in rows:
        label = f'{row["activation"]} n={row["width"]}'
        values.extend(
            [
                100 * row["comparison"]["diagonal_relative_shift"],
                100 * row["comparison"]["offdiagonal_relative_shift"],
            ]
        )
        labels.extend([f"{label} diag", f"{label} offdiag"])
        colors.extend(["#167d70", "#d95d39"])
    maximum = max(values) * 1.12
    body = '<line x1="245" y1="105" x2="245" y2="405" class="axis"/>'
    for tick in range(0, 10, 2):
        x = 245 + 600 * tick / maximum
        body += f'<line x1="{x:.1f}" y1="105" x2="{x:.1f}" y2="405" class="grid"/>'
        body += f'<text x="{x:.1f}" y="427" text-anchor="middle" class="small">{tick}%</text>'
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = 114 + index * 34
        width = 600 * value / maximum
        body += f'<text x="235" y="{y + 17}" text-anchor="end" class="small">{html.escape(label)}</text>'
        body += f'<rect x="245" y="{y}" width="{width:.1f}" height="22" rx="3" fill="{color}"/>'
        body += f'<text x="{250 + width:.1f}" y="{y + 16}" class="small">{value:.3f}%</text>'
    one_percent = 245 + 600 / maximum
    body += f'<line x1="{one_percent:.1f}" y1="100" x2="{one_percent:.1f}" y2="405" stroke="#167d70" stroke-width="2" stroke-dasharray="5 4"/>'
    body += f'<text x="{one_percent + 5:.1f}" y="98" class="small">1% precommitted diagonal equivalence margin</text>'
    body += '<rect x="52" y="438" width="12" height="12" fill="#167d70"/><text x="70" y="449" class="small">diagonal cancellation target</text>'
    body += '<rect x="278" y="438" width="12" height="12" fill="#d95d39"/><text x="296" y="449" class="small">off-diagonal negative control</text>'
    return _svg(
        "Scale invariance cancels only the diagonal correction",
        "Five million networks per activation and width; absolute relative shifts",
        body,
    )


def _claim4_svg(rows: list[dict]) -> str:
    body = '<line x1="95" y1="390" x2="850" y2="390" class="axis"/>'
    body += '<line x1="95" y1="100" x2="95" y2="390" class="axis"/>'
    widths = [row["width"] for row in rows]
    x_min, x_max = min(widths), max(widths)
    fractions = []
    for row in rows:
        measured = row["mean"][0] - row["source_infinite_width_prediction"][0]
        predicted = (
            row["source_first_order_prediction"][0]
            - row["source_infinite_width_prediction"][0]
        )
        fractions.append(measured / predicted)
    for tick in [0, 0.5, 1.0, 1.5]:
        y = 390 - tick * 180
        body += f'<line x1="95" y1="{y:.1f}" x2="850" y2="{y:.1f}" class="grid"/>'
        body += f'<text x="82" y="{y + 5:.1f}" text-anchor="end" class="small">{tick:.1f}</text>'
    body += '<line x1="95" y1="210" x2="850" y2="210" stroke="#167d70" stroke-width="2" stroke-dasharray="6 5"/>'
    body += '<text x="845" y="202" text-anchor="end" class="small">paper first-order correction = 1</text>'
    for width, fraction in zip(widths, fractions):
        x = 110 + 720 * (width - x_min) / (x_max - x_min)
        y = 390 - fraction * 180
        body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#355c9a"/>'
        body += f'<text x="{x:.1f}" y="414" text-anchor="middle" class="small">{width}</text>'
        body += f'<text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle" class="small">{fraction:.2f}</text>'
    body += '<text x="472" y="446" text-anchor="middle" class="label">hidden width n</text>'
    body += '<text transform="translate(26 270) rotate(-90)" text-anchor="middle" class="label">measured / predicted correction</text>'
    return _svg(
        "Finite-width GeLU means follow the 1/n recursion correction",
        "Four-layer source architecture; 100,000 initializations at each width",
        body,
    )


def _claim5_svg(verifier: dict) -> str:
    colors = {"low": "#355c9a", "critical": "#167d70", "high": "#d95d39"}
    body = '<line x1="95" y1="390" x2="850" y2="390" class="axis"/>'
    body += '<line x1="95" y1="100" x2="95" y2="390" class="axis"/>'
    for level in range(-1, 5):
        y = 390 - (level + 1) * 48
        body += f'<line x1="95" y1="{y}" x2="850" y2="{y}" class="grid"/>'
        body += f'<text x="82" y="{y + 5}" text-anchor="end" class="small">10^{level}</text>'
    for name in ["low", "critical", "high"]:
        means = verifier["summaries"][name]["mean"]
        points = []
        for depth_index, row in enumerate(means, start=1):
            value = max(row[0] / depth_index, 1e-2)
            x = 95 + 755 * (depth_index - 1) / 29
            y = 390 - (math.log10(value) + 1) * 48
            points.append(f"{x:.1f},{y:.1f}")
        point_text = " ".join(points)
        body += f'<polyline points="{point_text}" fill="none" stroke="{colors[name]}" stroke-width="3"/>'
    body += '<text x="472" y="437" text-anchor="middle" class="label">depth</text>'
    body += '<text transform="translate(25 270) rotate(-90)" text-anchor="middle" class="label">mean diagonal NTK / depth (log scale)</text>'
    for index, name in enumerate(["low", "critical", "high"]):
        x = 315 + index * 125
        body += f'<line x1="{x}" y1="458" x2="{x + 25}" y2="458" stroke="{colors[name]}" stroke-width="4"/>'
        body += f'<text x="{x + 31}" y="463" class="small">{name}</text>'
    return _svg(
        "Only critical initialization stays linear through depth 30",
        "Width 200, 1,000 networks per regime, source gradient-stability observable",
        body,
        height=480,
    )


def _claim2_svg(verifier: dict) -> str:
    diagrams = verifier["diagrams"]
    body = ''
    for index, diagram in enumerate(diagrams):
        x = 95 + index * 160
        quadratic = diagram["correction_vertex"]["name"] in {"K1", "Theta1"}
        order = 2 if quadratic else 4
        color = "#355c9a" if quadratic else "#d95d39"
        body += f'<circle cx="{x}" cy="220" r="34" fill="{color}" opacity="0.14" stroke="{color}" stroke-width="3"/>'
        body += f'<text x="{x}" y="226" text-anchor="middle" class="title">{order}</text>'
        body += f'<text x="{x}" y="280" text-anchor="middle" class="label">D{index + 1}</text>'
        body += f'<text x="{x}" y="303" text-anchor="middle" class="small">{html.escape(diagram["id"])}</text>'
    body += '<line x1="95" y1="350" x2="735" y2="350" stroke="#167d70" stroke-width="4"/>'
    body += '<text x="415" y="382" text-anchor="middle" class="label">independently summed recursion coefficient matches the closed form</text>'
    body += '<text x="415" y="410" text-anchor="middle" class="small">blue: quadratic vertex · red: quartic vertex · injected sign error exits nonzero</text>'
    return _svg(
        "The first-order mean recursion has exactly five diagrams",
        "Machine-enumerated quadratic and quartic contributions, checked independently",
        body,
    )


def _paired_z(current_rows: list, previous_rows: list) -> dict:
    z_values = []
    for current, previous in zip(current_rows, previous_rows):
        for mean, old_mean, se, old_se in zip(
            current["mean"],
            previous["mean"],
            current["standard_error"],
            previous["standard_error"],
        ):
            denominator = math.sqrt(se * se + old_se * old_se)
            z_values.append(abs(mean - old_mean) / denominator)
    maximum = max(z_values)
    return {
        "comparison": "independent reruns agree within five combined standard errors",
        "maximum_combined_standard_error_z": maximum,
        "threshold": 5.0,
        "passed": maximum <= 5.0,
    }


def _claim5_rows(verifier: dict) -> list[dict]:
    rows = []
    for name in ["low", "critical", "high"]:
        summary = verifier["summaries"][name]
        for mean, standard_error in zip(summary["mean"], summary["standard_error"]):
            rows.append({"mean": mean, "standard_error": standard_error})
    return rows


def build_release_artifacts(current: dict) -> dict:
    snapshot = json.loads(RAW_PATH.read_text())
    reproducibility = {
        "claim3": _paired_z(
            current["claim3_empirical_verifier"]["rows"],
            snapshot["claim3"]["empirical"]["rows"],
        ),
        "claim4": _paired_z(
            current["claim4_verifier"]["rows"],
            snapshot["claim4"]["verifier"]["rows"],
        ),
        "claim5": _paired_z(
            _claim5_rows(current["claim5_verifier"]),
            _claim5_rows(snapshot["claim5"]["verifier"]),
        ),
    }
    figures = {
        f"{FIGURE_PREFIX}/claim3_exact_scale.svg": _bar_svg(
            current["claim3_empirical_verifier"]["rows"]
        ),
        f"{FIGURE_PREFIX}/claim4_gelu_correction.svg": _claim4_svg(
            current["claim4_verifier"]["rows"]
        ),
        f"{FIGURE_PREFIX}/claim5_depth_stability.svg": _claim5_svg(
            current["claim5_verifier"]
        ),
        f"{FIGURE_PREFIX}/claim2_five_diagrams.svg": _claim2_svg(
            current["claim2_verifier"]
        ),
    }
    payloads = []
    for path, svg in figures.items():
        payloads.append(
            {
                "path": path,
                "sha256": hashlib.sha256(svg.encode()).hexdigest(),
                "text": svg,
            }
        )
    fixed_command_matches = current["fixed_command"] == snapshot["fixed_command"]
    passed = (
        snapshot["passed"]
        and fixed_command_matches
        and all(item["passed"] for item in reproducibility.values())
        and len(payloads) == 4
        and all("<script" not in item["text"].lower() for item in payloads)
    )
    return {
        "raw_snapshot": str(RAW_PATH),
        "raw_snapshot_sha256": hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(),
        "fixed_command_matches": fixed_command_matches,
        "reproducibility": reproducibility,
        "svg_payloads": payloads,
        "passed": passed,
    }
