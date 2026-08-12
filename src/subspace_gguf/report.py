"""Generate neutral JSON and Markdown reports from activation analysis."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .policies import DEFAULT_POLICIES, assign_modules
from .util import atomic_json


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _threshold_count(sorted_energy: list[float], fraction: float) -> int:
    target = fraction * sum(sorted_energy)
    cumulative = 0.0
    for index, value in enumerate(sorted_energy, start=1):
        cumulative += value
        if cumulative >= target:
            return index
    return len(sorted_energy)


def _parameters(module: dict) -> int:
    if "parameters" in module:
        return int(module["parameters"])
    return math.prod(int(value) for value in module["shape"])


def projection_summary(modules: dict) -> list[dict]:
    grouped = defaultdict(list)
    total_energy = sum(float(module["energy_total"]) for module in modules.values())
    for module in modules.values():
        grouped[module["projection"]].append(module)
    result = []
    for projection, values in sorted(grouped.items()):
        ranks = sum(int(value["atlas_rank"]) for value in values)
        result.append(
            {
                "projection": projection,
                "modules": len(values),
                "aggregate_energy_fraction": sum(
                    float(value["energy_total"]) for value in values
                )
                / total_energy,
                "per_module_top90_direction_fraction": sum(
                    int(value["energy_top90_count"]) for value in values
                )
                / ranks,
                "per_module_top95_direction_fraction": sum(
                    int(value["energy_top95_count"]) for value in values
                )
                / ranks,
            }
        )
    return result


def layer_summary(modules: dict) -> list[dict]:
    layers = sorted({module["layer"] for module in modules.values() if module["layer"] is not None})
    if not layers:
        return []
    chunk_size = max(1, math.ceil(len(layers) / 4))
    total_energy = sum(float(module["energy_total"]) for module in modules.values())
    result = []
    for start in range(0, len(layers), chunk_size):
        selected_layers = set(layers[start : start + chunk_size])
        values = [module for module in modules.values() if module["layer"] in selected_layers]
        ranks = sum(int(value["atlas_rank"]) for value in values)
        result.append(
            {
                "layers": f"{min(selected_layers)}-{max(selected_layers)}",
                "modules": len(values),
                "aggregate_energy_fraction": sum(
                    float(value["energy_total"]) for value in values
                )
                / total_energy,
                "per_module_top95_direction_fraction": sum(
                    int(value["energy_top95_count"]) for value in values
                )
                / ranks,
            }
        )
    return result


def global_threshold_summary(modules: dict, thresholds: list[float]) -> list[dict]:
    flat = []
    for name, module in modules.items():
        flat.extend((float(energy), name) for energy in module["direction_energy"])
    flat.sort(key=lambda item: item[0], reverse=True)
    sorted_energy = [item[0] for item in flat]
    total = sum(sorted_energy)
    result = []
    for threshold in thresholds:
        count = _threshold_count(sorted_energy, threshold)
        counts = Counter(name for _, name in flat[:count])
        selected_energy = Counter()
        for energy, name in flat[:count]:
            selected_energy[name] += energy
        rank_fractions = [
            counts[name] / int(module["atlas_rank"]) for name, module in modules.items()
        ]
        module_energy = [
            selected_energy[name] / float(module["energy_total"])
            if float(module["energy_total"]) else 0.0
            for name, module in modules.items()
        ]
        result.append(
            {
                "threshold": threshold,
                "directions": count,
                "direction_fraction": count / len(flat),
                "captured_energy_fraction": sum(sorted_energy[:count]) / total,
                "zero_direction_modules": sum(counts[name] == 0 for name in modules),
                "modules_below_1pct_directions": sum(value < 0.01 for value in rank_fractions),
                "modules_below_10pct_directions": sum(value < 0.10 for value in rank_fractions),
                "median_module_energy_retained": statistics.median(module_energy),
                "p10_module_energy_retained": _percentile(module_energy, 0.10),
            }
        )
    return result


def policy_summary(modules: dict) -> list[dict]:
    analyzed_parameters = sum(_parameters(module) for module in modules.values())
    result = []
    for policy in DEFAULT_POLICIES:
        assignments = assign_modules(modules, policy)
        energy = Counter()
        parameters = Counter()
        tensors = Counter()
        for assignment in assignments:
            precision = assignment["precision"]
            energy[precision] += assignment["energy_fraction"]
            parameters[precision] += assignment["parameters"]
            tensors[precision] += 1
        result.append(
            {
                "name": policy.name,
                "tiers": list(policy.tiers),
                "nominal_endpoints": list(policy.endpoints),
                "energy_percent_by_precision": {
                    precision: 100.0 * energy[precision] for precision in policy.tiers
                },
                "analyzed_weight_percent_by_precision": {
                    precision: 100.0 * parameters[precision] / analyzed_parameters
                    for precision in policy.tiers
                },
                "tensor_count_by_precision": {
                    precision: tensors[precision] for precision in policy.tiers
                },
            }
        )
    return result


def stability_summary(modules: dict, samples: int) -> dict:
    frequencies = []
    for module in modules.values():
        frequencies.extend(int(value) for value in module["energy_top95_sample_frequency"])
    if not frequencies:
        return {
            "directions": 0,
            "selected_in_all_samples_fraction": 0.0,
            "selected_in_at_least_90pct_samples_fraction": 0.0,
            "selected_in_at_least_50pct_samples_fraction": 0.0,
        }
    return {
        "directions": len(frequencies),
        "selected_in_all_samples_fraction": sum(value == samples for value in frequencies)
        / len(frequencies),
        "selected_in_at_least_90pct_samples_fraction": sum(
            value >= math.ceil(0.9 * samples) for value in frequencies
        )
        / len(frequencies),
        "selected_in_at_least_50pct_samples_fraction": sum(
            value >= math.ceil(0.5 * samples) for value in frequencies
        )
        / len(frequencies),
    }


def build_report(analysis: dict) -> dict:
    modules = analysis["modules"]
    thresholds = [float(value) for value in analysis["metadata"]["global_energy_thresholds"]]
    return {
        "source_analysis": analysis["metadata"],
        "analyzed_modules": len(modules),
        "analyzed_directions": sum(int(module["atlas_rank"]) for module in modules.values()),
        "full_matrix_rank_bound": int(analysis["coverage"]["full_matrix_rank_bound"]),
        "analyzed_parameters": sum(_parameters(module) for module in modules.values()),
        "projection_summary": projection_summary(modules),
        "layer_summary": layer_summary(modules),
        "global_thresholds": global_threshold_summary(modules, thresholds),
        "cross_sample_stability": stability_summary(
            modules, int(analysis["metadata"]["samples"])
        ),
        "quantization_policies": policy_summary(modules),
        "top_modules": analysis["coverage"]["top20_modules_by_raw_energy"],
    }


def markdown(report: dict) -> str:
    source = report["source_analysis"]
    lines = [
        "# Detailed activation-subspace effectiveness report",
        "",
        f"- Model: `{source['model']}`",
        f"- Dataset: `{source['dataset']}` ({source.get('dataset_split', 'unspecified split')})",
        f"- Calibration: {source['samples']} samples, seed {source['sample_seed']}",
        f"- Metric: `{source['metric']}`",
        f"- Analyzed modules: {report['analyzed_modules']:,}",
        f"- Analyzed directions: {report['analyzed_directions']:,}",
        f"- Full matrix-rank bound: {report['full_matrix_rank_bound']:,}",
        f"- Analyzed weights: {report['analyzed_parameters']:,}",
        "",
        "> Activation energy measures intermediate-output contribution on the calibration "
        "dataset. It is not a causal guarantee of final-answer importance.",
        "",
        "## Projection structure",
        "",
        "| Projection | Modules | Energy share | Per-module top-90 directions | Per-module top-95 directions |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["projection_summary"]:
        lines.append(
            f"| {row['projection']} | {row['modules']} "
            f"| {_percent(row['aggregate_energy_fraction'])} "
            f"| {_percent(row['per_module_top90_direction_fraction'])} "
            f"| {_percent(row['per_module_top95_direction_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Layer concentration",
            "",
            "| Layers | Modules | Energy share | Per-module top-95 directions |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["layer_summary"]:
        lines.append(
            f"| {row['layers']} | {row['modules']} "
            f"| {_percent(row['aggregate_energy_fraction'])} "
            f"| {_percent(row['per_module_top95_direction_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Global-threshold starvation",
            "",
            "| Energy target | Directions | Direction share | Zero modules | Modules <1% rank | Modules <10% rank | Median module energy kept | P10 module energy kept |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["global_thresholds"]:
        lines.append(
            f"| {_percent(row['threshold'])} | {row['directions']:,} "
            f"| {_percent(row['direction_fraction'])} | {row['zero_direction_modules']} "
            f"| {row['modules_below_1pct_directions']} "
            f"| {row['modules_below_10pct_directions']} "
            f"| {_percent(row['median_module_energy_retained'])} "
            f"| {_percent(row['p10_module_energy_retained'])} |"
        )
    lines.extend(
        [
            "",
            "## Whole-tensor GGUF policies",
            "",
            "These policies rank complete modules by aggregate activation energy. A module "
            "remains in the tier where it starts, so realized shares can cross a nominal boundary.",
            "",
            "| Policy | Precision tiers | Realized analyzed-energy shares | Analyzed-weight shares |",
            "|---|---|---|---|",
        ]
    )
    for row in report["quantization_policies"]:
        energy = ", ".join(
            f"{key}: {value:.2f}%" for key, value in row["energy_percent_by_precision"].items()
        )
        weights = ", ".join(
            f"{key}: {value:.2f}%"
            for key, value in row["analyzed_weight_percent_by_precision"].items()
        )
        lines.append(
            f"| {row['name']} | {' / '.join(row['tiers'])} | {energy} | {weights} |"
        )
    stability = report["cross_sample_stability"]
    lines.extend(
        [
            "",
            "## Cross-sample stability",
            "",
            f"Among {stability['directions']:,} aggregate per-module top-95 directions, "
            f"{_percent(stability['selected_in_all_samples_fraction'])} appeared in every "
            f"sample's top-95 set, "
            f"{_percent(stability['selected_in_at_least_90pct_samples_fraction'])} appeared "
            "in at least 90% of samples, and "
            f"{_percent(stability['selected_in_at_least_50pct_samples_fraction'])} appeared "
            "in at least 50% of samples.",
            "",
            "## Highest-energy modules",
            "",
            "| Rank | Module | Aggregate energy share |",
            "|---:|---|---:|",
        ]
    )
    for index, module in enumerate(report["top_modules"], start=1):
        lines.append(
            f"| {index} | `{module['module']}` | {_percent(module['energy_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A small direction share can capture high aggregate energy while starving some modules.",
            "- The GGUF policies use whole-tensor precision because unchanged llama.cpp kernels do "
            "not support different quantization types for individual SVD directions.",
            "- Unanalyzed quantizable tensors inherit the highest precision in each policy.",
            "- Evaluate every checkpoint on held-out data; calibration energy alone cannot establish accuracy.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(analysis_path: Path, json_path: Path, markdown_path: Path) -> None:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    report = build_report(analysis)
    atomic_json(json_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown(report), encoding="utf-8")
    print(f"[done] wrote {json_path} and {markdown_path}", flush=True)
