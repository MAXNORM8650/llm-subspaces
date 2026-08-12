"""Map analyzed Hugging Face modules to llama.cpp tensor-type recipes."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .policies import DEFAULT_POLICIES, QuantizationPolicy, assign_modules
from .util import atomic_json, sha256


def hf_to_gguf(name: str, architecture: str = "qwen-llama") -> str:
    if architecture != "qwen-llama":
        raise ValueError(f"unsupported GGUF mapper: {architecture}")
    match = re.search(
        r"(?:^|\.)layers\.(\d+)\.(mlp|self_attn)\.([a-z_]+)$", name
    )
    if match is None:
        raise ValueError(f"cannot map Hugging Face module name {name!r} to GGUF")
    layer, family, projection = match.groups()
    if family == "mlp":
        suffix = {
            "gate_proj": "ffn_gate",
            "up_proj": "ffn_up",
            "down_proj": "ffn_down",
        }.get(projection)
    else:
        suffix = {
            "q_proj": "attn_q",
            "k_proj": "attn_k",
            "v_proj": "attn_v",
            "o_proj": "attn_output",
        }.get(projection)
    if suffix is None:
        raise ValueError(f"unsupported projection in module {name!r}")
    return f"blk.{layer}.{suffix}.weight"


def write_policy_recipe(
    *,
    analysis_path: Path,
    policy: QuantizationPolicy,
    recipe_path: Path,
    manifest_path: Path,
    architecture: str,
) -> dict:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assignments = assign_modules(analysis["modules"], policy)
    mapped = []
    for assignment in assignments:
        mapped.append(
            {
                **assignment,
                "gguf_tensor": hf_to_gguf(assignment["module"], architecture),
            }
        )
    names = [value["gguf_tensor"] for value in mapped]
    if len(names) != len(set(names)):
        raise ValueError("multiple analyzed modules map to the same GGUF tensor")

    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    with recipe_path.open("w", encoding="utf-8") as stream:
        for value in sorted(mapped, key=lambda item: item["gguf_tensor"]):
            stream.write(
                f"^{re.escape(value['gguf_tensor'])}$={value['precision']}\n"
            )

    total_energy = sum(value["energy"] for value in mapped)
    total_parameters = sum(value["parameters"] for value in mapped)
    energy = Counter()
    parameters = Counter()
    tensors = Counter()
    for value in mapped:
        energy[value["precision"]] += value["energy"]
        parameters[value["precision"]] += value["parameters"]
        tensors[value["precision"]] += 1
    manifest = {
        "format": "subspace_gguf_whole_tensor_recipe_v1",
        "policy": policy.name,
        "exact_direction_level_mixing": False,
        "tiers_high_to_low": list(policy.tiers),
        "nominal_cumulative_energy_endpoints": list(policy.endpoints),
        "boundary_rule": "a whole module remains in the tier in which it starts",
        "unanalysed_quantizable_tensors": f"inherit base type {policy.tiers[0]}",
        "architecture_mapper": architecture,
        "analysis": str(analysis_path.resolve()),
        "analysis_sha256": sha256(analysis_path),
        "recipe": str(recipe_path.resolve()),
        "analyzed_tensor_count": len(mapped),
        "tensor_count_by_precision": {
            precision: tensors[precision] for precision in policy.tiers
        },
        "analyzed_energy_percent_by_precision": {
            precision: 100.0 * energy[precision] / total_energy for precision in policy.tiers
        },
        "analyzed_weight_percent_by_precision": {
            precision: 100.0 * parameters[precision] / total_parameters
            for precision in policy.tiers
        },
        "assignments": mapped,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def write_all_recipes(
    analysis_path: Path, output_dir: Path, architecture: str
) -> list[dict]:
    manifests = []
    for policy in DEFAULT_POLICIES:
        manifests.append(
            write_policy_recipe(
                analysis_path=analysis_path,
                policy=policy,
                recipe_path=output_dir / "recipes" / f"{policy.name}.txt",
                manifest_path=output_dir / "policies" / f"{policy.name}.json",
                architecture=architecture,
            )
        )
    return manifests
