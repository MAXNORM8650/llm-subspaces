"""Measure activation-weighted energy in a precomputed SVD atlas."""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .data import load_rows, sample_examples
from .gpu import require_gpu_job, selected_gpu_environment
from .model import causal_model_class
from .util import atomic_json, threshold_token, validate_thresholds


DEFAULT_THRESHOLDS = (
    0.80,
    0.85,
    0.875,
    0.90,
    0.91,
    0.92,
    0.93,
    0.94,
    0.95,
    0.97,
    0.98,
    0.99,
    0.995,
)


def _atlas_file(atlas_dir: Path, name: str) -> Path:
    return atlas_dir / f"{name.replace('.', '__')}.pt"


def _top_fraction(values, fraction: float):
    import torch

    order = torch.argsort(values, descending=True)
    total = values.sum()
    if total <= 0:
        return order[:0], 0.0
    cumulative = torch.cumsum(values[order], dim=0)
    count = int(torch.searchsorted(cumulative, total * fraction).item()) + 1
    return order[:count], float(cumulative[count - 1] / total)


def _module_metadata(name: str) -> dict:
    match = re.search(
        r"layers\.(\d+).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
        name,
    )
    return {
        "layer": int(match.group(1)) if match else None,
        "projection": match.group(2) if match else name.rsplit(".", 1)[-1],
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def serialize_messages(tokenizer, messages: list[dict[str, str]]) -> str:
    add_generation_prompt = messages[-1]["role"] != "assistant"
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except (TypeError, ValueError):
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except (TypeError, ValueError):
            return "\n\n".join(
                f"{message['role'].upper()}: {message['content']}" for message in messages
            )


class EnergyCollector:
    def __init__(self, model, atlas_dir: Path, device: str, projection_dtype):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.projection_dtype = projection_dtype
        self.index = json.loads((atlas_dir / "index.json").read_text(encoding="utf-8"))
        self.names = list(self.index["modules"])
        model_modules = dict(model.named_modules())
        missing = [name for name in self.names if name not in model_modules]
        if missing:
            raise KeyError(f"atlas modules missing from model: {missing[:5]}")

        self.bases = {}
        self.energy = {}
        self.sample_energy = {}
        self.sample_top95_frequency = {}
        self.weight_energy = {}
        self.atlas_weight_energy = {}
        self.full_output_energy = {}
        self.hooks = []
        self.enabled = False
        for position, name in enumerate(self.names, start=1):
            data = torch.load(_atlas_file(atlas_dir, name), map_location="cpu", weights_only=True)
            weight = model_modules[name].weight.detach().float()
            self.weight_energy[name] = float(weight.square().sum().cpu())
            self.atlas_weight_energy[name] = float(data["sigma"].float().square().sum())
            basis = data["V"].to(device=self.device, dtype=projection_dtype)
            singular = data["sigma"].to(device=self.device, dtype=torch.float32)
            rank = singular.numel()
            self.bases[name] = (basis, singular)
            self.energy[name] = torch.zeros(rank, device=self.device, dtype=torch.float64)
            self.sample_energy[name] = torch.zeros(rank, device=self.device, dtype=torch.float32)
            self.sample_top95_frequency[name] = torch.zeros(
                rank, device=self.device, dtype=torch.int32
            )
            self.full_output_energy[name] = torch.zeros(
                (), device=self.device, dtype=torch.float64
            )
            self.hooks.append(
                model_modules[name].register_forward_pre_hook(self._input_hook(name))
            )
            self.hooks.append(
                model_modules[name].register_forward_hook(self._output_hook(name))
            )
            if position % 16 == 0 or position == len(self.names):
                print(f"[atlas] loaded {position}/{len(self.names)} modules", flush=True)

    def _input_hook(self, name):
        def hook(_module, args):
            if not self.enabled:
                return
            inputs = args[0].reshape(-1, args[0].shape[-1])
            basis, singular = self.bases[name]
            projection = inputs.to(self.projection_dtype) @ basis
            weighted = projection.float() * singular
            self.sample_energy[name].add_(weighted.square().sum(dim=0))

        return hook

    def _output_hook(self, name):
        def hook(_module, _args, output):
            if self.enabled:
                self.full_output_energy[name].add_(output.float().square().sum().double())

        return hook

    def start_sample(self) -> None:
        for values in self.sample_energy.values():
            values.zero_()
        self.enabled = True

    def finish_sample(self) -> None:
        self.enabled = False
        for name, values in self.sample_energy.items():
            self.energy[name].add_(values.double())
            selected, _ = _top_fraction(values, 0.95)
            self.sample_top95_frequency[name][selected] += 1

    def close(self) -> None:
        self.enabled = False
        for hook in self.hooks:
            hook.remove()


def build_analysis(
    *,
    collector: EnergyCollector,
    metadata: dict,
    row_indices: list[int],
    token_lengths: list[int],
    truncated_samples: int,
    elapsed_seconds: float,
    thresholds: tuple[float, ...],
) -> dict:
    torch = collector.torch
    modules = {}
    flat_energy = []
    offsets = {}
    module_totals = {}
    offset = 0
    for name in collector.names:
        energy = collector.energy[name].cpu()
        frequency = collector.sample_top95_frequency[name].cpu()
        rank = energy.numel()
        total = float(energy.sum())
        full_output = float(collector.full_output_energy[name].cpu())
        module_totals[name] = total
        offsets[name] = (offset, offset + rank)
        flat_energy.append(energy)
        offset += rank
        module = {
            **_module_metadata(name),
            "shape": collector.index["modules"][name]["shape"],
            "parameters": math.prod(collector.index["modules"][name]["shape"]),
            "atlas_rank": rank,
            "full_matrix_rank_bound": min(collector.index["modules"][name]["shape"]),
            "atlas_weight_energy_fraction": _safe_ratio(
                collector.atlas_weight_energy[name], collector.weight_energy[name]
            ),
            "full_output_energy_total": full_output,
            "energy_total": total,
            "atlas_activation_energy_fraction": _safe_ratio(total, full_output),
            "direction_energy": [float(value) for value in energy.tolist()],
        }
        for threshold in thresholds:
            selected, captured = _top_fraction(energy, threshold)
            token = threshold_token(threshold)
            module.update(
                {
                    f"energy_top{token}_count": selected.numel(),
                    f"energy_top{token}_rank_fraction": selected.numel() / rank,
                    f"energy_top{token}_captured_fraction": captured,
                    f"energy_top{token}_indices": selected.tolist(),
                }
            )
            if threshold == 0.95:
                module["energy_top95_sample_frequency"] = frequency[selected].tolist()
        modules[name] = module

    flat = torch.cat(flat_energy)
    global_fields = {}
    for threshold in thresholds:
        selected, captured = _top_fraction(flat, threshold)
        mask = torch.zeros(flat.numel(), dtype=torch.bool)
        mask[selected] = True
        token = threshold_token(threshold)
        global_fields.update(
            {
                f"global_energy_top{token}_count": selected.numel(),
                f"global_energy_top{token}_direction_fraction": selected.numel() / flat.numel(),
                f"global_energy_top{token}_captured_fraction": captured,
            }
        )
        for name, (start, end) in offsets.items():
            local = torch.nonzero(mask[start:end], as_tuple=False).flatten()
            selected_energy = float(flat[start:end][local].sum())
            modules[name].update(
                {
                    f"global_energy_top{token}_count": local.numel(),
                    f"global_energy_top{token}_rank_fraction": local.numel() / (end - start),
                    f"global_energy_top{token}_module_energy_fraction": _safe_ratio(
                        selected_energy, module_totals[name]
                    ),
                    f"global_energy_top{token}_indices": local.tolist(),
                }
            )

    all_energy = sum(module_totals.values())
    full_output = sum(float(value.cpu()) for value in collector.full_output_energy.values())
    return {
        "metadata": {
            **metadata,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "metric": "sum_over_tokens((sigma_j * dot(v_j, x)) ** 2)",
            "elapsed_seconds": elapsed_seconds,
            "global_energy_thresholds": list(thresholds),
            "gpu_environment": selected_gpu_environment(),
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "sample": {
            "dataset_row_indices": row_indices,
            "token_lengths": token_lengths,
            "total_tokens": sum(token_lengths),
            "truncated_samples": truncated_samples,
        },
        "coverage": {
            "atlas_modules": len(collector.names),
            "atlas_directions": flat.numel(),
            "full_matrix_rank_bound": sum(
                min(value["shape"]) for value in collector.index["modules"].values()
            ),
            "global_energy_total": all_energy,
            "full_output_energy_total": full_output,
            "atlas_activation_energy_fraction_overall": _safe_ratio(all_energy, full_output),
            **global_fields,
            "top20_modules_by_raw_energy": [
                {
                    "module": name,
                    "energy": module_totals[name],
                    "energy_fraction": _safe_ratio(module_totals[name], all_energy),
                }
                for name in sorted(module_totals, key=module_totals.get, reverse=True)[:20]
            ],
        },
        "modules": modules,
    }


def analyze_dataset(
    *,
    model_path: str,
    atlas_dir: Path,
    dataset_source: str,
    dataset_config: str | None,
    dataset_split: str,
    data_files: str | None,
    text_field: str | None,
    messages_field: str | None,
    prompt_template: str | None,
    system_prompt: str | None,
    samples: int,
    seed: int,
    max_length: int,
    device: str,
    projection_dtype: str,
    thresholds: tuple[float, ...],
    output_json: Path,
    calibration_text: Path,
    local_files_only: bool,
    trust_remote_code: bool,
) -> None:
    require_gpu_job()
    import torch
    from transformers import AutoTokenizer

    thresholds = validate_thresholds(thresholds)
    if 0.95 not in thresholds:
        raise ValueError("thresholds must include 0.95 for cross-sample stability")
    rows = load_rows(
        dataset_source, split=dataset_split, config=dataset_config, data_files=data_files
    )
    examples = sample_examples(
        rows,
        samples=samples,
        seed=seed,
        text_field=text_field,
        messages_field=messages_field,
        prompt_template=prompt_template,
        system_prompt=system_prompt,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    rendered = [serialize_messages(tokenizer, example.messages) for example in examples]
    calibration_text.parent.mkdir(parents=True, exist_ok=True)
    calibration_text.write_text("\n".join(rendered) + "\n", encoding="utf-8")

    print(f"[model] loading {model_path}", flush=True)
    model_class = causal_model_class(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = model_class.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    collector = EnergyCollector(model, atlas_dir, device, getattr(torch, projection_dtype))
    token_lengths = []
    truncated = 0
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for position, text in enumerate(rendered, start=1):
                encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                original_length = int(input_ids.shape[1])
                if original_length > max_length:
                    input_ids = input_ids[:, -max_length:]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, -max_length:]
                    truncated += 1
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device) if attention_mask is not None else None
                collector.start_sample()
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=False,
                )
                collector.finish_sample()
                token_lengths.append(int(input_ids.shape[1]))
                if position % 5 == 0 or position == len(rendered):
                    print(
                        f"[analysis] {position}/{len(rendered)} samples; "
                        f"tokens={sum(token_lengths):,}",
                        flush=True,
                    )
    finally:
        collector.close()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    analysis = build_analysis(
        collector=collector,
        metadata={
            "model": model_path,
            "atlas": str(atlas_dir.resolve()),
            "dataset": dataset_source,
            "dataset_config": dataset_config,
            "dataset_split": dataset_split,
            "samples": samples,
            "sample_seed": seed,
            "max_length": max_length,
            "projection_dtype": projection_dtype,
            "calibration_text": str(calibration_text.resolve()),
        },
        row_indices=[example.row_index for example in examples],
        token_lengths=token_lengths,
        truncated_samples=truncated,
        elapsed_seconds=elapsed,
        thresholds=thresholds,
    )
    atomic_json(output_json, analysis)
    print(f"[done] wrote {output_json}", flush=True)
