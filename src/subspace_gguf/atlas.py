"""Extract right-singular-vector atlases from Hugging Face linear weights."""

from __future__ import annotations

import gc
from datetime import datetime, timezone
from pathlib import Path

from .gpu import require_gpu_job, selected_gpu_environment
from .model import causal_model_class
from .util import atomic_json


DEFAULT_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _atlas_file(output: Path, name: str) -> Path:
    return output / f"{name.replace('.', '__')}.pt"


def extract_atlas(
    *,
    model_path: str,
    output_dir: Path,
    energy: float,
    max_rank: int,
    device: str,
    storage_dtype: str,
    target_suffixes: tuple[str, ...],
    local_files_only: bool,
    trust_remote_code: bool,
    svd_driver: str | None,
) -> None:
    require_gpu_job()
    import torch

    if not 0.0 < energy <= 1.0:
        raise ValueError("energy must be in (0, 1]")
    if max_rank < 0:
        raise ValueError("max_rank must be non-negative; use 0 for no cap")
    dtype = getattr(torch, storage_dtype)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json"
    if index_path.exists():
        raise FileExistsError(f"refusing to overwrite existing atlas: {index_path}")

    print(f"[model] loading {model_path} on CPU for streamed SVD", flush=True)
    model_class = causal_model_class(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = model_class.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    ).eval()
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in target_suffixes
    ]
    if not targets:
        raise RuntimeError(
            "no matching linear modules found; adjust --target-suffixes for this architecture"
        )

    modules = {}
    for position, (name, module) in enumerate(targets, start=1):
        destination = _atlas_file(output_dir, name)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite atlas tensor: {destination}")
        weight = module.weight.detach().float().to(device)
        kwargs = {"full_matrices": False}
        if svd_driver:
            kwargs["driver"] = svd_driver
        _u, singular, vh = torch.linalg.svd(weight, **kwargs)
        squared = singular.square()
        cumulative = torch.cumsum(squared, dim=0) / squared.sum()
        rank = int(torch.searchsorted(cumulative, energy).item()) + 1
        if max_rank:
            rank = min(rank, max_rank)
        basis = vh[:rank].transpose(0, 1).contiguous().to(dtype).cpu()
        saved_singular = singular[:rank].contiguous().to(dtype).cpu()
        torch.save(
            {
                "V": basis,
                "sigma": saved_singular,
                "eff_rank": rank,
                "shape": tuple(weight.shape),
            },
            destination,
        )
        modules[name] = {
            "eff_rank": rank,
            "shape": list(weight.shape),
            "input_features": int(weight.shape[1]),
            "full_rank_bound": min(weight.shape),
        }
        del weight, _u, singular, vh, squared, cumulative, basis, saved_singular
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[atlas] {position}/{len(targets)} {name} rank={rank}", flush=True)

    atomic_json(
        index_path,
        {
            "format": "subspace_gguf_atlas_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_path,
            "energy": energy,
            "max_rank": max_rank,
            "storage_dtype": storage_dtype,
            "target_suffixes": list(target_suffixes),
            "gpu_environment": selected_gpu_environment(),
            "modules": modules,
        },
    )
    print(f"[done] wrote {len(modules)}-module atlas to {output_dir}", flush=True)
