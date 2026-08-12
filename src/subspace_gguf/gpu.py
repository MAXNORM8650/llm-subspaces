"""Safety checks for commands that load or quantize models."""

from __future__ import annotations

import os
from pathlib import Path


def require_gpu_job() -> None:
    """Refuse expensive model work outside an allocated GPU environment.

    Slurm is detected via ``SLURM_JOB_ID``. AMD GPU access is detected via
    ``/dev/kfd``; NVIDIA access is detected via device nodes or Slurm's GRES
    environment. Set ``SUBSPACE_GGUF_ALLOW_NON_SLURM_GPU=1`` only for a local
    GPU workstation.
    """

    allow_local = os.environ.get("SUBSPACE_GGUF_ALLOW_NON_SLURM_GPU") == "1"
    in_slurm = bool(os.environ.get("SLURM_JOB_ID"))
    has_amd = Path("/dev/kfd").exists()
    has_nvidia = any(Path("/dev").glob("nvidia[0-9]*"))
    slurm_gpu = "gpu" in os.environ.get("SLURM_JOB_GPUS", "").lower()

    if not (has_amd or has_nvidia or slurm_gpu):
        raise RuntimeError("No AMD/NVIDIA GPU device is visible; refusing model work")
    if not in_slurm and not allow_local:
        raise RuntimeError(
            "Not inside a Slurm job; refusing model work. On a local GPU workstation, "
            "set SUBSPACE_GGUF_ALLOW_NON_SLURM_GPU=1 explicitly."
        )


def selected_gpu_environment() -> dict[str, str | None]:
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
    }
