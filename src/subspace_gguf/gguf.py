"""Build and inspect llama.cpp GGUF checkpoints."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from .gpu import require_gpu_job, selected_gpu_environment
from .policies import DEFAULT_POLICIES
from .recipes import write_all_recipes
from .util import atomic_json, sha256


def _run(command: list[str], *, log_path: Path | None = None) -> None:
    print("[run] " + " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def build_imatrix(
    *,
    llama_imatrix: Path,
    bf16_gguf: Path,
    calibration_text: Path,
    output_path: Path,
    context_size: int,
    chunks: int,
) -> None:
    require_gpu_job()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite importance matrix: {output_path}")
    command = [
        str(llama_imatrix),
        "--model",
        str(bf16_gguf),
        "--file",
        str(calibration_text),
        "--output-file",
        str(output_path),
        "--ctx-size",
        str(context_size),
        "--n-gpu-layers",
        "all",
        "--parse-special",
    ]
    if chunks >= 0:
        command.extend(["--chunks", str(chunks)])
    _run(command, log_path=output_path.with_suffix(".log"))


def inspect_gguf(path: Path) -> dict:
    from gguf import GGUFReader

    reader = GGUFReader(path)
    parameters = Counter()
    stored_bytes = Counter()
    tensor_counts = Counter()
    for tensor in reader.tensors:
        precision = tensor.tensor_type.name
        parameters[precision] += int(tensor.n_elements)
        stored_bytes[precision] += int(tensor.n_bytes)
        tensor_counts[precision] += 1
    total_parameters = sum(parameters.values())
    total_tensor_bytes = sum(stored_bytes.values())
    return {
        "path": str(path.resolve()),
        "exact_file_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "tensor_count": len(reader.tensors),
        "total_parameters": total_parameters,
        "average_stored_bits_per_parameter": 8.0 * total_tensor_bytes / total_parameters,
        "parameter_percent_by_precision": {
            key: 100.0 * value / total_parameters for key, value in parameters.items()
        },
        "tensor_count_by_precision": dict(tensor_counts),
        "stored_bytes_by_precision": dict(stored_bytes),
        "metadata_and_alignment_bytes": path.stat().st_size - total_tensor_bytes,
    }


def build_checkpoints(
    *,
    analysis_path: Path,
    bf16_gguf: Path,
    imatrix_path: Path,
    output_dir: Path,
    llama_quantize: Path,
    architecture: str,
    threads: int,
    skip_existing: bool,
) -> list[dict]:
    require_gpu_job()
    for required in (analysis_path, bf16_gguf, imatrix_path, llama_quantize):
        if not required.exists():
            raise FileNotFoundError(required)
    manifests = write_all_recipes(analysis_path, output_dir, architecture)
    results = []
    for policy, manifest in zip(DEFAULT_POLICIES, manifests):
        destination = output_dir / "models" / f"Subspace-{policy.name}.gguf"
        destination.parent.mkdir(parents=True, exist_ok=True)
        inspection_path = output_dir / "inspections" / f"{policy.name}.json"
        if destination.exists() and not skip_existing:
            raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
        if not destination.exists():
            _run(
                [
                    str(llama_quantize),
                    "--imatrix",
                    str(imatrix_path),
                    "--pure",
                    "--tensor-type-file",
                    manifest["recipe"],
                    str(bf16_gguf),
                    str(destination),
                    policy.tiers[0],
                    str(threads),
                ],
                log_path=output_dir / "logs" / f"quantize-{policy.name}.log",
            )
        inspection = inspect_gguf(destination)
        inspection.update(
            {
                "policy": policy.name,
                "policy_manifest": str(
                    (output_dir / "policies" / f"{policy.name}.json").resolve()
                ),
                "gpu_environment": selected_gpu_environment(),
            }
        )
        atomic_json(inspection_path, inspection)
        results.append(inspection)
        print(
            f"[done] {policy.name}: {inspection['exact_file_bytes'] / 2**30:.2f} GiB, "
            f"{inspection['average_stored_bits_per_parameter']:.2f} bpw",
            flush=True,
        )
    atomic_json(output_dir / "checkpoint_index.json", results)
    return results


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def smoke_test(
    *,
    model_path: Path,
    llama_server: Path,
    port: int,
    context_size: int,
    output_log: Path,
) -> dict:
    require_gpu_job()
    alias = "subspace-smoke-test"
    command = [
        str(llama_server),
        "--model",
        str(model_path),
        "--alias",
        alias,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(context_size),
        "--parallel",
        "1",
        "--n-gpu-layers",
        "all",
        "--jinja",
        "--reasoning-format",
        "none",
    ]
    output_log.parent.mkdir(parents=True, exist_ok=True)
    with output_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        health = f"http://127.0.0.1:{port}/health"
        for _ in range(180):
            if process.poll() is not None:
                raise RuntimeError(f"llama-server exited; see {output_log}")
            try:
                with urllib.request.urlopen(health, timeout=2) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError(f"llama-server did not become ready; see {output_log}")
        result = _post_json(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            {
                "model": alias,
                "messages": [
                    {"role": "user", "content": "Reply with only the number: 2+2="}
                ],
                "temperature": 0,
                "max_tokens": 8,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            120,
        )
        content = result["choices"][0]["message"].get("content")
        if not content:
            raise RuntimeError(f"smoke generation returned no content for {model_path}")
        return {
            "model": str(model_path.resolve()),
            "response": content,
            "finish_reason": result["choices"][0].get("finish_reason"),
            "usage": result.get("usage"),
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
