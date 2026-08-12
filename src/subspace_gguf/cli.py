"""Command-line interface for the complete analysis and quantization pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import DEFAULT_THRESHOLDS, analyze_dataset
from .atlas import DEFAULT_TARGET_SUFFIXES, extract_atlas
from .gguf import build_checkpoints, build_imatrix, smoke_test
from .recipes import write_all_recipes
from .report import write_report
from .util import atomic_json


ALL_STAGES = ("atlas", "analyze", "report", "imatrix", "quantize", "smoke")


def _path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _model_reference(base: Path, value: str) -> str:
    candidate = Path(value).expanduser()
    if (
        candidate.is_absolute()
        or value.startswith(("./", "../"))
        or (base / candidate).exists()
    ):
        return str(_path(base, value))
    return value


def _dataset_reference(base: Path, value: str) -> str:
    candidate = Path(value).expanduser()
    if (
        candidate.is_absolute()
        or value.startswith(("./", "../"))
        or (base / candidate).exists()
    ):
        return str(_path(base, value))
    return value


def _load_config(path: Path) -> tuple[dict, Path]:
    resolved = path.expanduser().resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    for section in ("model", "dataset", "atlas", "analysis", "report", "llama_cpp", "quantization"):
        if section not in config:
            raise KeyError(f"configuration is missing section {section!r}")
    return config, resolved.parent


def run_pipeline(config_path: Path, stages: tuple[str, ...], resume: bool) -> None:
    config, base = _load_config(config_path)
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")

    model = config["model"]
    dataset = config["dataset"]
    atlas = config["atlas"]
    analysis = config["analysis"]
    report = config["report"]
    llama_cpp = config["llama_cpp"]
    quantization = config["quantization"]

    model_path = _model_reference(base, model["hf_model"])
    atlas_dir = _path(base, atlas["output_dir"])
    analysis_json = _path(base, analysis["output_json"])
    calibration_text = _path(base, analysis["calibration_text"])
    report_json = _path(base, report["output_json"])
    report_markdown = _path(base, report["output_markdown"])
    output_dir = _path(base, quantization["output_dir"])
    bf16_gguf = _path(base, model["bf16_gguf"])
    imatrix_path = _path(base, quantization["imatrix"])

    if "atlas" in stages:
        if resume and (atlas_dir / "index.json").exists():
            print(f"[resume] atlas exists: {atlas_dir}", flush=True)
        else:
            extract_atlas(
                model_path=model_path,
                output_dir=atlas_dir,
                energy=float(atlas.get("energy", 1.0)),
                max_rank=int(atlas.get("max_rank", 0)),
                device=str(config.get("runtime", {}).get("device", "cuda")),
                storage_dtype=str(atlas.get("storage_dtype", "float16")),
                target_suffixes=tuple(atlas.get("target_suffixes", DEFAULT_TARGET_SUFFIXES)),
                local_files_only=bool(model.get("local_files_only", False)),
                trust_remote_code=bool(model.get("trust_remote_code", False)),
                svd_driver=atlas.get("svd_driver"),
            )

    if "analyze" in stages:
        if resume and analysis_json.exists() and calibration_text.exists():
            print(f"[resume] analysis exists: {analysis_json}", flush=True)
        else:
            analyze_dataset(
                model_path=model_path,
                atlas_dir=atlas_dir,
                dataset_source=_dataset_reference(base, str(dataset["source"])),
                dataset_config=dataset.get("config"),
                dataset_split=str(dataset.get("split", "train")),
                data_files=(
                    str(_path(base, dataset["data_files"]))
                    if dataset.get("data_files")
                    else None
                ),
                text_field=dataset.get("text_field"),
                messages_field=dataset.get("messages_field"),
                prompt_template=dataset.get("prompt_template"),
                system_prompt=dataset.get("system_prompt"),
                samples=int(dataset.get("samples", 100)),
                seed=int(dataset.get("seed", 42)),
                max_length=int(dataset.get("max_length", 2048)),
                device=str(config.get("runtime", {}).get("device", "cuda")),
                projection_dtype=str(
                    config.get("runtime", {}).get("projection_dtype", "float16")
                ),
                thresholds=tuple(analysis.get("thresholds", DEFAULT_THRESHOLDS)),
                output_json=analysis_json,
                calibration_text=calibration_text,
                local_files_only=bool(model.get("local_files_only", False)),
                trust_remote_code=bool(model.get("trust_remote_code", False)),
            )

    if "report" in stages:
        write_report(analysis_json, report_json, report_markdown)
        write_all_recipes(
            analysis_json, output_dir, str(model.get("architecture_mapper", "qwen-llama"))
        )

    if "imatrix" in stages:
        if resume and imatrix_path.exists():
            print(f"[resume] importance matrix exists: {imatrix_path}", flush=True)
        else:
            imatrix_path.parent.mkdir(parents=True, exist_ok=True)
            build_imatrix(
                llama_imatrix=_path(base, llama_cpp["imatrix_bin"]),
                bf16_gguf=bf16_gguf,
                calibration_text=calibration_text,
                output_path=imatrix_path,
                context_size=int(llama_cpp.get("imatrix_context_size", 2048)),
                chunks=int(llama_cpp.get("imatrix_chunks", -1)),
            )

    if "quantize" in stages:
        build_checkpoints(
            analysis_path=analysis_json,
            bf16_gguf=bf16_gguf,
            imatrix_path=imatrix_path,
            output_dir=output_dir,
            llama_quantize=_path(base, llama_cpp["quantize_bin"]),
            architecture=str(model.get("architecture_mapper", "qwen-llama")),
            threads=int(llama_cpp.get("quantize_threads", 16)),
            skip_existing=resume,
        )

    if "smoke" in stages:
        smoke_results = []
        starting_port = int(llama_cpp.get("smoke_port", 8090))
        model_paths = sorted((output_dir / "models").glob("Subspace-*.gguf"))
        if not model_paths:
            raise FileNotFoundError(f"no checkpoints found in {output_dir / 'models'}")
        for index, checkpoint in enumerate(model_paths):
            result = smoke_test(
                model_path=checkpoint,
                llama_server=_path(base, llama_cpp["server_bin"]),
                port=starting_port + index,
                context_size=int(llama_cpp.get("smoke_context_size", 2048)),
                output_log=output_dir / "logs" / f"smoke-{checkpoint.stem}.log",
            )
            smoke_results.append(result)
            print(f"[smoke] {checkpoint.name}: {result['response']!r}", flush=True)
        atomic_json(output_dir / "smoke_tests.json", smoke_results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="subspace-gguf")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pipeline = subparsers.add_parser("pipeline", help="run configured pipeline stages")
    pipeline.add_argument("--config", type=Path, required=True)
    pipeline.add_argument(
        "--stages",
        default=",".join(ALL_STAGES),
        help=f"comma-separated subset of: {','.join(ALL_STAGES)}",
    )
    pipeline.add_argument("--resume", action="store_true")

    report = subparsers.add_parser("report", help="regenerate report without model loading")
    report.add_argument("--analysis", type=Path, required=True)
    report.add_argument("--output-json", type=Path, required=True)
    report.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "pipeline":
        stages = tuple(value.strip() for value in args.stages.split(",") if value.strip())
        run_pipeline(args.config, stages, args.resume)
    elif args.command == "report":
        write_report(args.analysis, args.output_json, args.output_markdown)


if __name__ == "__main__":
    main()
