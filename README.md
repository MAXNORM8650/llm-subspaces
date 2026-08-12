# Subspace-Guided GGUF Quantization

Analyze activation energy in model subspaces and create six mixed-precision
GGUF checkpoints compatible with unmodified `llama.cpp`.

## Supported inputs

- Hugging Face or local model directory
- Matching BF16/F16 GGUF
- Hugging Face dataset or local Arrow, JSON/JSONL, CSV, Parquet, or text file
- Qwen/Llama-style modules:
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

The included GGUF tensor mapper supports Qwen/Llama-style
`model.layers.N.{self_attn,mlp}` names.

## Quantization policies

Modules are ranked by aggregate activation energy. Complete tensors—not
individual SVD directions—receive each precision.

| Checkpoint | High-energy tier | Medium tier | Low tier |
|---|---|---|---|
| `Q8Q6Q4` | Q8_0 to 90% | Q6_K to 98% | Q4_K remainder |
| `Q6Q5Q4` | Q6_K to 90% | Q5_K to 98% | Q4_K remainder |
| `Q5Q4Q3` | Q5_K to 90% | Q4_K to 98% | Q3_K remainder |
| `Q4Q3Q2` | Q4_K to 90% | Q3_K to 98% | Q2_K remainder |
| `Q3Q2IQ1S` | Q3_K to 90% | Q2_K to 98% | IQ1_S remainder |
| `Q2IQ1S` | Q2_K to 80% | — | IQ1_S remainder |

Unanalyzed quantizable tensors use the highest precision in each policy.
Realized energy shares may cross nominal boundaries because tensors are not
split.

## Requirements

- Python 3.10+
- CUDA- or ROCm-enabled PyTorch
- Recent `llama.cpp` with:
  - `llama-imatrix`
  - `llama-quantize`
  - `llama-server`
- Slurm GPU allocation

## Installation

```bash
python -m pip install -e '.[dev]'
```

## Configuration

```bash
cp configs/qwen35_wikitext_example.json configs/config.local.json
```

Edit these fields:

- `model.hf_model`
- `model.bf16_gguf`
- `dataset.source` and one of:
  - `text_field`
  - `messages_field`
  - `prompt_template`
- `llama_cpp.imatrix_bin`
- `llama_cpp.quantize_bin`
- `llama_cpp.server_bin`

Paths in the configuration are resolved relative to the configuration file.

### Dataset examples

Text column:

```json
{
  "source": "wikitext",
  "config": "wikitext-2-raw-v1",
  "split": "train",
  "text_field": "text",
  "messages_field": null,
  "prompt_template": null
}
```

JSONL prompt template:

```json
{
  "source": "/data/questions.jsonl",
  "text_field": null,
  "messages_field": null,
  "prompt_template": "Question: {question}\n\nAnswer:"
}
```

## Run

Enter a GPU node first:

```bash
srun --overlap --jobid=YOUR_JOB_ID --pty bash -l
source /path/to/environment/bin/activate

export CUDA_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
```

Analysis and report:

```bash
scripts/run_analysis.sh configs/config.local.json
```

Build and smoke-test all six checkpoints:

```bash
scripts/build_models.sh configs/config.local.json
```

Run the complete pipeline:

```bash
scripts/run_pipeline.sh configs/config.local.json
```

All model-processing commands refuse to run outside a GPU allocation.

## Outputs

```text
artifacts/
├── atlas/
├── analysis/
│   ├── calibration.txt
│   └── subspace_analysis.json
├── reports/
│   ├── SUBSPACE_EFFECTIVENESS.md
│   └── subspace_report.json
└── quantization/
    ├── calibration.imatrix.gguf
    ├── checkpoint_index.json
    ├── inspections/
    ├── logs/
    ├── models/
    ├── policies/
    ├── recipes/
    └── smoke_tests.json
```

Regenerate only the report:

```bash
scripts/generate_report.sh \
  artifacts/analysis/subspace_analysis.json \
  artifacts/reports
```

## Validation

```bash
ruff check src tests
pytest -q
```

Subspace energy measures intermediate-output contribution on calibration data.
Evaluate every checkpoint on separate held-out tasks before use.

## License

MIT
