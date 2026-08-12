"""Architecture-aware Hugging Face model loading."""

from __future__ import annotations


def causal_model_class(
    model_path: str, *, local_files_only: bool, trust_remote_code: bool
):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if config.model_type == "qwen3_5":
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "This Transformers version does not support Qwen3.5. Install a version "
                "that provides Qwen3_5ForConditionalGeneration."
            ) from error
        return Qwen3_5ForConditionalGeneration
    return AutoModelForCausalLM
