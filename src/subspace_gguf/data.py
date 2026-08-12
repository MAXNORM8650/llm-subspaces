"""Dataset loading and deterministic prompt rendering."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CalibrationExample:
    row_index: int
    messages: list[dict[str, str]]


def load_rows(
    source: str,
    *,
    split: str = "train",
    config: str | None = None,
    data_files: str | None = None,
):
    """Load a Hugging Face dataset name or a local Arrow/JSON/CSV/Parquet/text file."""

    from datasets import Dataset, load_dataset

    path = Path(source).expanduser()
    if path.is_file() and path.suffix == ".arrow":
        return Dataset.from_file(str(path))
    if path.is_file():
        loaders = {
            ".json": "json",
            ".jsonl": "json",
            ".csv": "csv",
            ".parquet": "parquet",
            ".txt": "text",
        }
        loader = loaders.get(path.suffix.lower())
        if loader is None:
            raise ValueError(f"unsupported local dataset extension: {path.suffix}")
        return load_dataset(loader, data_files=str(path), split=split)
    kwargs: dict[str, Any] = {"split": split}
    if data_files:
        kwargs["data_files"] = data_files
    return load_dataset(source, config, **kwargs)


def _messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not value:
        raise ValueError("messages field must contain a non-empty list")
    result = []
    for message in value:
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            raise ValueError("each message must have role and content fields")
        result.append({"role": str(message["role"]), "content": str(message["content"])})
    return result


def render_row(
    row: dict[str, Any],
    *,
    text_field: str | None,
    messages_field: str | None,
    prompt_template: str | None,
    system_prompt: str | None,
) -> list[dict[str, str]]:
    choices = sum(value is not None for value in (text_field, messages_field, prompt_template))
    if choices != 1:
        raise ValueError(
            "select exactly one input format: text_field, messages_field, or prompt_template"
        )
    if messages_field:
        if messages_field not in row:
            raise KeyError(f"dataset row has no messages field {messages_field!r}")
        messages = _messages(row[messages_field])
    else:
        if text_field:
            if text_field not in row:
                raise KeyError(f"dataset row has no text field {text_field!r}")
            prompt = str(row[text_field])
        else:
            try:
                prompt = str(prompt_template).format_map(row)
            except KeyError as error:
                raise KeyError(f"prompt template references missing field {error.args[0]!r}") from error
        messages = [{"role": "user", "content": prompt}]
    if system_prompt and (not messages or messages[0]["role"] != "system"):
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def sample_examples(
    rows,
    *,
    samples: int,
    seed: int,
    text_field: str | None,
    messages_field: str | None,
    prompt_template: str | None,
    system_prompt: str | None,
) -> list[CalibrationExample]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    eligible = []
    for index in range(len(rows)):
        messages = render_row(
            dict(rows[index]),
            text_field=text_field,
            messages_field=messages_field,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
        )
        if any(message["content"].strip() for message in messages):
            eligible.append(CalibrationExample(row_index=index, messages=messages))
    if samples > len(eligible):
        raise ValueError(
            f"requested {samples} samples but only {len(eligible)} non-empty rows are eligible"
        )
    return random.Random(seed).sample(eligible, samples)
