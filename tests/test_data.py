import pytest

from subspace_gguf.data import render_row


def test_render_text_field_with_system_prompt():
    messages = render_row(
        {"text": "Hello"},
        text_field="text",
        messages_field=None,
        prompt_template=None,
        system_prompt="Be concise.",
    )
    assert messages == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]


def test_render_prompt_template():
    messages = render_row(
        {"question": "2+2?"},
        text_field=None,
        messages_field=None,
        prompt_template="Question: {question}",
        system_prompt=None,
    )
    assert messages[0]["content"] == "Question: 2+2?"


def test_exactly_one_format_is_required():
    with pytest.raises(ValueError):
        render_row(
            {"text": "x"},
            text_field="text",
            messages_field=None,
            prompt_template="{text}",
            system_prompt=None,
        )
