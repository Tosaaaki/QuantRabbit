import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.gpt_decider import (
    _extract_json_object,
    _get_responses_output_text,
    _normalize_json_content,
    _load_json_payload,
)


def test_normalize_json_content_strips_fence():
    raw = """```json
    {"focus_tag":"micro"}
    ```"""
    assert _normalize_json_content(raw) == '{"focus_tag":"micro"}'


def test_extract_json_object_from_wrapped_text():
    text = "prefix {\"foo\": 1, \"bar\": [2,3]} suffix"
    extracted = _extract_json_object(text)
    assert extracted == '{"foo": 1, "bar": [2,3]}'


def test_load_json_payload_falls_back_to_embedded_object():
    text = "code block ```\n{\"a\": 1}\n``` trailing"
    data = _load_json_payload(text)
    assert data == {"a": 1}


class _DummyBlock:
    def __init__(self, text):
        self.text = text


class _DummyItem:
    def __init__(self, text=None, blocks=None):
        self.text = text
        self.content = blocks or []


class _DummyResponse:
    def __init__(self, output_text=None, items=None):
        self.output_text = output_text
        self.output = items or []


@pytest.mark.parametrize(
    "resp,expected",
    [
        (_DummyResponse(output_text="  {\"a\":1}  "), '{"a":1}'),
        (
            _DummyResponse(
                items=[_DummyItem(text=" {\"b\":2} ")]
            ),
            '{"b":2}',
        ),
        (
            _DummyResponse(
                items=[_DummyItem(blocks=[_DummyBlock(text=" {\"c\":3} ")])]
            ),
            '{"c":3}',
        ),
    ],
)
def test_get_responses_output_text(resp, expected):
    assert _get_responses_output_text(resp, "dummy-model") == expected


def test_get_responses_output_text_empty_returns_blank():
    resp = _DummyResponse()
    assert _get_responses_output_text(resp, "dummy-model") == ""
