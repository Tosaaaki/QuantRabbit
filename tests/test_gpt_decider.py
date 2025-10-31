import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.gpt_decider import (  # noqa: E402
    _extract_json_object,
    _load_json_payload,
    _normalize_json_content,
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


@pytest.mark.parametrize(
    "payload",
    [
        '{"focus_tag":"micro"}',
        'prefix {"foo": 1} suffix',
        '{"foo": 1, "bar": [2,3]}',
    ],
)
def test_load_json_payload_various(payload):
    result = _load_json_payload(payload)
    assert isinstance(result, dict)
