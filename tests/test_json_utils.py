"""Tests for JSON and answer post-processing."""


from app.utils.json_utils import clean_answer, extract_code_block, validate_json_string


def test_clean_answer_strips_fences() -> None:
    raw = "```python\ndef foo():\n    pass\n```"
    assert "def foo" in extract_code_block(raw)
    assert "```" not in extract_code_block(raw)


def test_clean_answer_trims_whitespace() -> None:
    assert clean_answer("  hello world  ") == "hello world"


def test_validate_json_string() -> None:
    valid, parsed = validate_json_string('[{"text": "Maria", "type": "person"}]')
    assert valid is True
    assert isinstance(parsed, list)


def test_validate_invalid_json() -> None:
    valid, parsed = validate_json_string("not json")
    assert valid is False
    assert parsed is None


def test_extract_code_block_without_fences() -> None:
    code = "def add(a, b):\n    return a + b"
    assert extract_code_block(code) == code
