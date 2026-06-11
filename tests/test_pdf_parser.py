import pytest

from preprocess import pdf_parser


def _write_md(parsed_dir, model, doc_id, text):
    p = parsed_dir / model / f"{doc_id}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_parse_prefers_default_first_model(tmp_path):
    _write_md(tmp_path, "mineru2.5-pro", "1", "# 标题\nmineru body")
    _write_md(tmp_path, "paddleocr-vl-1.6", "1", "paddle body")
    text = pdf_parser.parse("data/raw/1.pdf", parsed_dir=tmp_path)
    assert "mineru body" in text


def test_parse_falls_through_when_preferred_missing(tmp_path):
    # No mineru output for this doc; paddleocr-vl is next in DEFAULT_MODEL_ORDER.
    _write_md(tmp_path, "paddleocr-vl-1.6", "text01", "paddle only")
    text = pdf_parser.parse("data/raw/text01.pdf", parsed_dir=tmp_path)
    assert text == "paddle only"


def test_parse_skips_empty_output(tmp_path):
    _write_md(tmp_path, "mineru2.5-pro", "9", "   \n  ")  # empty after strip
    _write_md(tmp_path, "paddleocr-vl-1.6", "9", "real content")
    text = pdf_parser.parse("9.pdf", parsed_dir=tmp_path)
    assert text == "real content"


def test_parse_honors_explicit_model(tmp_path):
    _write_md(tmp_path, "mineru2.5-pro", "1", "mineru")
    _write_md(tmp_path, "paddleocr-vl-1.6", "1", "paddle")
    assert pdf_parser.parse("1.pdf", model="paddleocr-vl-1.6", parsed_dir=tmp_path) == "paddle"


def test_parse_raises_when_nothing_available(tmp_path):
    with pytest.raises(FileNotFoundError):
        pdf_parser.parse("missing.pdf", parsed_dir=tmp_path)


def test_available_models_lists_non_empty_in_order(tmp_path):
    _write_md(tmp_path, "mineru2.5-pro", "1", "a")
    _write_md(tmp_path, "paddleocr-vl-1.6", "1", "b")
    assert pdf_parser.available_models("1.pdf", parsed_dir=tmp_path) == ["mineru2.5-pro", "paddleocr-vl-1.6"]


def test_extract_title_from_first_heading(tmp_path):
    _write_md(tmp_path, "mineru2.5-pro", "rep", "前言\n\n## 比亚迪 2024 年年度报告\n正文")
    assert pdf_parser.extract_title("rep.pdf", parsed_dir=tmp_path) == "比亚迪 2024 年年度报告"


def test_extract_title_none_when_no_markdown(tmp_path):
    assert pdf_parser.extract_title("nope.pdf", parsed_dir=tmp_path) is None
