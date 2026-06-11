from pathlib import Path

import pytest

from preprocess import txt_parser


def test_parse_strips_bom_and_normalizes_crlf(tmp_path):
    p = tmp_path / "doc.txt"
    # BOM + CRLF + a trailing-whitespace line + trailing newline.
    p.write_bytes("﻿第一条 测试 \r\n第二条 内容\r\n".encode("utf-8"))

    text = txt_parser.parse(p)

    assert text == "第一条 测试\n第二条 内容"
    assert "﻿" not in text  # BOM gone
    assert "\r" not in text  # CRLF normalized


def test_parse_normalizes_lone_cr_line_endings(tmp_path):
    p = tmp_path / "old_mac.txt"
    p.write_bytes("第一条\r第二条\r第三条".encode("utf-8"))

    text = txt_parser.parse(p)

    assert text == "第一条\n第二条\n第三条"


def test_parse_trims_file_edges_and_line_trailing_spaces(tmp_path):
    p = tmp_path / "spaces.txt"
    p.write_text("  标题  \n第一条 内容   \n第二条 内容\t\n\n", encoding="utf-8")

    text = txt_parser.parse(p)

    assert text == "标题\n第一条 内容\n第二条 内容"


def test_parse_preserves_internal_blank_lines(tmp_path):
    p = tmp_path / "paragraphs.txt"
    p.write_text("第一章\n\n第一条 内容\n\n第二条 内容", encoding="utf-8")

    text = txt_parser.parse(p)

    assert text == "第一章\n\n第一条 内容\n\n第二条 内容"


def test_parse_missing_file_raises_file_not_found(tmp_path):
    p = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        txt_parser.parse(p)


def test_parse_real_committed_txt():
    # examples/data is version-controlled (unlike the gitignored data/raw).
    p = Path("examples/data/raw/demo_insurance.txt")
    if not p.exists():
        return
    text = txt_parser.parse(p)
    assert text
    assert "\r" not in text
    assert not text.startswith("﻿")
