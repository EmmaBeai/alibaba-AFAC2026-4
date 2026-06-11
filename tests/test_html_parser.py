from pathlib import Path

from preprocess import html_parser


def _write(tmp_path, name, html):
    p = tmp_path / name
    p.write_bytes(html.encode("utf-8"))
    return p


def test_reconstructs_paragraph_from_fragmented_spans(tmp_path):
    # Mirrors the real pages: each char/number in its own span, body in detail-news.
    html = (
        "<html><body>"
        '<div class="detail-news">'
        "<p><span>〔</span><span>2024</span><span>〕</span><span>112</span><span>号</span></p>"
        "<p>当事人:某公司。</p>"
        "</div>"
        "</body></html>"
    )
    text = html_parser.parse(_write(tmp_path, "doc.html", html))
    assert text == "〔2024〕112号\n当事人:某公司。"


def test_excludes_site_chrome_outside_content(tmp_path):
    html = (
        "<html><body>"
        "<ul><li><a>首页</a></li><li><a>机构概况</a></li></ul>"
        '<div class="detail-news"><p>正文内容。</p></div>'
        '<div class="footer"><p>版权所有</p></div>'
        "</body></html>"
    )
    text = html_parser.parse(_write(tmp_path, "doc.html", html))
    assert text == "正文内容。"
    assert "首页" not in text and "版权所有" not in text


def test_strips_scripts_and_styles(tmp_path):
    html = (
        '<html><body><div class="content">'
        "<script>var x=1;</script><style>.a{}</style>"
        "<p>有效正文。</p>"
        "</div></body></html>"
    )
    text = html_parser.parse(_write(tmp_path, "doc.html", html))
    assert text == "有效正文。"
    assert "var x" not in text


def test_falls_back_to_body_when_no_content_div(tmp_path):
    html = "<html><body><p>孤立段落</p></body></html>"
    text = html_parser.parse(_write(tmp_path, "doc.html", html))
    assert text == "孤立段落"


def test_extract_title_prefers_article_title_meta(tmp_path):
    html = (
        "<html><head>"
        '<meta name="ArticleTitle" content="中国证监会市场禁入决定书（朱要文）"/>'
        "<title>中国证监会市场禁入决定书_中国证券监督管理委员会</title>"
        "</head><body><p>正文</p></body></html>"
    )
    assert html_parser.extract_title(_write(tmp_path, "d.html", html)) == "中国证监会市场禁入决定书（朱要文）"


def test_extract_title_falls_back_to_title_tag_without_site_suffix(tmp_path):
    html = "<html><head><title>某决定书_中国证券监督管理委员会</title></head><body></body></html>"
    assert html_parser.extract_title(_write(tmp_path, "d.html", html)) == "某决定书"


def test_extract_title_returns_none_when_absent(tmp_path):
    html = "<html><head></head><body><p>正文</p></body></html>"
    assert html_parser.extract_title(_write(tmp_path, "d.html", html)) is None


def test_real_csrc_html_if_present():
    # data/raw is gitignored; skip gracefully when the dataset isn't built.
    files = sorted(Path("data/raw").glob("csrc_*.html"))
    if not files:
        return
    text = html_parser.parse(files[0])
    assert text.startswith("〔")  # body starts with the document number
    assert "首页" not in text  # nav chrome excluded
    assert "【关闭窗口】" not in text  # trailing UI excluded
    title = html_parser.extract_title(files[0])
    assert title and "决定书" in title  # real title harvested from meta
