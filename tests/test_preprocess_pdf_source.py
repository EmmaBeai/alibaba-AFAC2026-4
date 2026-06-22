from agent.preprocess import score_pdf_source, select_pdf_source


def test_healthy_pypdf_is_preferred_when_close_to_best() -> None:
    pypdf_text = "营业收入 净利润 现金流 " * 200 + "中文正文" * 2000
    glm_text = "营业收入 净利润 现金流 " * 230 + "中文正文" * 2200
    candidates = [
        ("glm-ocr", glm_text, score_pdf_source("glm-ocr", glm_text)),
        ("pypdf", pypdf_text, score_pdf_source("pypdf", pypdf_text)),
    ]

    selected_model, _, quality = select_pdf_source(candidates)

    assert selected_model == "pypdf"
    assert quality.acceptable


def test_garbled_pypdf_is_not_acceptable() -> None:
    bad_text = "招商銀行股份有限公司\n" + "ඳ གิൕ ଢ\u0a63 \u0d64ၬ " * 1000
    quality = score_pdf_source("pypdf", bad_text)

    assert not quality.acceptable
    assert quality.suspicious_ratio > 0.03


def test_ocr_wins_when_pypdf_is_garbled() -> None:
    bad_pypdf = "招商銀行股份有限公司\n" + "ඳ གิൕ ଢ\u0a63 \u0d64ၬ " * 1000
    good_ocr = "招商银行 营业收入 净利润 资本充足率 不良贷款 " * 500
    candidates = [
        ("glm-ocr", good_ocr, score_pdf_source("glm-ocr", good_ocr)),
        ("pypdf", bad_pypdf, score_pdf_source("pypdf", bad_pypdf)),
    ]

    selected_model, _, quality = select_pdf_source(candidates)

    assert selected_model == "glm-ocr"
    assert quality.acceptable
