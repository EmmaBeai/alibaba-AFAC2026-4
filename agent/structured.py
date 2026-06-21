from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.preprocess import clean_text

ARTICLE_RE = re.compile(r"(第[一二三四五六七八九十百千万零〇两\d]+条)")
HEADING_RE = re.compile(
    r"^(?:#{1,6}\s*)?(第[一二三四五六七八九十百千万零〇两\d]+[章节部分]|"
    r"[一二三四五六七八九十]+[、.][^\n]{2,80}|"
    r"\d+(?:\.\d+){0,4}[、.\s][^\n]{2,100})"
)
NUMBER_RE = re.compile(
    r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|％|亿元|万元|元|年|月|日|个工作日|日内|倍)?"
)

DOMAIN_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "regulatory": {
        "deadline": ["工作日", "日内", "期限", "报送", "报告"],
        "penalty": ["处罚", "罚款", "责令", "警告", "违法"],
        "obligation": ["应当", "不得", "必须", "义务", "职责"],
    },
    "insurance": {
        "death_benefit": ["身故保险金", "身故", "保险金"],
        "cash_value": ["现金价值", "账户价值", "退保"],
        "exclusion": ["责任免除", "不承担", "除外"],
        "payment": ["保费", "给付", "领取", "缴费"],
    },
    "financial_contracts": {
        "issue_terms": ["发行金额", "发行规模", "本期债券", "募集资金", "债券期限", "票面利率"],
        "rating": ["主体信用评级", "债项信用评级", "评级", "信用等级"],
        "intermediary": ["主承销商", "受托管理人", "簿记管理人", "评级机构"],
        "guarantee": ["担保", "增信", "抵押", "质押"],
        "put_redemption": ["回售", "赎回", "调整票面利率"],
        "default": ["违约", "加速清偿", "救济", "纠纷解决"],
        "holder_meeting": ["债券持有人会议", "持有人会议", "表决"],
        "financial_metric": ["资产负债率", "净利润", "现金流", "营业收入", "净资产"],
    },
    "financial_reports": {
        "income": ["营业收入", "净利润", "归属于上市公司股东"],
        "cashflow": ["经营活动产生的现金流量", "现金流量净额"],
        "rd": ["研发投入", "研发费用", "研发人员"],
        "dividend": ["现金分红", "利润分配", "股利"],
        "balance_sheet": ["总资产", "净资产", "资产负债率"],
    },
    "research": {
        "trend": ["趋势", "景气", "行业", "需求", "供给"],
        "forecast": ["预计", "预测", "展望", "目标价"],
        "company": ["公司", "龙头", "市场份额"],
        "risk": ["风险", "不及预期"],
        "conclusion": ["结论", "观点", "建议", "维持"],
    },
}


@dataclass(slots=True)
class StructuredUnit:
    unit_id: str
    doc_id: str
    domain: str
    title: str
    page: int
    section_path: str
    clause_no: str
    chunk_type: str
    raw_text: str
    search_text: str
    numbers: list[str]
    keywords: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_processed_document(document_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((document_dir / "metadata.json").read_text(encoding="utf-8"))
    pages = []
    with (document_dir / "pages.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            pages.append(json.loads(line))
    return metadata, pages


def build_structured_units(
    metadata: dict[str, Any],
    pages: list[dict[str, Any]],
    *,
    max_chars: int = 1400,
) -> list[StructuredUnit]:
    domain = metadata["domain"]
    doc_id = metadata["doc_id"]
    title = metadata.get("title") or doc_id
    units: list[StructuredUnit] = []
    section_stack: list[str] = []

    for page in pages:
        page_number = int(page["page_number"])
        text = clean_text(page.get("text") or "")
        section_stack = _update_section_stack(section_stack, text)
        section_path = " > ".join(section_stack[-4:])
        blocks = _domain_blocks(domain, text, max_chars=max_chars)
        for block_index, (raw_text, clause_no) in enumerate(blocks, start=1):
            raw_text = clean_text(raw_text)
            if len(raw_text) < 8:
                continue
            keywords = _keywords(domain, raw_text)
            chunk_type = _chunk_type(domain, raw_text, clause_no, keywords)
            numbers = _numbers(raw_text)
            unit_id = f"{doc_id}_p{page_number}_{block_index:03d}"
            search_text = "\n".join(
                part
                for part in [
                    f"文档标题：{title}",
                    f"章节路径：{section_path}" if section_path else "",
                    f"条款号：{clause_no}" if clause_no else "",
                    f"类型：{chunk_type}",
                    f"关键词：{' '.join(keywords)}" if keywords else "",
                    raw_text,
                ]
                if part
            )
            units.append(
                StructuredUnit(
                    unit_id=unit_id,
                    doc_id=doc_id,
                    domain=domain,
                    title=title,
                    page=page_number,
                    section_path=section_path,
                    clause_no=clause_no,
                    chunk_type=chunk_type,
                    raw_text=raw_text,
                    search_text=search_text,
                    numbers=numbers,
                    keywords=keywords,
                )
            )
    return units


def iter_processed_document_dirs(processed_root: Path) -> Iterable[Path]:
    for domain_dir in sorted(path for path in processed_root.iterdir() if path.is_dir()):
        if domain_dir.name == "pdf_parsed":
            continue
        for document_dir in sorted(path for path in domain_dir.iterdir() if path.is_dir()):
            if (document_dir / "metadata.json").exists() and (document_dir / "pages.jsonl").exists():
                yield document_dir


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def _domain_blocks(domain: str, text: str, *, max_chars: int) -> list[tuple[str, str]]:
    if domain == "regulatory":
        clause_blocks = _split_regulatory_articles(text)
        if clause_blocks:
            return clause_blocks
    return [(block, "") for block in _split_by_paragraphs(text, max_chars=max_chars)]


def _split_regulatory_articles(text: str) -> list[tuple[str, str]]:
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return []
    blocks: list[tuple[str, str]] = []
    prefix = text[: matches[0].start()].strip()
    if prefix and len(prefix) > 40:
        blocks.append((prefix, ""))
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append((text[start:end].strip(), match.group(1)))
    return blocks


def _split_by_paragraphs(text: str, *, max_chars: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    blocks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        if current and size + len(paragraph) > max_chars:
            blocks.append("\n\n".join(current))
            current = []
            size = 0
        if len(paragraph) > max_chars:
            blocks.extend(_hard_split(paragraph, max_chars))
            continue
        current.append(paragraph)
        size += len(paragraph)
    if current:
        blocks.append("\n\n".join(current))
    return blocks


def _hard_split(text: str, max_chars: int) -> list[str]:
    return [text[start:start + max_chars] for start in range(0, len(text), max_chars)]


def _update_section_stack(section_stack: list[str], text: str) -> list[str]:
    stack = list(section_stack)
    for line in text.splitlines()[:80]:
        stripped = line.strip()
        if len(stripped) > 120:
            continue
        match = HEADING_RE.match(stripped)
        if not match:
            continue
        heading = stripped.lstrip("#").strip()
        if not heading:
            continue
        if heading.startswith("第") and "章" in heading:
            stack = [heading]
        elif heading.startswith("第") and ("节" in heading or "部分" in heading):
            stack = stack[:1] + [heading]
        else:
            stack = stack[:3] + [heading]
    return stack[-4:]


def _chunk_type(domain: str, text: str, clause_no: str, keywords: list[str]) -> str:
    if domain == "regulatory" and clause_no:
        return "clause"
    if "|" in text and domain in {"financial_reports", "financial_contracts"}:
        return "table"
    if keywords:
        return keywords[0]
    return {
        "regulatory": "regulatory_text",
        "insurance": "insurance_clause",
        "financial_contracts": "contract_section",
        "financial_reports": "report_section",
        "research": "research_section",
    }.get(domain, "text")


def _keywords(domain: str, text: str) -> list[str]:
    found: list[str] = []
    for chunk_type, words in DOMAIN_KEYWORDS.get(domain, {}).items():
        if any(word in text for word in words):
            found.append(chunk_type)
    return found


def _numbers(text: str) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for match in NUMBER_RE.finditer(text):
        value = match.group(0).strip()
        if value and value not in seen:
            seen.add(value)
            values.append(value)
        if len(values) >= 20:
            break
    return values
