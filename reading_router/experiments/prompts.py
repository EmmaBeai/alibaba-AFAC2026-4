from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


READING_SYSTEM_PROMPT = "你是题干阅读助手，负责把题干读成自然的 reading_chain。"
COMPACT_READING_SYSTEM_PROMPT = "题干阅读助手。只输出JSON。"


QUESTION_FIELDS = ("qid", "domain", "split", "question", "answer_format", "type", "doc_ids")
REASONING_MD_PROMPT_PREFIX = """六步读题,只用题干,不答题。输入i为qid。
JSON数组:[{"qid":"...","reading_chain":["1任务...","2对象...","3主题...","4变量...","5关系/口径...","6易错..."]}]
规则:
1写清题干要求判断什么事实、关系或适用情况;
2写清涉及的文档、主体、产品、公司、法规、报告对象或场景;
3写清真正要核对的责任、条款、指标、事实、义务或适用范围;
4摘出题干里的关键条件、时间、数字、术语和限定词;
5写清触发条件、适用范围、归属关系、比较关系或证据边界;
6写一个最容易误读的核对点,尤其注意不要把一个对象的信息套到另一个对象上。
禁补题外事实。
Q="""


def build_reading_prompt(questions: Sequence[Mapping[str, Any]]) -> str:
    payload = [_slim_question(question) for question in questions]
    return (
        "请阅读下面这批题干，为每道题写自然的 reading_chain。\n\n"
        "reading_chain 是 thinking chain。每个元素都直接写“我现在在想什么 + 题干里的具体对象/条件/口径”，"
        "不要只写抽象标签。\n\n"
        "按自然读题顺序写：先判断任务，再锁定对象/文档/主体/年度，再指出主题、变量、条件、比较关系和口径，"
        "最后说明后续核对路径和易错点。\n\n"
        "题型提示：多选题提醒逐项核对；单选题提醒唯一表述需完全准确；判断题拆分复合条件；"
        "计算/排序题先识别变量、公式、免赔额、比例或排序规则。\n\n"
        "reading_chain 的句式参考：\n"
        "- 我先判断题目任务：这题是在要求……，所以要先读清任务边界和判断口径。\n"
        "- 我再锁定对象和证据边界：题干涉及……，需要把这些文档/主体/年份分开看。\n"
        "- 我接着识别主题维度：真正要核对的是……，不是泛泛比较全部内容。\n"
        "- 我再抽出条件和变量：题干里的……会影响后续判断，尤其要注意……。\n"
        "- 我然后看关系和口径：这里是……关系，数字/术语/时间口径不能混用。\n"
        "- 我最后预判核对路径和易错点：后续应分别确认……，最容易错在……。\n\n"
        "输出内容为一个 JSON object，格式如下：\n"
        '{"items":[{"qid":"...","reading_chain":["..."],"reading_summary":"...","notes":"..."}]}\n\n'
        "题干输入：\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_compact_reading_prompt(questions: Sequence[Mapping[str, Any]]) -> str:
    payload = [_compact_question(question) for question in questions]
    return (
        "六步读题,只用题干,不答题。输入i为qid。\n"
        'JSON数组:[{"qid":"...","reading_chain":["1任务...","2对象...","3主题...",'
        '"4变量...","5关系...","6易错..."]}]\n'
        "规则:1-4短;4抄数字/时间/限定词;5写关系型:"
        "跨文档/跨公司/跨年/总分/且或/排序/计算/存在;6一个易错点;禁补题外事实。\n"
        f"Q={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def build_compact_rel_reading_prompt(questions: Sequence[Mapping[str, Any]]) -> str:
    payload = [_compact_question(question) for question in questions]
    return (
        "六步读题,只用题干,不答题。输入i为qid。\n"
        'JSON数组:[{"qid":"...","reading_chain":["1任务...","2对象...","3主题...",'
        '"4变量...","5关系/口径...","6易错..."]}]\n'
        "规则:1-3短且具体;4抄题干关键数字/时间/主体/限定词;"
        "5必须写清公式/分母/阈值/取大取小/扣减/排序方向/且或关系,"
        "不能只写计算/比较/跨文档;6写一个最可能错的核对点;禁补题外事实。\n"
        f"Q={json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def build_reasoning_md_reading_prompt(questions: Sequence[Mapping[str, Any]]) -> str:
    payload = [_stem_question(question) for question in questions]
    return REASONING_MD_PROMPT_PREFIX + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _slim_question(question: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in QUESTION_FIELDS:
        if field in question:
            output[field] = question[field]
    return output


def _compact_question(question: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "i": question.get("qid", ""),
        "q": question.get("question", ""),
    }
    if question.get("answer_format"):
        output["f"] = question["answer_format"]
    if question.get("type"):
        output["t"] = question["type"]
    if question.get("doc_ids"):
        output["d"] = question["doc_ids"]
    return output


def _stem_question(question: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "i": question.get("qid", ""),
        "q": question.get("question", ""),
    }
