from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


NUMERIC_DELTA = "numeric_delta"
TF_READING = "tf_reading"
MULTI_ROUTE_V2 = "multi_route_v2"
MCQ_ROUTE = "mcq_route"
GENERAL_REASONING = "general_reasoning"

RETAINED_ROUTES = (
    NUMERIC_DELTA,
    TF_READING,
    MULTI_ROUTE_V2,
    MCQ_ROUTE,
    GENERAL_REASONING,
)

NUMERIC_DELTA_PROMPT_PREFIX = """你是numeric evidence router。输入为qid、q、options。
只读题干和选项，不答题。把选项差异压成后续找证据用的三槽。

输出JSON数组:
[{"qid":"...","delta":"...","need":"...","risk":"..."}]

字段:
delta: 选项分歧或TF待验点；写出候选值/结论，如金额、排序、可赔/不赔、适用/不适用、正误条件；不要只概括题目任务。
need: 裁决分歧所需证据槽；写具体条款、公式、字段、免赔额、比例、时间、主体、补偿/扣减/叠加规则。
risk: 最大误判点；写清错因，如规则混用、扣减顺序、对象归属、单位/日期、只验证部分条件。
要求: 短句;不复述题干;不逐项解释;不答题;禁补题外事实。
Q="""

TF_READING_PROMPT_PREFIX = """你是判断题读题助手。输入为qid、q。
只读判断题题干，不答题。每题写3句reading_chain:
1先自然说明“这是判断题，陈述说……”，保留题干里的主体、时间、数字、术语和限定词;
2再拆出需要分别核对的条件、文档归属、比较关系、存在/未出现、且/或关系;
3最后写最容易误读的点，如只验一半、把一份文档套到另一份、相似术语当原文、口径/单位/时间混淆。
JSON数组:[{"qid":"...","reading_chain":["...","...","..."]}]
禁补题外事实，不写正确/错误。
Q="""

MULTI_ROUTE_V2_PROMPT_PREFIX = """你是multi-choice evidence router。输入qid、q、options。
不答题，不判断真假。只把每个选项压成后续找证据用的route。

route=最少证据定位短语，需含对象锚点+待核对事实+关键限定词。
对象锚点包括主体、公司、产品、文档、法规、报告、年份等。
关键限定词包括数字、比例、期限、单位、比较方向、存在/未出现、适用条件等。
必须原样保留选项里的比较方向、否定和全称限定词，如高于/低于/超过/不足/未/不/均/所有/至少；不要改写成泛泛的“对比/情况”。
route写短语，不写完整解释，不补题外事实，不删除选项中的关键数字或术语。

risk写本题共同易混点，一句话即可。
输出JSON数组:
[{"qid":"...","routes":["A ...","B ...","C ...","D ..."],"risk":"..."}]

Q="""

MCQ_ROUTE_PROMPT_PREFIX = """你是single-choice evidence router。输入qid、q、options。
只做证据路由，不答题，不判断A/B/C/D。目标是为“唯一正确项”找到最少裁决证据。

输出JSON数组:
[{"qid":"...","decision":"...","routes":["A ...","B ...","C ...","D ..."],"risk":"..."}]

字段:
decision: 本题唯一正确项的裁决轴；如果四个选项是不同事实，写“分别核对四项事实，唯一完全准确项为准”并点出口径。
routes: 每个选项一个最少证据定位短语，含对象锚点+待核对事实+关键限定词。
risk: 最容易误选的一点，如部分正确当全对、跨文档套用、期限/单位/金额/比例/例外条件混淆。

要求:
route写短语，不写完整解释。
必须原样保留选项里的数字、比例、期限、比较方向、否定和全称限定词，如高于/低于/超过/不足/未/不/均/所有/至少。
不要把具体裁决点改写成泛泛的“情况/对比/规定”。
禁补题外事实。
Q="""

GENERAL_REASONING_PROMPT_PREFIX = """六步读题,只用题干,不答题。输入i为qid。
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


def route_question(question: Mapping[str, Any]) -> str:
    """Return the retained primary route for one question by pipeline schema."""
    answer_format = str(question.get("answer_format", "")).lower()
    if answer_format == "tf":
        return TF_READING
    if answer_format == "multi":
        return MULTI_ROUTE_V2
    if answer_format == "mcq":
        return MCQ_ROUTE
    return GENERAL_REASONING


def build_prompt(route: str, questions: Sequence[Mapping[str, Any]]) -> str:
    if route == NUMERIC_DELTA:
        return build_numeric_delta_prompt(questions)
    if route == TF_READING:
        return build_tf_reading_prompt(questions)
    if route == MULTI_ROUTE_V2:
        return build_multi_route_v2_prompt(questions)
    if route == MCQ_ROUTE:
        return build_mcq_route_prompt(questions)
    if route == GENERAL_REASONING:
        return build_general_reasoning_prompt(questions)
    raise ValueError(f"Unknown retained reading-router route: {route}")


def build_numeric_delta_prompt(questions: Sequence[Mapping[str, Any]]) -> str:
    return NUMERIC_DELTA_PROMPT_PREFIX + _dumps([_qid_q_options(question) for question in questions])


def build_tf_reading_prompt(questions: Sequence[Mapping[str, Any]]) -> str:
    return TF_READING_PROMPT_PREFIX + _dumps([_qid_q(question) for question in questions])


def build_multi_route_v2_prompt(questions: Sequence[Mapping[str, Any]]) -> str:
    return MULTI_ROUTE_V2_PROMPT_PREFIX + _dumps([_qid_q_options(question) for question in questions])


def build_mcq_route_prompt(questions: Sequence[Mapping[str, Any]]) -> str:
    return MCQ_ROUTE_PROMPT_PREFIX + _dumps([_qid_q_options(question) for question in questions])


def build_general_reasoning_prompt(questions: Sequence[Mapping[str, Any]]) -> str:
    return GENERAL_REASONING_PROMPT_PREFIX + _dumps(
        [{"i": question.get("qid", ""), "q": question.get("question", "")} for question in questions]
    )


def _qid_q(question: Mapping[str, Any]) -> dict[str, str]:
    return {
        "qid": str(question.get("qid", "")),
        "q": str(question.get("question", "")),
    }


def _qid_q_options(question: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "qid": str(question.get("qid", "")),
        "q": str(question.get("question", "")),
        "options": _compact_options(question.get("options")),
    }


def _compact_options(options: Any) -> list[str]:
    if isinstance(options, Mapping):
        return [f"{key} {value}" for key, value in options.items()]
    if isinstance(options, list):
        return [str(option) for option in options if str(option)]
    return []


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
