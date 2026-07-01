# Reading Router Final Spec

This file records the current retained routing design and prompt set. The
pipeline-facing single-file implementation is `reading_router/final_prompts.py`;
exploratory scripts live under `reading_router/experiments/`.

## Retained Prompts

Current retained prompt families:

1. `numeric_delta`: retained number-sensitive prompt builder, not used by the default pipeline route.
2. `tf_reading`: true/false statement reading.
3. `multi_route_v2`: multi-choice option evidence routing.
4. `mcq_route`: single-choice evidence routing.
5. `general_reasoning`: fixed fallback/baseline reading chain.

All Qwen router experiments should pass `enable_thinking=false` unless a run is
explicitly testing thinking on/off. Do not set completion-token caps for quality
comparison runs.

## Routing Order

The default pipeline route is schema-based. A question should normally take
exactly one primary route:

```text
1. true/false by answer_format
2. multi-choice by answer_format
3. single-choice by answer_format
4. fallback general reasoning
```

Pseudo-code:

```python
if answer_format == "tf":
    return "tf_reading"

if answer_format == "multi":
    return "multi_route_v2"

if answer_format == "mcq":
    return "mcq_route"

return "general_reasoning"
```

`answer_format` is the trusted schema signal for `tf`, `multi`, and `mcq`.
No qid list or numeric/scalar override is part of the default pipeline route.

## Numeric/Scalar Definition

`numeric_delta` remains available as a prompt builder for manual experiments,
but the default pipeline no longer routes questions into it.

`calc_numeric`:

- Calculation, ordering, amount comparison, total, compensation, deductible,
  surrender value, formula, ratio, threshold, multiplier, or subtraction.
- Examples: insurance payment amount, surrender ranking, death-benefit ranking,
  cash-flow ratio, PE multiple range.

`scalar_fact`:

- Options contain scalar facts where the value/unit/year/threshold is itself the
  decision point.
- Examples: `6.63%`, `180亿元`, `1000条`, `8894.3亿元`, `200%`, `AAA` when paired
  with an issuer/document target.

Do not route to `numeric_delta` when numbers are incidental scene conditions and
the real decision is a rule/exception.

`incidental_number` examples:

- Waiting-period scenario days such as `第50天`, `第20天`, `第15天`, `第10天`,
  where the decision is "accident vs non-accident during waiting period."
- Calendar years that only identify a report year, unless the option is asking
  about a numeric metric in that year.

## Prompt: numeric_delta

Manual/experiment use for `calc_numeric` and `scalar_fact`.

Input fields:

```json
[{"qid":"...","q":"...","options":["A ...","B ..."]}]
```

Prompt:

```text
你是numeric evidence router。输入为qid、q、options。
只读题干和选项，不答题。把选项差异压成后续找证据用的三槽。

输出JSON数组:
[{"qid":"...","delta":"...","need":"...","risk":"..."}]

字段:
delta: 选项分歧或TF待验点；写出候选值/结论，如金额、排序、可赔/不赔、适用/不适用、正误条件；不要只概括题目任务。
need: 裁决分歧所需证据槽；写具体条款、公式、字段、免赔额、比例、时间、主体、补偿/扣减/叠加规则。
risk: 最大误判点；写清错因，如规则混用、扣减顺序、对象归属、单位/日期、只验证部分条件。
要求: 短句;不复述题干;不逐项解释;不答题;禁补题外事实。
Q=...
```

Output shape:

```json
[{"qid":"...","delta":"...","need":"...","risk":"..."}]
```

## Prompt: tf_reading

Use for non-numeric `answer_format == "tf"`.

Input fields:

```json
[{"qid":"...","q":"..."}]
```

Prompt:

```text
你是判断题读题助手。输入为qid、q。
只读判断题题干，不答题。每题写3句reading_chain:
1先自然说明“这是判断题，陈述说……”，保留题干里的主体、时间、数字、术语和限定词;
2再拆出需要分别核对的条件、文档归属、比较关系、存在/未出现、且/或关系;
3最后写最容易误读的点，如只验一半、把一份文档套到另一份、相似术语当原文、口径/单位/时间混淆。
JSON数组:[{"qid":"...","reading_chain":["...","...","..."]}]
禁补题外事实，不写正确/错误。
Q=...
```

Output shape:

```json
[{"qid":"...","reading_chain":["...","...","..."]}]
```

## Prompt: multi_route_v2

Use for non-numeric `answer_format == "multi"`.

Input fields:

```json
[{"qid":"...","q":"...","options":["A ...","B ...","C ...","D ..."]}]
```

Prompt:

```text
你是multi-choice evidence router。输入qid、q、options。
不答题，不判断真假。只把每个选项压成后续找证据用的route。

route=最少证据定位短语，需含对象锚点+待核对事实+关键限定词。
对象锚点包括主体、公司、产品、文档、法规、报告、年份等。
关键限定词包括数字、比例、期限、单位、比较方向、存在/未出现、适用条件等。
必须原样保留选项里的比较方向、否定和全称限定词，如高于/低于/超过/不足/未/不/均/所有/至少；不要改写成泛泛的“对比/情况”。
route写短语，不写完整解释，不补题外事实，不删除选项中的关键数字或术语。

risk写本题共同易混点，一句话即可。
输出JSON数组:
[{"qid":"...","routes":["A ...","B ...","C ...","D ..."],"risk":"..."}]

Q=...
```

Output shape:

```json
[{"qid":"...","routes":["A ...","B ...","C ...","D ..."],"risk":"..."}]
```

## Prompt: mcq_route

Use for non-numeric `answer_format == "mcq"`.

Input fields:

```json
[{"qid":"...","q":"...","options":["A ...","B ...","C ...","D ..."]}]
```

Prompt:

```text
你是single-choice evidence router。输入qid、q、options。
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
Q=...
```

Output shape:

```json
[{"qid":"...","decision":"...","routes":["A ...","B ...","C ...","D ..."],"risk":"..."}]
```

## Prompt: general_reasoning

Use only as fallback or fixed baseline.

Input fields:

```json
[{"i":"...","q":"..."}]
```

Prompt:

```text
六步读题,只用题干,不答题。输入i为qid。
JSON数组:[{"qid":"...","reading_chain":["1任务...","2对象...","3主题...","4变量...","5关系/口径...","6易错..."]}]
规则:
1写清题干要求判断什么事实、关系或适用情况;
2写清涉及的文档、主体、产品、公司、法规、报告对象或场景;
3写清真正要核对的责任、条款、指标、事实、义务或适用范围;
4摘出题干里的关键条件、时间、数字、术语和限定词;
5写清触发条件、适用范围、归属关系、比较关系或证据边界;
6写一个最容易误读的核对点,尤其注意不要把一个对象的信息套到另一个对象上。
禁补题外事实。
Q=...
```

Output shape:

```json
[{"qid":"...","reading_chain":["1任务...","2对象...","3主题...","4变量...","5关系/口径...","6易错..."]}]
```

## Current Experiment Takeaways

- `numeric_delta` is strongest for insurance calculations, rankings, deductibles,
  compensation totals, and scalar fact checks.
- `multi_route_v2` fixed the main no-thinking failure mode by preserving
  comparison direction, negation, and universal qualifiers.
- `mcq_route` is useful after numeric-first routing; it captures the unique
  decision axis for the small non-numeric single-choice tail.
- `tf_reading` is more routing-useful than the fixed six-step baseline for
  true/false statements because it explicitly splits a compound statement.
- `general_reasoning` remains useful as a stable baseline and fallback, but
  should not be the primary route when a more specific prompt applies.

## Experiment Code

Exploratory scripts and prior prompt variants live in:

```text
reading_router/experiments/
```

Key modules:

```text
reading_router.experiments.numeric_prompt
reading_router.experiments.tf_prompt
reading_router.experiments.multi_prompt
reading_router.experiments.mcq_prompt
reading_router.experiments.experiment
```
