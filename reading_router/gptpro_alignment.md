# GPTPro Alignment Notes

## Short Answer

当前 `model_outputs.jsonl` 和 GPTPro 的前置思考是 **方向匹配，但阶段不完全匹配**。

它已经覆盖 GPTPro 解题链里的前半段：

1. 读清题目任务。
2. 拆出主体、产品、文档、指标、条款、时间、金额、比较关系。
3. 对选择题，把选项压成后续裁决用的证据路线。
4. 提醒最容易误判的口径或归属关系。

但它还没有进入 GPTPro 后半段：

1. 真实定位证据。
2. 摘 source / quote。
3. 判断每个选项 supported / contradicted / not_found。
4. 计算或比对。
5. 给 `gold_answer`。
6. 写完整 `solve_trace` 和 `reasoning_summary`。

所以它不是 fake gold，也不是 answerer。它更像是 **evidence router 的输入层**。

## What GPTPro Does

以 `fake_gold_gptpro.json` 里的样例看，GPTPro 输出的是完整解题结果：

```text
qid
answer_format
gold_answer
confidence
score_eligible
evidence
solve_trace
reasoning_summary
caveat
```

其中关键链路大概是：

```text
read_question
locate_evidence
compute / compare / check_condition
check_option_A/B/C/D
derive_answer
final_check
```

`read_question` 是读题，`locate_evidence` 开始进入证据，`check_option_*` 才是逐项裁决。

## What We Produce Now

当前轻量文件是：

```text
runs/reading_router_final/qwen37max_no_thinking_pipeline_all_20260701/model_outputs.jsonl
```

它按题型保留模型原始结构：

```text
mcq:   qid, decision, routes, risk
multi: qid, routes, risk
tf:    qid, reading_chain
```

对应到 GPTPro：

```text
decision / reading_chain
  ≈ GPTPro solve_trace.read_question

routes / risk
  ≈ GPTPro locate_evidence 之前的证据意图压缩

不包含：
  evidence.source
  evidence.quote
  evidence.verdict
  check_option_*
  derive_answer
```

## Match Assessment

### Matched

`mcq_route` 和 GPTPro 的前置步骤比较贴近。

它会先写一个 `decision`，说明这题最终裁决轴是什么；然后把 A/B/C/D 压成 option-level routes。这和 GPTPro 后面 `check_option_A/B/C/D` 的输入形状是对得上的。

例如 `ins_a_001`：

```text
decision: 核对四款产品身故保险金公式和数值代入结果，唯一排序准确项为准
routes: A/B/C/D 四个候选排序
risk: 已领养老年金扣减、账户价值/基本保额取值错误
```

这正好是 GPTPro 后续计算四个产品金额、比较排序、逐项裁决的前置材料。

`multi_route_v2` 也基本匹配。

它把每个选项变成独立证据路线，适合作为后续检索或证据定位的 query/probe。比如 `ins_a_005` 把四个产品的免责情形分别拆成：

```text
A 平安智盈金生 酒后驾驶 失能失智护理状态 免责
B 平安安佑福重疾险 感染艾滋病病毒 非输血非职业非器官移植 重大疾病 免责
C 平安预防接种意外险 不具有接种条件的单位 接种疫苗 异常反应 免责
D 众安食责险 食品超过保质期 消费者食物中毒 免责
```

这和 GPTPro 逐项找免责条款的路径一致。

`tf_reading` 匹配 GPTPro 的 `read_question`，但不如选择题 route 直接。

它能拆出判断题中的复合条件、且/或关系、口径和易错点。对于后续 evidence 层，它够用作题干理解，但还没有显式拆成多个 probe。

### Not Matched

当前输出没有 evidence 层。

GPTPro 的关键强度在于：

```text
source: 具体文档和行号
quote: 原文摘录
verdict: supported / contradicted / not_found
explanation: 为什么这个选项成立或不成立
```

我们现在完全没有这些。

当前输出也没有 answer 层。

它不应该给 `gold_answer`，也没有资格给，因为它还没有读证据。

当前输出不是完整 `solve_trace`。

它只像 `solve_trace` 的前一两步，而不是完整链。尤其缺少：

```text
locate_evidence
compute_*
check_option_*
derive_answer
final_check
```

## Stage Naming

可以把现在的 pipeline 定义成 Stage 1：

```text
Stage 0: schema route
  answer_format -> tf / multi / mcq / general

Stage 1: question reading and evidence routing
  input: qid, question, options
  output: model_outputs.jsonl

Stage 2: evidence retrieval / locating
  input: qid + routes / decision / reading_chain
  output: candidate source spans

Stage 3: option adjudication
  input: question + options + evidence spans
  output: verdict per option / statement

Stage 4: answer derivation
  input: verdicts
  output: answer + trace
```

现在我们完成的是 Stage 1。

## Important Implication

如果后续 retrieval 直接吃原始 question，`model_outputs` 的价值不大。

它真正有用的方式是：后续不要把整题丢给 embedding，而是消费结构化 route：

```text
multi/mcq:
  每个 option route 单独检索或匹配证据

tf:
  reading_chain 中拆出的复合条件转成 1 到多个 evidence probes
```

也就是说，`model_outputs` 不是为了回答问题，而是为了降低后续“找证据”的搜索空间。

## Current Verdict

当前前置步骤 **匹配 GPTPro 的读题和裁决点拆解意图**。

但它 **不匹配 GPTPro 的完整解题链**，也不应该假装匹配。

更具体地说：

```text
mcq:   匹配度较高，已经接近 GPTPro check_option 的输入形状
multi: 匹配度较高，option-level route 很适合后续证据检索
tf:    中等匹配，只是读题链，还没有显式 probe 化
```

下一步最自然的工作不是继续改 Stage 1，而是做 Stage 2：

```text
model_outputs.jsonl
  -> evidence_probes.jsonl
  -> source spans
  -> option verdicts
```

其中最值得先试的是 overlap 的 6 题：看我们的 routes 能不能找回 GPTPro evidence 里那些 source / quote。
