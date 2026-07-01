# Qwen Thinking 中的 Evidence Gap 模式分析

日期：2026-06-24

分析对象：

- 文件：`runs/03_bm25_qwen_thinking_v0/evidence.json`
- 字段：`evidence_retrieval[0].qwen_thinking`

本分析只参考 `qwen_thinking` 文本本身。题号只作为定位索引，所有模式判断都来自 thinking text。

## 1. 结论

`qwen_thinking` 里最值得利用的信号，是模型经常自己识别到 evidence gap：它会在内部推理中说某些信息“未提及”“无证据”“无法确认”“只覆盖部分内容”或“还需要另一份文档/更精确条件”。这些信号说明模型并不是完全没意识到证据不足，而是当前 prompt 没有把这些 gap 强制转化为最终决策约束。

换句话说，thinking 暴露的问题不是“模型不会检查证据”，而是：

1. 模型检查出了证据缺口；
2. 但缺口没有被结构化记录；
3. 缺口也没有被稳定地转化为“不选/保守处理/要求更多证据”。

因此下一版 prompt 最应该约束的是：只要 thinking 中出现 evidence gap，模型必须把相关选项标记为 `insufficient`，而不是继续用排除法、常识或局部证据补完。

## 2. Evidence Gap 的判定方式

我只对 `qwen_thinking` 文本做关键词/语义模式扫描。命中不代表题目一定错，只代表模型在 thinking 中显式暴露了某类证据缺口。

本次把 evidence gap 分成五类：

| 模式 | 含义 | thinking 中的典型触发词 |
|---|---|---|
| not_mentioned | 给定证据没有提到某信息 | 未提及、没有提及、未显示、未说明、未明确、未找到 |
| no_evidence | 模型明确认为证据不足 | 无证据、没有证据、证据不足、无法判断、不能确定 |
| partial_coverage | 证据只覆盖选项的一部分 | 仅、只、部分、未覆盖、无法确认 |
| need_more_docs | 当前证据缺少其他文档/主体 | 另一份文档、其他文档、第二份、第三份、两份文档、所有文档、每份文档 |
| need_exact_condition | 缺少精确条件、口径或限定 | 未说明是否、未明确是否、未给出、缺少信息、不能推出 |

这些模式有重叠。例如一个 thinking 可能同时命中 `not_mentioned` 和 `partial_coverage`，表示模型既发现“没提到”，也发现“只覆盖了部分条件”。

## 3. 总体命中情况

| 项目 | 数量 |
|---|---:|
| 总题数 | 100 |
| 命中任一 evidence gap 模式 | 57 |
| 未命中 evidence gap 模式 | 43 |

按模式统计：

| 模式 | 命中题数 |
|---|---:|
| no_evidence | 28 |
| partial_coverage | 27 |
| not_mentioned | 23 |
| need_more_docs | 13 |
| need_exact_condition | 12 |

注意：各模式之和大于 57，因为同一道题可以命中多个模式。

## 4. 五类 Evidence Gap 详解

### 4.1 not_mentioned：证据未提及

这类 thinking 会明确指出给定 evidence 没有提到某个信息。它是最直观的 evidence gap。

命中题目：

```text
fc_a_005
fc_a_008
fc_a_009
fc_a_010
fc_a_015
fc_a_016
fc_a_017
fc_a_020
fin_a_002
fin_a_004
fin_a_007
fin_a_014
fin_a_017
fin_a_018
ins_a_008
ins_a_014
ins_a_017
ins_a_018
reg_a_011
reg_a_019
res_a_002
res_a_004
res_a_017
```

这个模式对 prompt 的启发：

- “未提及”不能被自动当作“错误”。
- 如果 evidence 是 top-k 片段，不是完整原文，那么“未提及”更应该转为 `insufficient`。
- 被选中的选项不能依赖“其他选项未提及”来成立。

建议 prompt 约束：

```text
如果 thinking 中判断某选项的关键信息在 evidence 中未提及，则该选项必须标记为 insufficient，不能选入 answer。
```

### 4.2 no_evidence：无证据/证据不足

这类 thinking 比 not_mentioned 更强，模型不只是说没提到，而是明确承认证据不足、无证据或无法判断。

命中题目：

```text
fc_a_002
fc_a_007
fc_a_008
fc_a_009
fc_a_010
fc_a_011
fc_a_014
fc_a_017
fc_a_018
fc_a_019
fin_a_002
fin_a_005
fin_a_006
fin_a_007
fin_a_014
fin_a_018
fin_a_020
ins_a_005
ins_a_008
ins_a_011
ins_a_014
ins_a_017
ins_a_018
reg_a_005
res_a_002
res_a_014
res_a_019
res_a_020
```

这个模式是最应该被 prompt 利用的，因为它说明模型自己已经给出了“不应强答”的信号。

建议 prompt 约束：

```text
如果某选项被判断为无证据、证据不足、无法判断或不能确定，则该选项不能进入 answer。禁止在证据不足后继续用常识、背景知识或排除法补全。
```

### 4.3 partial_coverage：只覆盖部分条件

这类 thinking 说明 evidence 不是完全缺失，而是只覆盖了选项的一部分。它比“无证据”更隐蔽，因为模型很容易被局部证据诱导。

命中题目：

```text
fc_a_003
fc_a_004
fc_a_007
fc_a_008
fc_a_010
fc_a_012
fc_a_013
fc_a_014
fc_a_016
fin_a_001
fin_a_006
fin_a_008
fin_a_012
fin_a_014
fin_a_015
fin_a_017
fin_a_018
ins_a_006
ins_a_014
ins_a_017
ins_a_018
ins_a_020
reg_a_006
reg_a_010
reg_a_012
reg_a_013
reg_a_018
```

典型风险：

- 选项包含多个主体，但 evidence 只覆盖一个主体。
- 选项包含多个条件，但 evidence 只支持其中一个条件。
- 选项包含比较关系，但 evidence 只给出单边数据。
- 选项包含产品责任/免责范围，但 evidence 只覆盖某个产品或某个责任。

建议 prompt 约束：

```text
如果 evidence 只支持选项的一部分，则该选项必须标记为 insufficient。只有选项中的所有关键主体、条件、数值和关系都被支持，才能标记为 supported。
```

### 4.4 need_more_docs：需要其他文档/主体证据

这类 thinking 说明模型意识到当前证据池可能没有覆盖所有 doc_id 或所有被比较对象。

命中题目：

```text
fc_a_001
fc_a_002
fc_a_003
fc_a_004
fc_a_005
fc_a_006
fc_a_009
fc_a_013
fc_a_016
fc_a_017
fc_a_018
fc_a_019
fc_a_020
```

thinking 里出现这类 gap 时，通常意味着当前 evidence 没有覆盖选项所需的全部文档或全部主体。

对 prompt 的启发：

- 如果选项涉及多个 doc_id，就不能只用其中一个 doc_id 的证据支持整个选项。
- 如果 thinking 发现还需要另一份文档或其他文档的信息，相关选项应该进入 `insufficient`。
- 这类 gap 更适合后续用 `option × doc_id` 检索解决。

建议 prompt 约束：

```text
如果选项涉及多个文档、多个发行主体或多个材料来源，则每个相关 doc_id 都必须有支持证据。任一 doc_id 缺证据时，该选项为 insufficient，不能选。
```

### 4.5 need_exact_condition：缺少精确条件

这类 thinking 说明 evidence 提到了相关主题，但缺少能决定选项真假的精确条件。

命中题目：

```text
fc_a_005
fc_a_018
fin_a_007
fin_a_009
fin_a_011
fin_a_012
fin_a_014
fin_a_018
ins_a_011
ins_a_014
res_a_014
res_a_017
```

典型缺口包括：

- 是否满足某个前提条件；
- 是否覆盖某个具体产品；
- 是否是同一年份、同一单位、同一指标口径；
- 是否有具体数值支撑；
- 是否能从已有证据推出比较关系。

建议 prompt 约束：

```text
如果 evidence 只提到相近主题，但缺少决定选项真假的精确条件、口径、单位、年份或前提，则该选项必须标记为 insufficient。
```

## 5. 组合模式

有些题会同时命中多种 evidence gap。组合模式比单一模式更值得关注，因为它说明 thinking 里存在多层证据缺口。

高频组合：

| 组合 | 题数 |
|---|---:|
| partial_coverage | 11 |
| no_evidence | 7 |
| not_mentioned | 5 |
| not_mentioned + no_evidence + partial_coverage | 4 |
| partial_coverage + need_more_docs | 3 |
| no_evidence + partial_coverage | 3 |
| not_mentioned + no_evidence | 3 |
| not_mentioned + no_evidence + partial_coverage + need_exact_condition | 3 |

最需要警惕的是：

```text
not_mentioned + no_evidence + partial_coverage
```

这个组合表示：模型既发现证据没有提及某些内容，又承认证据不足，同时还发现已有证据只覆盖部分条件。此时如果最终还给出确定选项，就很可能是 prompt 没有把 evidence gap 压住。

## 6. 全量命中题目清单

下面只列出 qid 和命中的 evidence gap 类型。

```text
fc_a_001: need_more_docs
fc_a_002: no_evidence + need_more_docs
fc_a_003: partial_coverage + need_more_docs
fc_a_004: partial_coverage + need_more_docs
fc_a_005: not_mentioned + need_more_docs + need_exact_condition
fc_a_006: need_more_docs
fc_a_007: no_evidence + partial_coverage
fc_a_008: not_mentioned + no_evidence + partial_coverage
fc_a_009: not_mentioned + no_evidence + need_more_docs
fc_a_010: not_mentioned + no_evidence + partial_coverage
fc_a_011: no_evidence
fc_a_012: partial_coverage
fc_a_013: partial_coverage + need_more_docs
fc_a_014: no_evidence + partial_coverage
fc_a_015: not_mentioned
fc_a_016: not_mentioned + partial_coverage + need_more_docs
fc_a_017: not_mentioned + no_evidence + need_more_docs
fc_a_018: no_evidence + need_more_docs + need_exact_condition
fc_a_019: no_evidence + need_more_docs
fc_a_020: not_mentioned + need_more_docs
fin_a_001: partial_coverage
fin_a_002: not_mentioned + no_evidence
fin_a_004: not_mentioned
fin_a_005: no_evidence
fin_a_006: no_evidence + partial_coverage
fin_a_007: not_mentioned + no_evidence + need_exact_condition
fin_a_008: partial_coverage
fin_a_009: need_exact_condition
fin_a_011: need_exact_condition
fin_a_012: partial_coverage + need_exact_condition
fin_a_014: not_mentioned + no_evidence + partial_coverage + need_exact_condition
fin_a_015: partial_coverage
fin_a_017: not_mentioned + partial_coverage
fin_a_018: not_mentioned + no_evidence + partial_coverage + need_exact_condition
fin_a_020: no_evidence
ins_a_005: no_evidence
ins_a_006: partial_coverage
ins_a_008: not_mentioned + no_evidence
ins_a_011: no_evidence + need_exact_condition
ins_a_014: not_mentioned + no_evidence + partial_coverage + need_exact_condition
ins_a_017: not_mentioned + no_evidence + partial_coverage
ins_a_018: not_mentioned + no_evidence + partial_coverage
ins_a_020: partial_coverage
reg_a_005: no_evidence
reg_a_006: partial_coverage
reg_a_010: partial_coverage
reg_a_011: not_mentioned
reg_a_012: partial_coverage
reg_a_013: partial_coverage
reg_a_018: partial_coverage
reg_a_019: not_mentioned
res_a_002: not_mentioned + no_evidence
res_a_004: not_mentioned
res_a_014: no_evidence + need_exact_condition
res_a_017: not_mentioned + need_exact_condition
res_a_019: no_evidence
res_a_020: no_evidence
```

## 7. Prompt 应如何利用这些 Evidence Gap

下一版 prompt 不需要让模型“多想一点”。thinking 已经显示它会主动发现 gap。需要的是让模型在发现 gap 后改变输出策略。

建议加入如下判题协议：

```text
在内部核验每个选项时，必须给出 supported / contradicted / insufficient 三类之一。

supported:
给定 evidence 中有直接正向证据，且覆盖选项的所有关键主体、doc_id、条件、数值、年份、单位和关系。

contradicted:
给定 evidence 中有直接证据反驳该选项。

insufficient:
thinking 中出现未提及、无证据、证据不足、无法判断、只覆盖部分内容、缺少其他文档、缺少精确条件等 evidence gap。

最终 answer 只能包含 supported 选项。
insufficient 选项不能进入 answer。
禁止用排除法、常识或背景知识把 insufficient 改成 supported。
```

再加一条更直接的 evidence gap 约束：

```text
如果你在内部判断中发现某个选项存在 evidence gap，不要猜测该选项正确或错误；将它视为 insufficient，并从最终 answer 中排除。
```

## 8. 最小 Prompt Patch

如果只想做最小改动，可以在现有 prompt 的 JSON 输出要求前加这一段：

```text
Evidence gap rule:
For every option, first decide whether the provided evidence is sufficient.
If the evidence does not mention the key fact, lacks direct support, only supports part of the option, misses a required document/entity/product, or lacks exact numeric/date/unit/condition information, classify that option as insufficient.
Do not include insufficient options in answer.
Do not select an option by elimination alone.
Every selected option must have direct positive support in the provided evidence.
```

中文版本：

```text
证据缺口规则：
对每个选项，先判断给定 evidence 是否足够。
如果 evidence 未提及关键事实、没有直接支持、只支持选项的一部分、缺少必要文档/主体/产品，或缺少精确数值/日期/单位/条件，则该选项为 insufficient。
insufficient 选项不能进入 answer。
禁止只用排除法选择选项。
每个被选中的选项都必须有给定 evidence 中的正向证据。
```

## 9. 总结

只看 `qwen_thinking` 文本，可以得到一个很清楚的结论：模型已经频繁识别到 evidence gap，但当前 prompt 没有把 gap 变成硬约束。

最应该加的不是更长解释，而是一个简单的 evidence gap gate：

```text
命中 evidence gap -> 选项 insufficient -> 不进入最终 answer
```

如果后续再做 `option × doc_id` 检索，这个 gate 仍然有用，因为它能让 Qwen 明确地区分“证据支持”“证据反驳”和“证据没覆盖”。
