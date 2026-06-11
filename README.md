# Financial Long Document QA Agent

AFAC2026 金融长文本 Agent 赛题(赛题四:金融长文本 Agent 的动态记忆压缩与高效问答挑战)的工程仓库。

目标不是写一次性提交脚本,而是沉淀一套**可复现、可评估、可复盘**的 long-document RAG/Agent pipeline,并且每一步改动都能用分数说话。

> **当前状态:`agent/` 和 `script/` 已退回 not-implemented 骨架(stub),正在按下面的新路线重构。** 保留了模块结构和签名作为参考,函数体均为 `raise NotImplementedError`。

## 工程目标

最终希望能清楚回答:

1. 系统从什么 baseline 出发。
2. 每次优化改了什么,准确率 / Token / 分领域表现怎么变。
3. 错误主要来自解析、检索、证据压缩、推理还是答案格式。
4. 这套系统如何一键复现并生成可提交的 `answer.csv`。

## 比赛事实(决定一切策略)

打分公式:

```text
TokenScore = max(0, min(1, (TokenBudget - TotalTokens) / TokenBudget))   # TokenBudget = 5,000,000
FinalScore = 100 * Accuracy * (0.7 + 0.3 * TokenScore)
```

从公式能读出三件事,直接决定路线:

1. **准确率绝对主导。** Token 效率最多影响 30% 的加权系数,且是在准确率之上做乘法。Baseline 准确率只有 17%,所以早期省 Token 几乎无意义——**所有早期精力砸在准确率上**,Token 优化是后期的事。
2. **A 榜和 B 榜是两个不同的问题,不是难度递增的同一个问题。**
   - **A 榜给 `doc_ids`**:每题只关联 2–4 篇文档,检索基本是送分的。难点在**读懂 + 跨文档比对 + 数值推理 + 答案格式**。
   - **B 榜不给 `doc_ids`**:要从大文档池里**先盲检找对文档**,再阅读判断。难点在**召回**。
3. **多选题不设部分分。** 100 道 A 榜题里 65 道是 multi(漏选/错选/多选均计错),20 道 tf,15 道 mcq。多选的全对要求让"读全证据"比"读到一条证据"重要得多。

关键时间点:A 榜 2026-06-08 ~ 07-21;B 榜 2026-07-22 ~ 07-24(仅 3 天,且需 A 榜有有效成绩才能进)。基准模型 **Qwen3.6-plus**,只能经阿里云百炼或魔搭调用 Qwen 系列,不得用其他模型。

官方 baseline:A 组 17/100、B 组 13/100,总计 15%,Token 约 751 万。说明"长文直接塞通用模型"不行。

## 路线(纵切 + 错误驱动,不是 pipeline 横切)

不按"解析→chunk→检索→答题"逐层铺。先用最糙的方式打通端到端、把分数打到榜上、建立量化闭环,之后**每一步都是一个带假设的实验,改完 A/B 量分**。各阶段内部"具体做什么"故意不写死——由上一阶段的 bad case 分布决定。

| 阶段 | 做什么 | 验收 |
|---|---|---|
| **S0 Walking skeleton** | 最糙端到端:数据适配 + 裸 Qwen 答题 + 答案格式化,产出合法 `answer.csv` | A 榜能跑完、CSV 合法、summary token 正确,复现 ~17% baseline,量化闭环建立 |
| **S1 A 榜准确率攻坚** | doc 给定的前提下死磕读懂 / 跨文档比对 / 数值推理 / 多选全对 / 答案格式 | 按错误分布展开实验(分领域 prompt、证据拼装、选项级检索、低置信自检……),每个改动 A/B 量分 |
| **S2 B 榜盲检** | 把"找对文档"作为独立问题攻克;A 榜去掉 `doc_ids` 模拟盲检验证召回 | 模拟盲检时目标文档多数进 top-k;召回率可量化 |
| **S3 Token 优化 + 审核包** | 准确率到顶后才做:动态压缩降 token;打包代码 / `processed_data` / `evidence` / `logs` / `README` | Token 下降不牺牲命中证据;能一键复现,压缩包 < 1GB |

> S1/S2 内部的具体技术(选项级检索、分领域策略、记忆压缩、二阶段自检等)是**实验队列**,不是预排路线。先看 S0 的错误来自哪,再决定先做哪个。

## 数据概览(已核对)

`public_dataset_upload/`(343MB,git 忽略)覆盖五个领域,A 榜共 100 题、引用 68 个去重文档(全部可解析,missing = 0):

| 领域 | A 榜题数 | raw 文件 | 备注 |
|---|---|---|---|
| insurance 保险条款 | 20 | `1.pdf`…`16.pdf` | 含计算题(身故保险金、退保) |
| regulatory 监管法规 | 20 | 6 txt + 377 html + 130 att.pdf | 引用仅 15 个;其余大概率是 B 榜盲检干扰池 |
| financial_contracts 金融合同 | 20 | `text01`…`text14`.pdf | 债券募集说明书 |
| financial_reports 财务报表 | 20 | 10 份 `.PDF`(注意大写) | 多为同公司两年年报对比 |
| research 行业研报 | 20 | `pack2_text01`…`20`.pdf | 跨报告指标核验 |

> 解析坑(S0/解析阶段注意):监管公文 att.pdf 有**竖排版式**,pypdf 抽出来是一行一字(中文不靠空格,拼回即可);3 个引用文档是 HTML;财报扩展名是大写 `.PDF`;监管 txt 的 `doc_id` 是含全角括号〔〕的完整文件名 stem,不要清洗。

## 输入契约

`data/documents.jsonl` 每行一个文档:

```json
{"doc_id":"annual_byd_2024_report","path":"...","title":"比亚迪 2024 年年度报告","domain":"financial_reports"}
```

`data/questions.jsonl` 每行一道题(兼容 A 榜含 `doc_ids` / B 榜无 `doc_ids`):

```json
{"qid":"fin_a_001","domain":"financial_reports","split":"A","question":"...","options":{"A":"...","B":"...","C":"...","D":"..."},"answer_format":"multi","raw_type":"财务指标对比分析","doc_ids":["annual_byd_2024_report","annual_byd_2025_report"]}
```

## 字段一致性审计(A 榜 questions,已核对)

逐字段核过官方 `questions/group_a/*_questions.json`(每域 20 题)。**结论:绝大多数字段是干净的,但有两个字段在五个领域间根本不是同一个东西,不要当成统一字段用。**

**干净、可直接透传:**

- `domain`:5 个值,永远等于文件名;无 mismatch。
- `split`:全为 `"A"`(B 榜数据尚未到位,届时再说)。
- `question`:题干干净,没有把选项嵌进题干,也没有重复选项字母。
- `options`:键永远连续(`tf` 用 `AB`,其余 `ABCD`);选项数**永远**和 `answer_format` 对应;`tf` **永远**是 `A=正确 / B=错误`。
- `answer_format`:值干净 `{multi, tf, mcq}`,且能精确预测选项数。⚠️ 但**题型分布按领域不同**:其余四域都是 `13 multi / 5 tf / 2 mcq`,**唯独 insurance 没有 tf,是 7 mcq / 13 multi**。不是数据错误,是真实差异,做分领域 prompt 时要记得。

**不一致,需要管理:**

- **`raw_type`(原 `type`):五域语义不统一,已重命名以示警。** 它混了两个轴、用了不一致的词表:
  - regulatory 的 `type` 基本是在用 8 种拼法复述 `answer_format`(`多选题`/`multi`/`多选`/`判断题`/`判断`/`单选`/`tf`/`mcq`),零语义。
  - insurance 是一套像样的 5 类技能标签(`逻辑推理`/`推理判断`/`比较分析`/`事实查询`/`计算题`)。
  - research 几乎是**每题一个自由英文 slug**(`pet_market`/`chip_security`…)。
  - financial_contracts 中英文 slug 混用(`clause_verification`/`time_comparison`…)。
  - **处置:** 保留原值但字段名改为 `raw_type`,明确"按领域、未归一化、不得用于跨领域路由"。真正的归一化 `category`(若需要)留到 S0/S1 错误分析后再做,不预先发明。

- **`qid`:结构统一 `{prefix}_a_{NNN}`、全局唯一,但前缀有坑。** `fc`=contracts、**`fin`=reports(不是 contracts)**、`ins`/`reg`/`res`。**铁律:永远不要从 qid 前缀推 domain,用 `domain` 字段。** `qid` 是 `answer.csv` 的强制提交键,不可删;`domain` 虽可由前缀推导,但因这个坑,保留它更省事。两者都是**簿记键,不是模型检索/推理的对象**——给它们起"AI 友好"的名字优化错了层。

- **`doc_ids`:元素恒为 str,但 doc_id 命名空间五域完全不同——这正是 step 2(`documents.jsonl`)文件映射的核心难点。**

  | 领域 | doc_id 形如 | 映射 raw 文件 |
  |---|---|---|
  | insurance | `"1"`、`"15"`(裸数字) | `raw/insurance/1.pdf`;且每题 **2–4** 篇(其余域恒为 2) |
  | financial_contracts | `"text01"` | `raw/financial_contracts/text01.pdf`,直接 stem |
  | research | `"pack2_text01"` | `raw/research/pack2_text01.pdf`,直接 stem |
  | regulatory | `"strict_v3_008_…令〔2025〕第12号(…)"` | stem **就是**完整文件名,含全角 `〔〕()`,**不要清洗** |
  | financial_reports | `"annual_byd_2024_report"` | **语义别名,不是文件名** → 10 份大写 `.PDF`,需手建查表 |

  **处置:** `doc_id` **原样保留**作为机器键(它是官方 join 键 + `evidence.json` 审计可追溯键,改名会破坏两侧关联)。step 2 用**按领域的 doc_id resolver**(前四域 `glob('raw/<domain>/<doc_id>.*')` 取唯一匹配;financial_reports 手建别名→`.PDF` 映射),硬校验"每个被引用 doc_id 唯一命中一个文件、`missing == 0`"。文件系统不安全字符(regulatory 的 〔〕())只在 `processed_data/` 文件名上用干净 slug,与 doc_id 解耦。

  > 提升"AI 找文档"的杠杆不在 id 命名,而在 **`title`(语义化:公司+年份+文档类型)+ 注入 chunk 的正文**——A 榜 doc 已给(查表即得),B 榜召回走 chunk 内容/向量,doc_id 字符串根本不进索引。

## 提交格式

`answer.csv` 第一行为 `summary`(全程 token 统计),其后每题一行:

```text
qid,answer,prompt_tokens,completion_tokens,total_tokens
```

答案规范:单选/判断一个大写字母;多选多个大写字母按字母序、无分隔符(如 `ABC`)。空答案、非法字符、顺序不规范或与标准答案不完全一致均计错。

审核包(B 榜前 15 名需提交)结构:`answer.csv`、**`evidence.json`**(含 `doc_id` / `quoted_clause` / `reasoning` 的证据可追溯结构)、`processed_data/`、`agent/`、`script/`、`logs/`、`requirements.txt`、`README.md`,总大小 < 1GB。

## 比赛边界

**预处理 / 解析阶段**可用任意工具,包括非 Qwen 模型(OCR、版面分析、表格恢复、PDF 转结构化,如 MinerU),但只能用于把文档变成更可读的结构化输入。

**正式答题阶段**(文档定位、段落检索、记忆压缩、证据判断、答案生成、自检纠错)只能用 Qwen 系列。明确禁止:非 Qwen 模型产生的向量 / 排序 / 召回参与答题;非 Qwen 参与 rerank / 候选过滤 / 投票 / 纠错;直接使用预处理阶段生成的语义摘要 / FAQ / 结论 / 知识库参与答题。

## 目录

```text
public_dataset_upload/   # 官方数据(git 忽略),raw 按领域分目录 + questions/group_a/*.json
data/                    # 适配后的工程输入:documents.jsonl / questions.jsonl + raw/(staged 68 篇引用文档,path 相对 data/)
processed_data/          # 解析后的全文与检索 chunk
agent/                   # Agent 系统代码(当前为 stub)
script/                  # 可复现运行脚本(当前为 stub)
tests/
```

## 环境与运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.json config.json
export DASHSCOPE_API_KEY="your-api-key"   # Qwen / 百炼
```

> 运行入口(`script/preprocess.py`、`script/run_answer.py` 等)随 S0 重构后补全;当前为 not-implemented stub。
