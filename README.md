# Financial Long-Document Agent

面向 AFAC2026 赛题四的无向量 PageIndex 工作流。系统把 PDF、HTML 和 TXT
预处理为页文本与层级目录树，正式答题阶段仅由 Qwen 完成文档路由、目录树检索、
证据判断和答案生成。

## Workflow

```text
Dataset
  -> 文档发现与 doc_id 校验
  -> 页级文本抽取
  -> 结构化 PageIndex 树
  -> A 榜按 doc_ids / B 榜由 Qwen 路由
  -> Qwen 选择 PageIndex 叶节点
  -> 加载原始证据页
  -> Qwen 逐项判断
  -> 答案规范化 + Token 台账
  -> answer.csv + evidence.json
```

预处理阶段不生成语义摘要、向量或非 Qwen 排序结果。`processed_data` 只包含来源
元数据、原始页文本和结构目录，避免把预处理能力越界用于正式推理。

## Project Layout

```text
agent/              Agent 核心代码
config/             模型与流程配置
script/             数据校验、索引构建、答题入口
processed_data/     生成的页文本与 PageIndex
logs/               Qwen 调用及 Token 记录
tests/              单元测试
Dataset/            竞赛原始数据
```

## Quick Start

```powershell
python -m pip install -r requirements.txt
python -m script.validate_dataset
python -m script.build_index

$env:DASHSCOPE_API_KEY="..."
python -m script.run_pipeline --qid fin_a_003
```

一次性构建所需索引、回答当前问题目录中的全部题目并生成 `answer.csv`：

```powershell
$env:DASHSCOPE_API_KEY=""
python -m script.run_all
```

`answer.csv` 包含每题的 `prompt_tokens`、`completion_tokens`、`total_tokens`，
第一行 `summary` 为全部题目的 Token 汇总。脚本每完成一道题就刷新
`answer.csv` 和 `evidence.json`，中途异常时已完成结果仍会保留。修复 API
或网络问题后再次执行同一命令，会自动跳过已有答案并继续运行。需要完全重跑时使用：

```powershell
python -m script.run_all --fresh
```

按领域或单文档构建索引：

```powershell
python -m script.build_index --domain insurance
python -m script.build_index --doc-id annual_cscec_2025_report
```

运行指定问题文件：

```powershell
python -m script.run_pipeline --questions Dataset/questions/group_a/insurance_questions.json
```

默认模型、页分组大小、证据上限和输出路径均可在
`config/default.yaml` 中调整。
