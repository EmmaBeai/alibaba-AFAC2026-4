# Financial Long-Document BM25 + Qwen V0

面向 AFAC2026 赛题四的 v0 答案链路。当前默认主流程只有一条，不包含 PageIndex：

```text
public_dataset_upload/raw/<domain>/*
public_dataset_upload/questions/group_a/*.json
  -> read processed_data/chunks.jsonl from OCR/Markdown
  -> BM25 select top5 evidence patches
  -> qwen3.7-max answers single/multi/tf question from those patches
  -> runs/<run_id>/answer.csv / evidence.json / config.yaml / manifest.json
```

默认 answerer 是 `bm25_qwen`。它需要 Qwen 兼容接口 key，优先读取 `DASHSCOPE_API_KEY`，也兼容本机常见的 `QWEN_API_KEY`。`bm25_top1` 仍保留为 retrieval-only 对照，不算正式 v0。

## Install

```powershell
python -m pip install -r requirements.txt
```

## Prepare Data

代码默认读取：

```text
public_dataset_upload/raw/<domain>/*
public_dataset_upload/questions/group_a/*.json
```

先检查题目里的 `doc_ids` 是否都能在 `public_dataset_upload/raw` 中找到：

```powershell
python -m script.validate_dataset
```

## Generate answer.csv

全量运行：

```powershell
python -m script.run_all
```

默认输出在：

```text
runs/01_bm25_qwen_v0/
├── answer.csv
├── evidence.json
├── config.yaml
└── manifest.json
```

完全重跑：

```powershell
python -m script.run_all --fresh
```

只跑部分题目、改 run 目录或改输出文件：

```powershell
python -m script.run_pipeline --qid ins_a_001 --fresh
python -m script.run_pipeline --limit 20 --run-dir runs/debug_bm25_qwen --fresh
python -m script.run_pipeline --limit 20 --output-csv runs/debug_bm25_qwen/answer.csv --evidence-json runs/debug_bm25_qwen/evidence.json
```

输出格式：

```csv
qid,answer,prompt_tokens,completion_tokens,total_tokens
summary,,0,0,0
ins_a_001,AC,1200,80,1280
```

`evidence.json` 会保存每题进入 Qwen 的 BM25 top5 patch、命中文档、chunk id、Qwen 原始答案和简要理由。

## Validate and Compare Runs

提交前检查 `answer.csv`、`evidence.json` 和题目集是否对齐：

```powershell
python -m script.validate_submission --run-dir runs/01_bm25_qwen_v0
```

按 `qid` 比较两个 run：

```powershell
python -m script.compare_runs runs/00_bm25_top1 runs/01_bm25_qwen_v0 --out runs/01_bm25_qwen_v0/qid_diff.csv
```

## Notes

- `answer.csv` 第一行 `summary` 保留提交格式；`bm25_qwen` 会记录 Qwen prompt/completion token，`bm25_top1` 的 token 为 0。
- 默认 BM25 读取 `processed_data/chunks.jsonl`，这个文件来自 OCR/Markdown 解析产物；只有缺少可用 chunk 时才会回退 raw PDF/TXT/HTML。
- `bm25_qwen` 只做 BM25 top5 patch + qwen3.7-max，不调用 PageIndex、文档树选择或多轮 agent。
- `runs/<run_id>/manifest.json` 记录数据路径和版本、Git commit、chunk/index 配置、模型和 prompt 版本、token 使用量、输出检查结果以及相对上一个 run 改变的 qid。
- `processed_data/`、`logs/`、顶层 `answer*.csv` 和 `evidence*.json` 是可再生成产物，默认不进 Git。
