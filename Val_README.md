# BM25 Baseline Validation

当前代码只保留 BM25 top1 baseline：本地读文档，本地打分，直接输出最高分选项。

## Quick Checks

检查数据引用：

```powershell
python -m script.validate_dataset
```

跑一小批题：

```powershell
python -m script.run_pipeline --limit 5 --fresh --output-csv answer_debug.csv --evidence-json evidence_debug.json
```

全量跑：

```powershell
python -m script.run_all --fresh
```

对比两份答案：

```powershell
python -m script.compare_answers answer_old.csv answer.csv --out diff.csv
```

## Evidence

`evidence.json` 每题包含：

- `mode`: 固定为 `bm25_top1`
- `selected_option`: 被选为答案的选项
- `top_unit_id`, `top_doc_id`, `top_page`: top1 证据块位置
- `option_scores`: 每个选项的最高 BM25 分数

这个 baseline 的目标是快速打通闭环和提供一个粗糙下限，不做语义判断，也不会验证选项真假。
