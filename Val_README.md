# Validation And Debug Guide

本文放验证、预览、调参和小规模试跑命令。完整生成 `answer.csv` 的主流程见
[README.md](README.md)。

## 1. Dataset Check

确认问题里的 `doc_ids` 都能在 `Dataset/raw` 中找到：

```powershell
python -m script.validate_dataset
```

## 2. Unit Tests

运行全部单元测试：

```powershell
python -m pytest
```

当前测试覆盖：

```text
输出 answer.csv summary
断点续跑
PDF Markdown 优先读取
非法字符清理
结构化 evidence units
BM25 / 数字实体强召回
第一份/第二份文档约束
```

## 3. Structured Retrieval Preview

不调用 Qwen、不消耗模型 Token，只预览全库 BM25 和数字/实体召回结果：

```powershell
python -m script.preview_structured_retrieval --qid fc_a_001
```

限制每条 evidence card 展示字符数：

```powershell
python -m script.preview_structured_retrieval --qid fc_a_001 --chars 160
```

预览会按选项展示 evidence cards，并标注召回来源：

```text
字段命中：issuer / issue_size / trustee / rating_subject
值命中：10亿元 / 国信证券股份有限公司 / AAA
召回原因：bm25 / field / value / field_value_same_unit / chunk / noise
```

注意：正式答题默认是 `pageindex_first_structured: true`，会先用 PageIndex 选页，
但当前默认 `page_filter_mode: soft`，PageIndex 命中页只给 BM25 候选加小幅分数，不会硬过滤其它页。
这个预览脚本不调用 Qwen，所以展示的是未经过 PageIndex soft boost 的本地召回 sanity check。

更细的 BM25 实验脚本：

```powershell
python -m script.preview_bm25_debug --qid fc_a_001 --top-k 5 --chars 220
```

这个脚本不会调用 Qwen，会额外展示：

```text
parsed.doc_ids       第一份/第二份文档约束
parsed.fields        从选项抽出的核对字段，如 issuer / issue_size / trustee
parsed.values        从选项抽出的数字、机构名、评级
parsed.compare       是否识别为多文档/比较题
field_hits           候选证据里命中的字段
value_hits           候选证据里命中的值
reasons              BM25、field、value、field+value、chunk、noise 的加减分原因
per_doc_candidates   比较题下，每份文档各自召回的候选证据
```

它还会在内存里把表格拆成 `table_row` 候选，方便观察“表格整页召回”和“表格行召回”的差异。

## 4. Small Online Run

只跑前 3 道题：

```powershell
$env:DASHSCOPE_API_KEY="your_api_key"
python -m script.run_pipeline --limit 3 --build-missing --fresh
```

只跑指定题号：

```powershell
python -m script.run_pipeline --qid fc_a_001 --build-missing --fresh
```

运行指定问题文件：

```powershell
python -m script.run_pipeline --questions Dataset/questions/group_a/insurance_questions.json --build-missing --fresh
```

`--fresh` 表示不复用已有 `answer.csv`，适合对比不同检索参数的效果。

## 5. Retrieval Mode Ablation

`run_pipeline` 和 `run_all` 都支持 `--retrieval-mode`：

```text
config          使用 config/default.yaml
pageindex       纯 PageIndex
bm25            纯 field-aware BM25 evidence cards，不调用 PageIndex 选页
pageindex-bm25  PageIndex 先选页作为 soft boost，field-aware BM25 仍可召回其它页
```

建议每种模式指定不同输出文件，避免覆盖：

```powershell
python -m script.run_pipeline --limit 20 --fresh --build-missing `
  --retrieval-mode pageindex `
  --output-csv answer_pageindex_exp.csv `
  --evidence-json evidence_pageindex_exp.json

python -m script.run_pipeline --limit 20 --fresh --build-missing `
  --retrieval-mode bm25 `
  --output-csv answer_bm25_exp.csv `
  --evidence-json evidence_bm25_exp.json

python -m script.run_pipeline --limit 20 --fresh --build-missing `
  --retrieval-mode pageindex-bm25 `
  --output-csv answer_pageindex_bm25_exp.csv `
  --evidence-json evidence_pageindex_bm25_exp.json
```

全量跑法同理：

```powershell
python -m script.run_all --fresh --retrieval-mode pageindex --output-csv answer_pageindex.csv --evidence-json evidence_pageindex.json
python -m script.run_all --fresh --retrieval-mode bm25 --output-csv answer_bm25.csv --evidence-json evidence_bm25.json
python -m script.run_all --fresh --retrieval-mode pageindex-bm25 --output-csv answer_pageindex_bm25.csv --evidence-json evidence_pageindex_bm25.json
```

对比两份 answer：

```powershell
python -m script.compare_answers answer_pageindex.csv answer_pageindex_bm25.csv --out answer_diff.csv
```

## 6. Structured Units Debug

重新生成全部结构化证据：

```powershell
python -m script.format_structured
```

只处理某个领域：

```powershell
python -m script.format_structured --domain financial_contracts
```

只处理某个文档，输出到样例文件：

```powershell
python -m script.format_structured --doc-id text01 --out processed_data/structured_text01.jsonl
```

`structured_text01.jsonl` 这类文件只是调试样例，正式流程默认读取：

```text
processed_data/structured_units.jsonl
```

## 7. Retrieval Parameters

主要参数在 [config/default.yaml](config/default.yaml)：

```yaml
structured_retrieval:
  enabled: true
  units_path: processed_data/structured_units.jsonl
  split_tables: true
  max_units: 24
  per_option: 3
  per_option_per_doc: 1
  per_doc: 8
  max_evidence_chars: 18000
  min_score: 0.1
  bm25_k1: 1.5
  bm25_b: 0.75
  force_number_hits: 3
  force_entity_hits: 3
  force_rating_hits: 2
  number_bonus: 14.0
  organization_bonus: 18.0
  rating_bonus: 5.0
  context_phrase_bonus: 20.0
  noise_penalty: 18.0
  link_page_index_context: false
  pageindex_first_structured: true
  page_filter_mode: soft
  page_boost: 10.0
  linked_page_window: 0
  linked_max_pages_per_doc: 4
  linked_max_chars: 12000
```

调参建议：

```text
证据太长 -> 降低 max_evidence_chars / max_units
PageIndex soft boost 太强导致偏题 -> 降低 page_boost 或设置 page_filter_mode: off
PageIndex 选页可靠且想强约束 -> 设置 page_filter_mode: hard
页内漏证据 -> 增大 per_option / per_option_per_doc
数字题漏召回 -> 增大 force_number_hits 或 number_bonus
机构名题漏召回 -> 增大 force_entity_hits 或 organization_bonus
AAA/评级噪声太多 -> 降低 force_rating_hits 或 rating_bonus
声明页/签字页噪声太多 -> 增大 noise_penalty
需要退回纯 BM25 全库召回 -> 设置 pageindex_first_structured: false 或 --retrieval-mode bm25
确实需要追加 PageIndex 原页上下文 -> 设置 link_page_index_context: true，并调大 linked_max_chars
```

## 8. Output Files

正式输出：

```text
answer.csv
evidence.json
```

调试时可以对比：

```text
answer_v0.csv
evidence_v2.json
logs/llm_calls.jsonl
```

`answer.csv` 第一行 `summary` 是 Token 汇总：

```csv
qid,answer,prompt_tokens,completion_tokens,total_tokens
summary,,3627557,629,3628186
```

## 9. PDF Source Quality Check

如果发现某个文档在重建 PageIndex 或 structured evidence 后准确率下降，先检查它实际使用了哪个 PDF parsed source：

```powershell
python -X utf8 -m script.inspect_text_quality --doc-id text01
python -X utf8 -m script.inspect_text_quality --doc-id annual_byd_2024_report --terms 比亚迪 营业收入 净利润 现金流
python -X utf8 -m script.inspect_text_quality --doc-id annual_cmb_2025_report --terms 招商银行 营业收入 净利润 资本充足率
```

重点看 `[selector]` 部分：

```text
selected_model      build_index 会使用的解析源
split               page_markers 表示保留了 PDF 页标，char_chunks 表示只能按字数硬切
acceptable          当前解析源是否通过质量门控
suspicious_ratio    异常 Unicode 比例，过高通常表示 pypdf 文本层乱码
finance_term_hits   金融关键词命中数，过低通常表示正文丢失
```

如果某个 PDF 缺少 GLM-OCR/pypdf markdown，先补跑对应解析，再重建索引：

```powershell
python -m script.pdf_parse_three Dataset/raw --models glm-ocr pypdf --glmocr-mode maas
python -m script.build_index --force
python -m script.format_structured
```
