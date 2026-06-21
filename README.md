# Financial Long-Document Agent

面向 AFAC2026 赛题四的金融长文本问答 Agent。主流程是：

```text
Dataset/raw
  -> PDF/TXT/HTML 文档解析
  -> PageIndex 索引
  -> structured_units 结构化证据单元
  -> PageIndex 定位页范围
  -> field-aware BM25 / 数字实体索引做页内证据压缩
  -> Qwen 证据判断
  -> answer.csv
```

本文只保留从环境准备到生成 `answer.csv` 的完整运行流程。验证、预览和调参命令见
[Val_README.md](Val_README.md)。

## 1. Install

```powershell
python -m pip install -r requirements.txt
```

## 2. Prepare Data

确认官方数据放在：

```text
Dataset/raw/<domain>/*
Dataset/questions/group_a/*.json
```

检查题目里的 `doc_ids` 是否都能在 `Dataset/raw` 中找到：

```powershell
python -m script.validate_dataset
```

## 3. PDF To Markdown

PDF 建议先离线转成 Markdown。正式跑分优先用 GLM-OCR：

```powershell
$env:ZHIPU_API_KEY="your_zhipu_api_key"
python -m script.pdf_parse_three Dataset/raw --models glm-ocr --glmocr-mode maas
```

如果只是先打通流程，可以用 `pypdf` 快速生成文本层 Markdown：

```powershell
python -m script.pdf_parse_three Dataset/raw --models pypdf
```

输出位置：

```text
processed_data/pdf_parsed/<model>/<doc_id>.md
```

当前 PDF Markdown 读取优先级在 [config/default.yaml](config/default.yaml)：

```yaml
preprocess:
  pdf_parsed_dir: processed_data/pdf_parsed
  pdf_model_order:
    - glm-ocr
    - mineru2.5-pro
    - paddleocr-vl-1.6
    - pypdf
```

TXT/HTML 不需要单独转 Markdown，后续 `build_index` 会直接读取并规范化。

## 4. Build PageIndex

构建全部文档索引：

```powershell
python -m script.build_index
```

这一步会生成：

```text
processed_data/<domain>/<doc_id>/metadata.json
processed_data/<domain>/<doc_id>/pages.jsonl
processed_data/<domain>/<doc_id>/page_index.json
```

如果替换了某个 PDF 的 OCR Markdown，需要强制重建对应文档：

```powershell
python -m script.build_index --doc-id text02 --force
```

## 5. Format Structured Evidence Units

把 `pages.jsonl` 转成 BM25 和数字/实体召回使用的结构化证据单元：

```powershell
python -m script.format_structured
```

输出：

```text
processed_data/structured_units.jsonl
```

正式答题时会先由 Qwen 根据 PageIndex 选择相关页，再只在这些页对应的 evidence units
上做选项解析、表格行拆分、field/value rerank、比较题 per-doc RAG 和选项级 evidence
cards。这样 PageIndex 负责“去哪读”，field-aware BM25 负责“页内保留什么证据”。如果
结构化压缩没有命中，会直接使用 PageIndex 选中的原页上下文。

## 6. Configure Qwen

设置 DashScope API Key：

```powershell
$env:DASHSCOPE_API_KEY="your_api_key"
```

模型、输出路径、Token 上限和检索参数都在 [config/default.yaml](config/default.yaml)。

## 7. Generate answer.csv

全量运行并生成 `answer.csv`：

```powershell
python -m script.run_all
```

默认使用 [config/default.yaml](config/default.yaml) 里的检索配置。也可以用
`--retrieval-mode` 临时切换：

```powershell
python -m script.run_all --retrieval-mode pageindex
python -m script.run_all --retrieval-mode bm25
python -m script.run_all --retrieval-mode pageindex-bm25
```

三种模式分别对应纯 PageIndex、纯 field-aware BM25、PageIndex 选页后 field-aware BM25
页内压缩。

输出格式：

```csv
qid,answer,prompt_tokens,completion_tokens,total_tokens
summary,,3627557,629,3628186
ins_a_001,B,37201,1,37202
```

脚本每完成一道题就刷新 `answer.csv` 和 `evidence.json`。中途异常时，修复 API 或网络问题后
再次执行同一命令，会自动跳过已有答案继续运行。

完全重跑：

```powershell
python -m script.run_all --fresh
```

## 8. Minimal One-Shot Flow

```powershell
python -m pip install -r requirements.txt
python -m script.validate_dataset
$env:ZHIPU_API_KEY="your_zhipu_api_key"
python -m script.pdf_parse_three Dataset/raw --models glm-ocr --glmocr-mode maas
python -m script.build_index
python -m script.format_structured
$env:DASHSCOPE_API_KEY="your_api_key"
python -m script.run_all
```

## Notes

- `processed_data/`、`logs/`、`answer*.csv` 和 `evidence*.json` 是可再生成产物，默认不进 Git。
- `answer.csv` 第一行 `summary` 是全部题目的 Token 汇总。
- `evidence.json` 保存每道题的证据和模型返回信息，方便赛后排查。
