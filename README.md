# Financial Long-Document Agent

面向 AFAC2026 赛题四的金融长文本问答 Agent。项目采用 PageIndex 思路：
先把原始文档预处理成页文本和结构目录树，正式答题阶段只使用 Qwen 完成文档路由、
页节点选择、证据判断和答案生成。

## Workflow

```text
Dataset/raw
  -> preprocess 文档解析
       PDF  : 优先消费 GLM-OCR / MinerU / PaddleOCR / pypdf 生成的 Markdown
       TXT  : 规范编码和换行，保留条文结构
       HTML : 抽正文，去导航/脚本/页脚
  -> build_index 生成 PageIndex
       processed_data/<domain>/<doc_id>/metadata.json
       processed_data/<domain>/<doc_id>/pages.jsonl
       processed_data/<domain>/<doc_id>/page_index.json
  -> format_structured 生成结构化证据单元
       processed_data/structured_units.jsonl
  -> Qwen 检索和答题
       A 榜直接使用题目给出的 doc_ids
       B 榜由 Qwen 先做文档路由
       优先使用 structured_units 做 option-level retrieval
       未命中时回退 PageIndex 选页
  -> answer.csv + evidence.json
```

预处理阶段可以使用非 Qwen 工具做 OCR、版面恢复和结构化解析，但不能把非 Qwen
生成的语义摘要、向量、排序结果或答案判断结果带入正式答题。

## Project Layout

```text
agent/              Qwen Agent、PageIndex、输出和 Token 统计
preprocess/         文档解析器，属于非 Qwen 预处理边界
config/             模型、预处理和 PageIndex 配置
script/             数据校验、索引构建、答题入口
processed_data/     生成的页文本、PageIndex 和可选 PDF Markdown
logs/               Qwen 调用与 Token 日志
tests/              单元测试
Dataset/            竞赛数据
```

## 1. Install

```powershell
python -m pip install -r requirements.txt
```

## 2. Validate Data

确认问题里的 `doc_ids` 都能在 `Dataset/raw` 中找到：

```powershell
python -m script.validate_dataset
```

## 3. Material Preprocess

本项目不是把所有格式都先转成 `.md`。当前处理方式是：

```text
PDF  -> 可先离线转 Markdown，输出到 processed_data/pdf_parsed/<model>/<doc_id>.md
TXT  -> 不需要单独转 md，build_index 会读取并规范化
HTML -> 不需要单独转 md，build_index 会抽取正文
```

所以只有 PDF 建议先做 OCR/Markdown 预处理。TXT/HTML 会在 `script.build_index`
里自动完成轻量解析。

### PDF To Markdown

正式跑分建议优先用 GLM-OCR：

```powershell
$env:ZHIPU_API_KEY="your_zhipu_api_key"
python -m script.pdf_parse_three Dataset/raw --models glm-ocr --glmocr-mode maas
```

如果只是先打通流程，可以用 `pypdf` 快速生成文本层 Markdown：

```powershell
python -m script.pdf_parse_three Dataset/raw --models pypdf
```

这个命令可以重复执行，已生成的文件会自动跳过。输出位置：

```text
processed_data/pdf_parsed/<model>/<doc_id>.md
```

例如：

```text
processed_data/pdf_parsed/glm-ocr/text02.md
processed_data/pdf_parsed/glm-ocr/annual_byd_2024_report.md
```

当前读取优先级在 [config/default.yaml](config/default.yaml)：

```yaml
preprocess:
  pdf_parsed_dir: processed_data/pdf_parsed
  pdf_model_order:
    - glm-ocr
    - mineru2.5-pro
    - paddleocr-vl-1.6
    - pypdf
```

如果没有这些 Markdown，系统会回退到 `pypdf` 直接抽取 PDF 文本。

## 4. Build PageIndex

这一步会统一消费所有材料：

```text
PDF  -> 优先读取第 3 步生成的 Markdown；没有 Markdown 时回退到 pypdf
TXT  -> 直接读取原始 txt
HTML -> 直接解析原始 html
```

构建全部索引：

```powershell
python -m script.build_index
```

按领域或单文档构建：

```powershell
python -m script.build_index --domain insurance
python -m script.build_index --doc-id text02
```

如果新增或替换了 GLM-OCR Markdown，需要强制重建对应索引：

```powershell
python -m script.build_index --doc-id text02 --force
```

当前 PageIndex 主流程不需要单独生成 `parsed_documents.jsonl` 或 `chunks.jsonl`。
这些属于 BM25/chunk 检索实验路线；现在的正式链路是 `script.build_index`
直接生成 `pages.jsonl` 和 `page_index.json`。

## 5. Format Structured Evidence Units

把 `pages.jsonl` 进一步格式化成适合“按选项检索”的结构化证据单元：

```powershell
python -m script.format_structured
```

输出：

```text
processed_data/structured_units.jsonl
```

每条记录包含 `doc_id/domain/title/page/section_path/clause_no/chunk_type/raw_text/search_text/numbers/keywords`。
正式答题时会优先检索这些 evidence units，而不是整页原文；如果文件不存在或没有命中，
系统会自动回退到 PageIndex。

只处理某个领域或文档：

```powershell
python -m script.format_structured --domain financial_contracts
python -m script.format_structured --doc-id text01 --out processed_data/structured_text01.jsonl
```

## 6. Run A Small Test

不调用 Qwen，只预览结构化召回：

```powershell
python -m script.preview_structured_retrieval --qid fc_a_001
```

```powershell
$env:DASHSCOPE_API_KEY="your_api_key"
python -m script.format_structured
python -m script.run_pipeline --qid fin_a_003 --build-missing
```

运行指定问题文件：

```powershell
python -m script.run_pipeline --questions Dataset/questions/group_a/insurance_questions.json --build-missing
```

## 7. Run All Questions & some Questions

```powershell
python -m script.format_structured
python -m script.run_pipeline --limit 3 --build-missing --fresh
```
一次性补建所需索引、回答当前问题目录中的全部题目，并生成 `answer.csv`：

```powershell
$env:DASHSCOPE_API_KEY="your_api_key"
python -m script.run_all
```

输出格式符合线上评测要求：

```csv
qid,answer,prompt_tokens,completion_tokens,total_tokens
summary,,3627557,629,3628186
ins_a_001,B,37201,1,37202
```

`answer.csv` 第一行 `summary` 是全部题目的 Token 汇总。脚本每完成一道题就刷新
`answer.csv` 和 `evidence.json`，中途异常时已完成结果会保留。修复 API 或网络问题后
再次执行同一命令，会自动跳过已有答案继续运行。

如果需要完全重跑：

```powershell
python -m script.run_all --fresh
```

## Notes

- 默认输出路径、模型、PageIndex 参数和 Token 上限都在 [config/default.yaml](config/default.yaml)。
- `processed_data/`、`logs/`、`answer*.csv` 和 `evidence*.json` 是可再生成产物，默认不进 Git。
- 当前降耗配置关闭了 Qwen 思考输出，并限制了不同阶段的最大 completion tokens。
