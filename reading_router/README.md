# Reading Router

The retained routing rules and five final prompts live in
[`final_prompts.py`](final_prompts.py). A human-readable copy is in
[`router_spec.md`](router_spec.md).

Run the retained final router:

```bash
PYTHONPATH=. python -m agent.run_final_reading_router \
  --input data/questions.jsonl \
  --out-dir runs/reading_router_final/qwen37max_no_thinking_all_20260701 \
  --model qwen3.7-max \
  --batch-size 10
```

Outputs:

- `model_outputs.jsonl`: lightweight handoff rows with `qid` plus the route-specific model output fields.
- `reading_routes.jsonl`: debug/audit rows with original question metadata, parsed router output, and raw batch text.
- `route_summary.csv`: token and validity summary by retained route.
- `run_meta.json`: model, batch size, thinking flag, and route counts.

Exploratory code lives in `reading_router/experiments/`. The older reading-chain
model-sweep CLI is still available through `agent.run_reading_router`.

Run:

```bash
PYTHONPATH=. python -m agent.run_reading_router \
  --input eval/fake_gold/gptpro_reading_trace_request.json \
  --out-dir runs/reading_router \
  --limit 10 \
  --batch-size 5
```

Default preset: `reading_core`

- `tiny_0_6b=qwen3-0.6b`: smallest feasibility probe.
- `small_1_7b=qwen3-1.7b`: small lower-bound probe.
- `compact_4b=qwen3-4b`: compact structured-output probe.
- `small_8b=qwen3-8b`: small quality probe.
- `moe_a3b=qwen3-30b-a3b`: small-active-parameter MoE probe.
- `turbo=qwen-turbo`: hosted speed/cost probe.
- `turbo_latest=qwen-turbo-latest`: optional latest turbo alias.
- `plus=qwen-plus`: practical quality/cost baseline.
- `repo_default=qwen3.7-max`: Project Y default-model anchor.

The default set is now small-model-heavy:

- Use `tiny_0_6b` / `small_1_7b` to find the failure floor.
- Use `compact_4b` / `small_8b` to see whether the prompt is enough for usable structured reading chains.
- Use `moe_a3b` if available to test small active-parameter behavior.
- Use `turbo` / `turbo_latest` as hosted low-cost references.
- Use `plus` as the likely production candidate if small routes are too brittle.
- Use `repo_default` to stay comparable with current Project Y Qwen runs.

Small-only sweep:

```bash
PYTHONPATH=. python -m agent.run_reading_router \
  --preset reading_small \
  --limit 5 \
  --batch-size 1
```

Optional broader sweep:

```bash
PYTHONPATH=. python -m agent.run_reading_router \
  --preset reading_extended \
  --limit 5 \
  --batch-size 1
```

`reading_extended` adds:

- `strong=qwen-max`: stronger hosted quality probe.
- `manual_latest=qwen-max-latest`: only useful if the account exposes that alias.
- `long_context=qwen-long`: useful later if the reading input grows beyond compact question stems.

Failed model routes are written as invalid rows with `request_failed`; other routes continue, and token totals are still printed for successful calls.

Qwen3 open-weight routes in this preset send `enable_thinking=false`, which is required for this non-streaming JSON experiment path.

Manual model override:

```bash
PYTHONPATH=. python -m agent.run_reading_router \
  --models tiny=qwen3-0.6b,small=qwen3-4b,turbo=qwen-turbo \
  --limit 10
```

Outputs:

- `reading_chains.jsonl`: one row per model/question with parsed fields when available, `raw_content`, loose `schema_issues`, and token usage.
- `token_summary.csv`: per-model prompt/completion/total token summary.

`valid=true` means the model request returned. It does not require `reading_summary` or any other schema field. Missing or oddly shaped fields are recorded in `schema_issues`, while the original model text is kept in `raw_content`.
