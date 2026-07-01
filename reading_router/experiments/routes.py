from __future__ import annotations

import re
from dataclasses import dataclass

from agent.qwen_client import DEFAULT_QWEN_MODEL


@dataclass(frozen=True, slots=True)
class ModelRoute:
    name: str
    model: str
    role: str = "custom"
    note: str = ""
    max_tokens: int | None = None
    extra_body: dict[str, object] | None = None


SMALL_QWEN_ROUTES: tuple[ModelRoute, ...] = (
    ModelRoute(
        name="tiny_0_6b",
        model="qwen3-0.6b",
        role="tiny_feasibility_probe",
        note="Smallest Qwen3 open-weight API route candidate; tests whether the prompt survives a very small model.",
        extra_body={"enable_thinking": False},
    ),
    ModelRoute(
        name="small_1_7b",
        model="qwen3-1.7b",
        role="small_feasibility_probe",
        note="Small Qwen3 candidate; useful for checking the lower bound before quality collapses.",
        extra_body={"enable_thinking": False},
    ),
    ModelRoute(
        name="compact_4b",
        model="qwen3-4b",
        role="compact_probe",
        note="Compact Qwen3 candidate; often the first route where structured JSON behavior may become usable.",
        extra_body={"enable_thinking": False},
    ),
    ModelRoute(
        name="small_8b",
        model="qwen3-8b",
        role="small_quality_probe",
        note="Small Qwen3 quality probe; checks whether reading-chain quality is acceptable below plus/max routes.",
        extra_body={"enable_thinking": False},
    ),
    ModelRoute(
        name="moe_a3b",
        model="qwen3-30b-a3b",
        role="small_active_moe_probe",
        note="MoE route with small active parameters; useful if available because it may trade quality for lower active compute.",
        extra_body={"enable_thinking": False},
    ),
    ModelRoute(
        name="turbo",
        model="qwen-turbo",
        role="speed_cost_probe",
        note="Fast hosted route; practical low-cost baseline for compact question-stem reading.",
    ),
    ModelRoute(
        name="turbo_latest",
        model="qwen-turbo-latest",
        role="latest_turbo_probe",
        note="Latest turbo-family alias if exposed by the account; failed routes are recorded without stopping the run.",
    ),
)


READING_MODEL_PRESETS: dict[str, tuple[ModelRoute, ...]] = {
    "repo_default": (
        ModelRoute(
            name="repo_default",
            model=DEFAULT_QWEN_MODEL,
            role="repo_default",
            note="Project Y current default Qwen model; useful as the anchor for local comparisons.",
        ),
    ),
    "reading_small": SMALL_QWEN_ROUTES,
    "reading_core": (
        *SMALL_QWEN_ROUTES,
        ModelRoute(
            name="plus",
            model="qwen-plus",
            role="balanced_baseline",
            note="Practical hosted baseline; compare small routes against this before paying for stronger models.",
        ),
        ModelRoute(
            name="repo_default",
            model=DEFAULT_QWEN_MODEL,
            role="repo_default_anchor",
            note="Current repository default, kept as the anchor against existing Project Y runs.",
        ),
    ),
    "reading_extended": (
        *SMALL_QWEN_ROUTES,
        ModelRoute(
            name="plus",
            model="qwen-plus",
            role="balanced_baseline",
            note="Default practical candidate for batch reading chains if quality is close to stronger models.",
        ),
        ModelRoute(
            name="strong",
            model="qwen-max",
            role="strong_quality_probe",
            note="General strong candidate; catches whether extra capacity improves structure or just burns tokens.",
        ),
        ModelRoute(
            name="repo_default",
            model=DEFAULT_QWEN_MODEL,
            role="repo_default_anchor",
            note="Current repo anchor; compare against existing Project Y Qwen behavior and token scale.",
        ),
        ModelRoute(
            name="manual_latest",
            model="qwen-max-latest",
            role="latest_probe",
            note="Optional newest max-family probe if the account exposes it; failed routes are recorded and do not stop the run.",
        ),
        ModelRoute(
            name="long_context",
            model="qwen-long",
            role="long_context_probe",
            note="Optional long-context probe; mostly useful if later batches include larger question/context payloads.",
        ),
    ),
}


def preset_routes(name: str) -> list[ModelRoute]:
    try:
        return list(READING_MODEL_PRESETS[name])
    except KeyError as exc:
        choices = ", ".join(sorted(READING_MODEL_PRESETS))
        raise ValueError(f"Unknown reading-router preset: {name}. Choices: {choices}") from exc


def parse_model_routes(value: str) -> list[ModelRoute]:
    routes: list[ModelRoute] = []
    for item in value.split(","):
        spec = item.strip()
        if not spec:
            continue
        if "=" in spec:
            name, model = spec.split("=", 1)
        elif ":" in spec:
            name, model = spec.split(":", 1)
        else:
            name = spec
            model = spec
        model = model.strip()
        name = _slug(name.strip() or model)
        routes.append(ModelRoute(name=name, model=model, note="Manual CLI route."))
    if not routes:
        raise ValueError("At least one model route is required.")
    return routes


def resolve_routes(*, preset: str | None, models: str | None) -> list[ModelRoute]:
    if models:
        return parse_model_routes(models)
    return preset_routes(preset or "reading_core")


def _slug(value: str) -> str:
    output = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    return output.strip("_") or "model"
