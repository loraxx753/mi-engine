from __future__ import annotations

from typing import Any

from mi_observatory.services.model_runtime import ModelRuntime


def parse_ablation_request(data: dict[str, Any], default_layer: int) -> dict[str, Any] | None:
    raw_ablation = data.get("ablation")
    if not isinstance(raw_ablation, dict):
        return None

    mode = str(raw_ablation.get("mode", "zero_pattern"))

    normalized_heads: list[dict[str, int]] = []

    # Shorthand: { ablation: { headIndices: [1, 2] } } on the requested layer.
    raw_head_indices = raw_ablation.get("headIndices")
    if isinstance(raw_head_indices, list):
        for head_index in raw_head_indices:
            normalized_heads.append({
                "layer": default_layer,
                "headIndex": int(head_index),
            })

    # Explicit: { ablation: { heads: [{ layer, headIndex }, ...] } }
    raw_heads = raw_ablation.get("heads")
    if isinstance(raw_heads, list):
        for head in raw_heads:
            if not isinstance(head, dict):
                continue

            head_layer = int(head.get("layer", default_layer))
            head_index = int(head.get("headIndex", -1))
            if head_index >= 0:
                normalized_heads.append({
                    "layer": head_layer,
                    "headIndex": head_index,
                })

    if len(normalized_heads) == 0:
        return None

    return {
        "mode": mode,
        "heads": normalized_heads,
    }


def parse_analyze_request(data: dict[str, Any]) -> tuple[str, int, list[str], str | None, dict[str, Any] | None]:
    prompt = str(data.get("prompt", "Hello"))
    layer = int(data.get("layer", 0))
    raw_components = data.get("components", ["TOKENS"])
    components = [str(component) for component in raw_components]
    raw_graph_type = data.get("graphType")
    graph_type = str(raw_graph_type).upper() if isinstance(raw_graph_type, str) else None
    ablation = parse_ablation_request(data, default_layer=layer)
    return prompt, layer, components, graph_type, ablation


def handle_analyze_request(runtime: ModelRuntime, data: dict[str, Any]) -> dict[str, Any]:
    prompt, layer, components, graph_type, ablation = parse_analyze_request(data)
    return runtime.analyze(
        prompt=prompt,
        layer=layer,
        components=components,
        graph_type=graph_type,
        ablation=ablation,
    )


def handle_status_request(runtime: ModelRuntime) -> dict[str, Any]:
    return runtime.status()
