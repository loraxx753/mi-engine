from __future__ import annotations

from typing import Any


def _group_heads_by_layer(ablation: dict[str, Any]) -> dict[int, set[int]]:
    grouped: dict[int, set[int]] = {}
    for item in ablation.get("heads", []):
        if not isinstance(item, dict):
            continue

        layer = int(item.get("layer", -1))
        head_index = int(item.get("headIndex", -1))
        if layer < 0 or head_index < 0:
            continue

        grouped.setdefault(layer, set()).add(head_index)

    return grouped


class ModelRuntime:
    """Owns model lifecycle + raw tensor extraction logic."""

    def __init__(self) -> None:
        self.model = None

    def load_model(self) -> None:
        from transformer_lens import HookedTransformer

        print("Systems Online: Loading GPT-2 Small...")
        self.model = HookedTransformer.from_pretrained("gpt2-small")

    def _build_ablation_hooks(self, ablation: dict[str, Any] | None):
        if not isinstance(ablation, dict):
            return []

        mode = str(ablation.get("mode", "zero_pattern"))
        if mode != "zero_pattern":
            raise ValueError(f"Unsupported ablation mode: {mode}")

        from transformer_lens import utils

        grouped_heads = _group_heads_by_layer(ablation)
        hooks = []

        for layer, head_set in grouped_heads.items():
            target_heads = sorted(head_set)
            hook_name = utils.get_act_name("pattern", layer)

            def zero_pattern_for_heads(pattern, _hook, selected_heads=target_heads):
                pattern[:, selected_heads, :, :] = 0.0
                return pattern

            hooks.append((hook_name, zero_pattern_for_heads))

        return hooks

    def _run_with_optional_ablation(self, prompt: str, ablation: dict[str, Any] | None):
        hooks = self._build_ablation_hooks(ablation)
        if len(hooks) == 0:
            return self.model.run_with_cache(prompt)

        return self.model.run_with_cache(prompt, fwd_hooks=hooks)

    def _ablation_metadata(self, ablation: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(ablation, dict):
            return None

        heads = [
            {
                "layer": int(item["layer"]),
                "headIndex": int(item["headIndex"]),
            }
            for item in ablation.get("heads", [])
            if isinstance(item, dict) and "layer" in item and "headIndex" in item
        ]

        if len(heads) == 0:
            return None

        return {
            "mode": str(ablation.get("mode", "zero_pattern")),
            "heads": heads,
            "applied": True,
        }

    def analyze(
        self,
        prompt: str,
        layer: int,
        components: list[str],
        ablation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _logits, cache = self._run_with_optional_ablation(prompt, ablation)
        response: dict[str, Any] = {}

        if "TOKENS" in components:
            response["tokens"] = self.model.to_str_tokens(prompt)

        if "ATTENTION_MAP" in components:
            # [heads, queries, keys]
            response["attention"] = cache["pattern", layer][0].tolist()

        if "RESIDUAL_STREAM" in components:
            # [pos, d_model]
            response["residual"] = cache["resid_post", layer][0].tolist()

        ablation_metadata = self._ablation_metadata(ablation)
        if ablation_metadata is not None:
            response["ablation"] = ablation_metadata

        response["result"] = "Live analyze snapshot returned"
        return response

    def layer_snapshot(
        self,
        prompt: str,
        layer: int,
        ablation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _logits, cache = self._run_with_optional_ablation(prompt, ablation)

        tokens = self.model.to_str_tokens(prompt)
        attention = cache["pattern", layer][0].tolist()

        heads = [
            {
                "headIndex": head_index,
                "matrix": matrix,
            }
            for head_index, matrix in enumerate(attention)
        ]

        response = {
            "layer": layer,
            "metadata": {
                "tokens": tokens,
                "totalHeads": len(heads),
            },
            "heads": heads,
            "result": "Live layer snapshot returned",
            "status": "Live layer snapshot returned",
            "device": "LiveEngine",
        }

        ablation_metadata = self._ablation_metadata(ablation)
        if ablation_metadata is not None:
            response["ablation"] = ablation_metadata

        return response

    def status(self) -> dict[str, bool]:
        return {"isLive": True, "isModelLoaded": hasattr(self, "model") and self.model is not None}
