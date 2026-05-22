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

            def zero_pattern_for_heads(pattern, hook=None, selected_heads=target_heads, **_kwargs):
                pattern[:, selected_heads, :, :] = 0.0
                return pattern

            hooks.append((hook_name, zero_pattern_for_heads))

        return hooks

    def _run_with_optional_ablation(self, prompt: str, ablation: dict[str, Any] | None):
        hooks = self._build_ablation_hooks(ablation)
        if len(hooks) == 0:
            return self.model.run_with_cache(prompt)

        # Some transformer_lens versions do not accept fwd_hooks on run_with_cache.
        # Apply hooks via the model hook context for broader compatibility.
        with self.model.hooks(fwd_hooks=hooks):
            return self.model.run_with_cache(prompt)

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

    def _num_layers(self) -> int:
        """Return the number of transformer layers in the loaded model."""
        try:
            return self.model.cfg.n_layers
        except Exception:
            return 12  # GPT-2 small default

    def analyze(
        self,
        prompt: str,
        layer: int,
        components: list[str],
        graph_type: str | None = None,
        ablation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        logits, cache = self._run_with_optional_ablation(prompt, ablation)
        response: dict[str, Any] = {}
        tokens = self.model.to_str_tokens(prompt)
        attention = cache["pattern", layer][0].tolist()

        next_token_logits = logits[0, -1, :]
        predicted_token_id = next_token_logits.argmax(dim=-1).item()
        response["prediction"] = self.model.to_string(predicted_token_id)

        if "TOKENS" in components:
            response["tokens"] = tokens

        if "ATTENTION_MAP" in components:
            # [heads, queries, keys]
            response["attention"] = attention

        if "ALL_LAYERS_ATTENTION" in components:
            # Return attention patterns for every layer in one pass.
            # Shape per layer: [heads, queries, keys]
            n_layers = self._num_layers()
            response["attentionByLayer"] = [
                cache["pattern", layer_idx][0].tolist()
                for layer_idx in range(n_layers)
            ]
            response["numLayers"] = n_layers

        if "RESIDUAL_STREAM" in components:
            # [pos, d_model]
            response["residual"] = cache["resid_post", layer][0].tolist()

        ablation_metadata = self._ablation_metadata(ablation)
        if ablation_metadata is not None:
            response["ablation"] = ablation_metadata

        if graph_type == "HEAT_MAP":
            response["graph"] = {
                "type": "HEAT_MAP",
                "heatmap": self._build_heatmap(tokens=tokens, attention=attention),
            }

        response["result"] = "Live analyze snapshot returned"
        return response

    def _build_heatmap(self, tokens: list[str], attention: list[list[list[float]]]) -> list[dict[str, Any]]:
        heatmap: list[dict[str, Any]] = []

        for head_index, head_matrix in enumerate(attention):
            for query_index, row in enumerate(head_matrix):
                for key_index, value in enumerate(row):
                    heatmap.append({
                        "head": head_index,
                        "query": f"{query_index}: {tokens[query_index] if query_index < len(tokens) else ''}",
                        "key": f"{key_index}: {tokens[key_index] if key_index < len(tokens) else ''}",
                        "value": float(value),
                    })

        return heatmap

    def status(self) -> dict[str, Any]:
        is_model_loaded = hasattr(self, "model") and self.model is not None
        return {
            # Keep legacy flags for existing callers.
            "isLive": True,
            "isModelLoaded": is_model_loaded,
            # Mirror GraphQL EngineStatus fields for direct backend mapping.
            "modelLoaded": "gpt2-small" if is_model_loaded else None,
            "gpuType": "ModalGPU",
        }
