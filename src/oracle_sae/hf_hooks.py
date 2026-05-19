from __future__ import annotations

from typing import Any


def register_hidden_ablations(model: Any, ablations: list[tuple[int, int, float]]):
    """Register hidden-state dimension edits on common decoder-only HF stacks."""
    layers, final_norm, architecture = _decoder_layers(model)
    by_block: dict[int, list[tuple[int, float]]] = {}
    final_edits: list[tuple[int, float]] = []
    for layer, dimension, value in ablations:
        if layer == len(layers):
            final_edits.append((dimension, value))
            continue
        block_index = layer - 1
        if block_index < 0 or block_index >= len(layers):
            raise ValueError(f"Layer {layer} cannot be ablated through {architecture}")
        by_block.setdefault(block_index, []).append((dimension, value))
    handles = []

    for block_index, edits in by_block.items():

        def hook(_module, _inputs, output, edits=edits):
            return _edit_hidden_output(output, lambda hidden: _apply_dimension_edits(hidden, edits))

        handles.append(layers[block_index].register_forward_hook(hook))

    if final_edits:
        if final_norm is None:
            raise RuntimeError(f"{architecture} does not expose a final normalization module")

        def final_hook(_module, _inputs, output):
            return _edit_hidden_output(output, lambda hidden: _apply_dimension_edits(hidden, final_edits))

        handles.append(final_norm.register_forward_hook(final_hook))
    return _HookGroup(handles)


def register_hidden_steering(model: Any, layer: int, direction: Any, strength: float):
    """Register a residual-direction steering hook on common decoder-only HF stacks."""
    layers, final_norm, architecture = _decoder_layers(model)
    if layer == len(layers):
        if final_norm is None:
            raise RuntimeError(f"{architecture} does not expose a final normalization module")

        def final_hook(_module, _inputs, output):
            return _edit_hidden_output(output, lambda hidden: _add_last_token_direction(hidden, direction, strength))

        return final_norm.register_forward_hook(final_hook)
    block_index = layer - 1
    if block_index < 0 or block_index >= len(layers):
        raise ValueError(f"Layer {layer} cannot be steered through {architecture}")

    def hook(_module, _inputs, output):
        return _edit_hidden_output(output, lambda hidden: _add_last_token_direction(hidden, direction, strength))

    return layers[block_index].register_forward_hook(hook)


class _HookGroup:
    def __init__(self, handles: list[Any]):
        self._handles = handles

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()


def _decoder_layers(model: Any) -> tuple[Any, Any | None, str]:
    candidates = [
        ("transformer.h", _nested_attr(model, "transformer.h"), _nested_attr(model, "transformer.ln_f")),
        ("model.layers", _nested_attr(model, "model.layers"), _nested_attr(model, "model.norm")),
        (
            "language_model.layers",
            _nested_attr(model, "language_model.layers"),
            _nested_attr(model, "language_model.norm"),
        ),
        (
            "model.language_model.layers",
            _nested_attr(model, "model.language_model.layers"),
            _nested_attr(model, "model.language_model.norm"),
        ),
        (
            "model.text_model.layers",
            _nested_attr(model, "model.text_model.layers"),
            _nested_attr(model, "model.text_model.norm"),
        ),
    ]
    for name, layers, final_norm in candidates:
        if layers is not None:
            try:
                if len(layers) > 0:
                    return layers, final_norm, name
            except TypeError:
                continue
    raise RuntimeError(
        "HF intervention hooks need a decoder stack such as transformer.h, model.layers, or language_model.layers"
    )


def _nested_attr(root: Any, path: str) -> Any | None:
    value = root
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _edit_hidden_output(output: Any, edit):
    if isinstance(output, tuple):
        hidden = edit(output[0].clone())
        return (hidden, *output[1:])
    return edit(output.clone())


def _apply_dimension_edits(hidden: Any, edits: list[tuple[int, float]]):
    for dimension, value in edits:
        hidden[..., dimension] = value
    return hidden


def _add_last_token_direction(hidden: Any, direction: Any, strength: float):
    hidden[:, -1, :] = hidden[:, -1, :] + strength * _direction_for_hidden(direction, hidden)
    return hidden


def _direction_for_hidden(direction: Any, hidden: Any):
    return direction.to(device=hidden.device, dtype=hidden.dtype)
