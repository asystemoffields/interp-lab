from oracle_sae.hf_hooks import _direction_for_hidden, register_hidden_ablations, register_hidden_steering


def test_hidden_hooks_find_gemma_style_decoder_stack():
    model = _GemmaStyleModel(layer_count=2)

    group = register_hidden_ablations(model, [(1, 2, 0.0), (2, 3, 0.0)])
    steering = register_hidden_steering(model, 1, _Direction(), 3.0)

    assert len(model.model.layers[0].hooks) == 2
    assert len(model.model.norm.hooks) == 1
    group.remove()
    steering.remove()
    assert all(handle.removed for handle in model.handles)


def test_hidden_hooks_find_gpt2_style_decoder_stack():
    model = _Gpt2StyleModel(layer_count=2)

    group = register_hidden_ablations(model, [(1, 2, 0.0), (2, 3, 0.0)])

    assert len(model.transformer.h[0].hooks) == 1
    assert len(model.transformer.ln_f.hooks) == 1
    group.remove()
    assert all(handle.removed for handle in model.handles)


def test_steering_direction_moves_to_hidden_device_and_dtype():
    direction = _Direction()
    hidden = _Hidden(device="cuda:1", dtype="float16")

    assert _direction_for_hidden(direction, hidden) is direction
    assert direction.to_calls == [{"device": "cuda:1", "dtype": "float16"}]


class _Layer:
    def __init__(self, model):
        self.model = model
        self.hooks = []

    def register_forward_hook(self, hook):
        handle = _Handle()
        self.hooks.append(hook)
        self.model.handles.append(handle)
        return handle


class _Handle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class _Direction:
    def __init__(self):
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append({"args": args, **kwargs} if args else dict(kwargs))
        return self


class _Hidden:
    def __init__(self, *, device, dtype):
        self.device = device
        self.dtype = dtype


class _GemmaStyleModel:
    def __init__(self, layer_count: int):
        self.handles = []
        self.model = _GemmaInner(self, layer_count)


class _GemmaInner:
    def __init__(self, model, layer_count: int):
        self.layers = [_Layer(model) for _ in range(layer_count)]
        self.norm = _Layer(model)


class _Gpt2StyleModel:
    def __init__(self, layer_count: int):
        self.handles = []
        self.transformer = _Gpt2Inner(self, layer_count)


class _Gpt2Inner:
    def __init__(self, model, layer_count: int):
        self.h = [_Layer(model) for _ in range(layer_count)]
        self.ln_f = _Layer(model)
