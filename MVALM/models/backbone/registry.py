import sys

_model_registry = {}


def register_model(fn):
    mod = sys.modules[fn.__module__]

    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    if model_name not in _model_registry.keys():
        _model_registry[model_name] = fn
    else:
        raise ValueError(f'Multiple declaration of the same model {model_name} found.')

    return fn


def list_models():
    return list(_model_registry.keys())


def get_model(model_name: str):
    model_name = model_name.lower().strip()
    if model_name not in _model_registry.keys():
        raise KeyError(f"Unknown model {model_name}. Use on of: {', '.join(_model_registry)}.")
    return _model_registry[model_name]
