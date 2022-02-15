from importlib import import_module


def get_model(model_name, model_kwargs):
    module, attr = model_name.rsplit(".", 1)
    model_cls = getattr(import_module("." + module, "model"), attr)
    return model_cls(**model_kwargs)
