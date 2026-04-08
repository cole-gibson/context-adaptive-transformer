from base.simple_gpt import Transformer
import base.operations.trace_and_patch as tnp
from torch import Tensor
import torch

# Tracing

# brittle one first

def get_special_act(model: Transformer, data: tuple):
    with tnp.special_act_tracer(model) as tracer:
        x, y = data
        with torch.no_grad():
            tracer._model(x)
    return tracer._activations

def get_res_preact(model: Transformer, data: tuple):
    with tnp.res_preact_tracer(model) as tracer:
        x, y = data
        tracer._model(x)
    return tracer._pre_activations

def get_res_act(model: Transformer, data: tuple):
    with tnp.res_act_tracer(model) as tracer:
        x, y = data
        tracer._model(x)
    return tracer._activations

def get_preact(model: Transformer, data: tuple):
    with tnp.preact_tracer(model) as tracer:
        x, y = data
        tracer._model(x)
    return tracer._pre_activations

def get_act(model: Transformer, data: tuple):
    with tnp.act_tracer(model) as tracer:
        x, y = data
        with torch.no_grad():
            tracer._model(x)
    return tracer._activations

# Patching
def patched_res_preact_performance(model: Transformer, layer: Tensor, position: Tensor, patch: Tensor, data: tuple, last_token_loss = False):
    with tnp.res_preact_patching(model, layer, position, patch) as patcher:
        x, y = data
        logits, loss = patcher._model(x, y, last_token_loss = last_token_loss)
    return loss

def patched_res_act_performance(model: Transformer, layer: list, patch: list, data: tuple, last_token_loss = False, reduction = 'mean'):
    with tnp.res_act_patching(model, layer, patch) as patcher:
        x, y = data
        logits, loss = patcher._model(x, y, last_token_loss = last_token_loss, reduction = reduction)
    return logits, loss

def patched_preact_performance(model: Transformer, layer: Tensor, position: Tensor, patch: Tensor, data: tuple):
    with tnp.preact_patching(model, layer, position, patch) as patcher:
        x, y = data
        logits, loss = patcher._model(x, y, last_token_loss = True)
    return loss

def patched_full_preact_performance(model: Transformer, layer: Tensor, patch: Tensor, data: tuple):
    """
    Evaluate performance with patched preactivations at chosen layers.
    """
    with tnp.full_preact_patching(model, layer, patch) as patcher:
        x, y = data
        logits, loss = patcher._model(x, y)
    return loss

def patched_act_performance(model: Transformer, layer: Tensor, position: Tensor, patch: Tensor, data: tuple):
    with tnp.act_patching(model, layer, position, patch) as patcher:
        x, y = data
        logits, loss = patcher._model(x, y, last_token_loss = True)
    return loss