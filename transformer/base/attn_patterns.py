import torch
from torch import Tensor
from base.simple_gpt import Transformer
from base.utils import safe_getattr

class attn_pattern_tracer(object):
    "Record the attention patterns of each attention layer"
    def __init__(self, model: Transformer):
        self._model = model
        self._patterns = {}
        self._hooks = []

    def _register_forward_pre_hooks(self):
        def get_activations(i):
            def hook(module, input):
                attn_pattern = input[0]
                self._patterns[i] = attn_pattern
            
            return hook
        
        for i, module in enumerate(self._model.blocks):
            if 'Attention' not in module.__class__.__name__:
                try:
                    if module.attention == True:
                        pass
                    else:
                        continue
                except AttributeError:
                    continue
            try:
                res_function = getattr(getattr(module, 'sa'), 'dropout')
            except AttributeError:
                res_function = getattr(module, 'dropout')
            h = res_function.register_forward_pre_hook(get_activations(i), prepend=False)
            self._hooks.append(h)
    
    def __enter__(self):
        self._register_forward_pre_hooks()

        return self
    
    def __exit__(self, *args, **kwargs):
        for h in self._hooks:
            h.remove()

def get_attn_pattern(model: Transformer, data: tuple, to_device = None):
    with attn_pattern_tracer(model) as tracer:
        if safe_getattr(model.config, 'inject_tasks'):
            x, y, tasks = data
            tracer._model(x, y, tasks)
        else:
            x, y = data
            tracer._model(x)
    out = {}
    for key, value in tracer._patterns.items():
        if to_device is not None:
            out[key] = value.to(to_device)
        else:
            out[key] = value.detach()
    return out

# get attention pattern gradient
class attn_pattern_grad_tracer(object):
    "Record the attention patterns of each attention layer"
    def __init__(self, model: Transformer):
        self._model = model
        self._grads = {}
        self._hooks = []

    def _register_forward_hooks(self):
        def get_gradient(i: int):
            def hook(module, inputs, outputs):
                wei = module.wei
                wei.retain_grad()
                self._grads[i] = wei
            
            return hook
        
        for i, module in enumerate(self._model.blocks):
            if 'Attention' not in module.__class__.__name__:
                try:
                    if module.attention == True:
                        pass
                    else:
                        continue
                except AttributeError:
                    continue
            try:
                submodule = getattr(module, 'sa')
            except AttributeError:
                submodule = module
            h = submodule.register_forward_hook(get_gradient(i))
            self._hooks.append(h)
    
    def __enter__(self):
        self._register_forward_hooks()

        return self
    
    def __exit__(self, *args, **kwargs):
        for h in self._hooks:
            h.remove()

def get_attn_grads(model: Transformer, data: tuple, to_device = None):
    with attn_pattern_grad_tracer(model) as tracer:
        if safe_getattr(model.config, 'inject_tasks'):
            x, y, tasks = data
            logits, loss = tracer._model(x, y, tasks)
        else:
            x, y = data
            logits, loss = tracer._model(x, y)
        loss.backward()

    out = {}
    for key, wei in tracer._grads.items():
        g = wei.grad
        if g is None:
            continue  # or raise / warn if you expect a grad

        if to_device is not None:
            out[key] = g.detach().to(to_device)
        else:
            out[key] = g.detach().clone()
    return out

class attn_pattern_patcher(object):
    "Patch the attention patterns of each attention layer"
    def __init__(self, model: Transformer, layer: torch.Tensor, patch: torch.Tensor):
        self._model = model
        self._hooks = []
        self._layer = layer         # (B, 1)
        self._patch = patch        # (B, n_embd)

    def _register_forward_pre_hooks(self):
        def patch_activation(patch_vec):
            def hook(module, input):
                device = input[0].device
                input = (patch_vec.to(device),)
                return input
            return hook
        
        for i, patch_vec in enumerate(self._patch):
            l = self._layer[i]
            module = self._model.blocks[l]
            location = getattr(getattr(module, 'sa'), 'dropout')
            
            h = location.register_forward_pre_hook(patch_activation(patch_vec))
            self._hooks.append(h)
    
    def __enter__(self):
        self._register_forward_pre_hooks()

        return self
    
    def __exit__(self, *args, **kwargs):
        for h in self._hooks:
            h.remove()

def patched_attn_pattern_perf(model: Transformer, layer: Tensor, patch: Tensor, data: tuple):
    with attn_pattern_patcher(model, layer, patch) as patcher:
        x, y = data
        logits, loss = patcher._model(x, y, last_token_loss = True)
    return loss