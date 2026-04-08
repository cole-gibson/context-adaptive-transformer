from base.simple_gpt import Transformer, Attention, MLP
import torch

res_func_lookup = {
    'Attention': 'sa', 
    'MLP': 'ffwd',
    'Pool': 'sa'
}

# Tracing

# brittle one first

class special_act_tracer(object):
    "Record the residual activations of the GELU module in any MLP in the model. WARNING: Brittle"
    def __init__(self, model: Transformer):
        self._model = model
        self._activations = {}
        self._hooks = []

    def _register_forward_hooks(self):
        def get_activations(i):
            def hook(module, args, output):
                hidden_state = output.cpu()
                self._activations[i] = hidden_state
            
            return hook
        
        for i, module in enumerate(self._model.blocks):
            try:
                res_function = module.ffwd.net[1]
            except AttributeError:
                continue
            
            h = res_function.register_forward_hook(get_activations(i), prepend=False)
            self._hooks.append(h)
    
    def __enter__(self):
        self._register_forward_hooks()

        return self
    
    def __exit__(self, *args, **kwargs):
        for h in self._hooks:
            h.remove()

class res_act_tracer(object):
    "Record the residual activations by layer"
    def __init__(self, model: Transformer):
        self._model = model
        self._activations = {}
        self._hooks = []

    def _register_forward_hooks(self):
        def get_activations(i):
            def hook(module, args, output):
                hidden_state = output
                self._activations[i] = hidden_state
            
            return hook
        
        for i, module in enumerate(self._model.blocks):
            attr = res_func_lookup[module.__class__.__name__]
            res_function = getattr(module, attr)
            h = res_function.register_forward_hook(get_activations(i), prepend=False)
            self._hooks.append(h)
    
    def __enter__(self):
        self._register_forward_hooks()

        return self
    
    def __exit__(self, *args, **kwargs):
        for h in self._hooks:
            h.remove()

class res_preact_tracer(object):
    "Record the residual pre-activations by layer"
    def __init__(self, model: Transformer):
        self._model = model
        self._pre_activations = {}
        self._hooks = []

    def _register_forward_pre_hooks(self):
        def get_pre_activations(i):
            def hook(module, input):
                hidden_state = input
                self._pre_activations[i] = hidden_state[0].cpu()
            
            return hook
        
        for i, module in enumerate(self._model.blocks):
            attr = res_func_lookup[module.__class__]
            res_function = getattr(module, attr)
            h = res_function.register_forward_pre_hook(get_pre_activations(i))
            self._hooks.append(h)
    
    def __enter__(self):
        self._register_forward_pre_hooks()

        return self
    
    def __exit__(self, *args, **kwargs):
        for h in self._hooks:
            h.remove()

class act_tracer(object):
    "Record the activations by layer"
    def __init__(self, model: Transformer):
        self._model = model
        self._activations = {}
        self._hooks = []

    def _register_forward_hooks(self):
        def get_activations(i):
            def hook(module, args, output):
                hidden_state = output.cpu()
                self._activations[i] = hidden_state
            
            return hook
        
        for i, module in enumerate(self._model.blocks):
            h = module.register_forward_hook(get_activations(i), prepend=False)
            self._hooks.append(h)
    
    def __enter__(self):
        self._register_forward_hooks()

        return self
    
    def __exit__(self, *args, **kwargs):
        for h in self._hooks:
            h.remove()

class preact_tracer(object):
    "Record the pre-activations by layer"
    def __init__(self, model: Transformer):
        self._model = model
        self._pre_activations = {}
        self._hooks = []

    def _register_forward_pre_hooks(self):
        def get_pre_activations(i):
            def hook(module, input):
                hidden_state = input
                self._pre_activations[i] = hidden_state[0].cpu()
            
            return hook
        
        for i, module in enumerate(self._model.blocks):
            h = module.register_forward_pre_hook(get_pre_activations(i))
            self._hooks.append(h)
    
    def __enter__(self):
        self._register_forward_pre_hooks()

        return self
    
    def __exit__(self, *args, **kwargs):
        for h in self._hooks:
            h.remove()

# Patching

class res_preact_patching():
    "Patch the residual pre-activation of specific positions at specific layers"
    def __init__(self, model: Transformer, layer: torch.Tensor, position: torch.Tensor, patch: torch.Tensor):
        self._model = model
        self._hooks = []
        self._layer = layer         # (B, 1)
        self._position = position   # (B, 1)
        self._patch = patch         # (B, n_embd)

    def _register_forward_pre_hooks(self):
        def patch_activation(pos, patch_vec):
            def hook(module, input):               # Might be brittle, documentation is unclear
                input[0][:, pos, :] = patch_vec
                return input
            return hook
        
        for i, patch_vec in enumerate(self._patch):
            pos = self._position[i]
            l = self._layer[i]
            module = self._model.blocks[l]

            attr = res_func_lookup[module.__class__]
            res_function = getattr(module, attr)
            h = res_function.register_forward_pre_hook(patch_activation(pos, patch_vec))
            self._hooks.append(h)

    def __enter__(self):
        self._register_forward_pre_hooks()
        
        return self
    
    def __exit__(self, *args, **kwargs):
        for h in self._hooks:
            h.remove()

class res_act_patching():
    "Patch the residual activation at specific layers"
    def __init__(self, model: Transformer, layer: list, patch: list):
        self._model = model
        self._hooks = []
        self._layer = layer         # length n list of layer indices
        self._patch = patch         # length n list of (B, T, n_embd)

    def _register_forward_hooks(self):
        def patch_activation(patch_vec):
            def hook(module, args, output):               # Might be brittle, documentation is unclear
                return patch_vec
            return hook
        
        for i, patch_vec in enumerate(self._patch):
            l = self._layer[i]
            module = self._model.blocks[l]

            attr = res_func_lookup[module.__class__.__name__]
            res_function = getattr(module, attr)
            h = res_function.register_forward_hook(patch_activation(patch_vec))
            self._hooks.append(h)

    def __enter__(self):
        self._register_forward_hooks()
        
        return self
    
    def __exit__(self, *args, **kwargs):
        for h in self._hooks:
            h.remove()

class preact_patching():
    "Patch the pre_activation of specific positions at specific layers"
    def __init__(self, model: Transformer, layer: torch.Tensor, position: torch.Tensor, patch: torch.Tensor):
        self._model = model
        self._hooks = []
        self._layer = layer         # (B, 1)
        self._position = position   # (B, 1)
        self._patch = patch         # (B, n_embd)

    def _register_forward_pre_hooks(self):
        def patch_activation(pos, patch_vec):
            def hook(module, input):               # Might be brittle, documentation is unclear
                input[0][:, pos, :] = patch_vec
                return input
            return hook
        
        for i, patch_vec in enumerate(self._patch):
            pos = self._position[i]
            l = self._layer[i]
            module = self._model.blocks[l]

            h = module.register_forward_pre_hook(patch_activation(pos, patch_vec))
            self._hooks.append(h)

    def __enter__(self):
        self._register_forward_pre_hooks()
        
        return self
    
    def __exit__(self, *args, **kwargs):
        for h in self._hooks:
            h.remove()

class full_preact_patching():
    "Patch the pre_activation at specific layers"
    def __init__(self, model: Transformer, layer: torch.Tensor, patch: torch.Tensor):
        self._model = model
        self._hooks = []
        self._layer = layer         # (B, 1)
        self._patch = patch         # (B, T, n_embd)

    def _register_forward_pre_hooks(self):
        def patch_activation(patch_vec):
            def hook(module, input):               # Might be brittle, documentation is unclear
                return (patch_vec, )
            return hook
        
        for i, patch_vec in enumerate(self._patch):
            l = self._layer[i]
            module = self._model.blocks[l]

            h = module.register_forward_pre_hook(patch_activation(patch_vec))
            self._hooks.append(h)

    def __enter__(self):
        self._register_forward_pre_hooks()
        
        return self
    
    def __exit__(self, *args, **kwargs):
        for h in self._hooks:
            h.remove()

class act_patching():
    "Patch the activation of specific positions at specific layers"
    def __init__(self, model: Transformer, layer: torch.Tensor, position: torch.Tensor, patch: torch.Tensor):
        self._model = model
        self._hooks = []
        self._layer = layer         # (B, 1)
        self._position = position   # (B, 1)
        self._patch = patch         # (B, n_embd)

    def _register_forward_hooks(self):
        def patch_activation(pos, patch_vec):
            def hook(module, args, output):               # Might be brittle, documentation is unclear
                output[:, pos, :] = patch_vec
                return output
            return hook
        
        for i, patch_vec in enumerate(self._patch):
            pos = self._position[i]
            l = self._layer[i]
            module = self._model.blocks[l]

            h = module.register_forward_hook(patch_activation(pos, patch_vec))
            self._hooks.append(h)

    def __enter__(self):
        self._register_forward_hooks()
        
        return self
    
    def __exit__(self, *args, **kwargs):
        for h in self._hooks:
            h.remove()