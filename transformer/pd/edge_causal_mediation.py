import torch.nn.functional as F
import pandas as pd
import torch
import base.utils as u
from pd.generic_experiment import BaseRunner
from typing import Callable
from base.seed import set_seed
from contextlib import contextmanager

from base.simple_gpt import Transformer, Head

res_func_lookup = {'Attention': 'sa', 'MLP': 'ffwd'}

# Tracing

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
                self._pre_activations[i] = hidden_state[0]
            
            return hook
        
        for i, module in enumerate(self._model.blocks):
            attr = res_func_lookup[module.__class__.__name__]
            res_function = getattr(module, attr)
            h = res_function.register_forward_pre_hook(get_pre_activations(i))
            self._hooks.append(h)
    
    def __enter__(self):
        self._register_forward_pre_hooks()

        return self
    
    def __exit__(self, *args, **kwargs):
        for h in self._hooks:
            h.remove()

def get_res_preact(model: Transformer, data: tuple):
    with res_preact_tracer(model) as tracer:
        x, y = data
        tracer._model(x)
    return tracer._pre_activations

def get_res_act(model: Transformer, data: tuple):
    with res_act_tracer(model) as tracer:
        x, y = data
        tracer._model(x)
    return tracer._activations

# hooking into key and query calculation
@contextmanager
def override_kq_inputs(head: Head, k_in=None, q_in=None):
    """
    Temporarily override the *inputs* to head.key and head.query.
    k_in, q_in should have the same shape as x: (B, T, C).
    """
    hooks = []

    if k_in is not None:
        def _k_pre_hook(module, inputs):
            # inputs is a tuple (x,)
            return (k_in,)
        hooks.append(head.key.register_forward_pre_hook(_k_pre_hook))

    if q_in is not None:
        def _q_pre_hook(module, inputs):
            return (q_in,)
        hooks.append(head.query.register_forward_pre_hook(_q_pre_hook))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()

def run_head_with_custom_kq(head: Head, x, k_in=None, q_in=None):
    """
    Run head(x) but feed k_in / q_in into head.key/query instead of x.
    """
    with override_kq_inputs(head, k_in=k_in, q_in=q_in):
        return head(x)

def ln_transfer(x1, x2, weight=None, bias=None, eps=1e-5):
    """
    Apply LayerNorm induced by x1 to x2.
    If weight/bias are provided, they must match x1.shape[-1].
    """
    d = x1.shape[-1]
    _, mean, rvar = torch.ops.aten.native_layer_norm(x1, (d,), None, None, eps)

    return (x2-mean)*rvar

def ablate_edges(model, data: tuple, edge_list: list, ablation: str = 'mean'):
    df = {
        'loss': [],
        'kl': [],
        'kl_norm': [],
        'ablated_edge': []
        # 'res': []
    }

    x, y = data

    baseline_logits, baseline_loss = model(x, y) # compute unperturbed performance
    kl = u.KL(baseline_logits, baseline_logits)
    norm = u.KL(baseline_logits, torch.zeros_like(baseline_logits))
    df['loss'].append(baseline_loss.item())
    df['kl'].append(kl)
    df['kl_norm'].append(kl/norm)
    df['ablated_edge'].append(())

    tok_embd = model.token_embedding_table(x)
    B, T, C = tok_embd.shape

    # 0:        token                           0
    # 1, 0:     attention 1_Q    dest           1
    # 1, 1:     attention 1_K    dest           2
    # 1, 2:     attention 1_V    source         3
    # 2:        MLP                             4
    # 3, 0:     attention 2_Q    dest           5
    # 3, 1:     attention 2_K    dest           6
    # 3, 2, 0:  attention 2_V_0  ablate source  7
    # 3, 2, 1:  attention 2_V_1  ablate source  8
    # 3, 2, 2:  attention 2_V_2  ablate source  9
    # 3, 2:     attention 2_V    source         10
    # 4:        MLP                             11
    # 5:        unembed                         12

    for e in edge_list:
        src, dest = e    # edge i -> j

        assert dest not in (3, 7, 8, 9, 10)
        assert src not in (1, 2, 5, 6)

        res = {
            0: tok_embd
        }

        value_channels = {}
        lookup = {7: 0, 8: 3, 9: 4} # convert from edge to value_channel
        
        # generate model residual stream up to unembedding
        # iteration over source blocks in order of execution
        for bidx in [3, 4, 7, 8, 9, 10, 11]:
            # map block index to block
            if bidx == 3:
                block = model.blocks[0]
            elif bidx == 4:
                block = model.blocks[1]
            elif 4 < bidx < 11:
                block = model.blocks[2]
            elif bidx == 11:
                block = model.blocks[3]
            
            # compute unperturbed input to block
            input = sum(res.values())
            clean = input.clone()

            if bidx == dest:  # current module is destination
                # make ablation to input
                if src not in (7, 8, 9):    # not a value channel source
                    # normal ablation from res
                    if ablation == 'mean':
                        input += res[src].mean(0, keepdim=True) - res[src]
                    elif ablation == 'zero':
                        input += -1*res[src]
                else:
                    # ablate using value channel
                    if ablation == 'mean':
                        input += value_channels[src].mean(0, keepdim=True) - value_channels[src]
                    elif ablation == 'zero':
                        input += -1*value_channels[src]
                            
            # modify key and query inputs
            # does NOT handle simultaneous edge ablation to the keys and queries
            if (dest==1 and bidx == 3) or (dest==5 and 6 < bidx < 11):  # ablations to keys
                if ablation == 'mean':
                    input += res[src].mean(0, keepdim=True) - res[src]
                elif ablation == 'zero':
                    input += -1*res[src]
                k_in = clean
                q_in = input    # only query input perturbed
                v_in = clean
            elif (dest==2 and bidx == 3) or (dest==6 and 6 < bidx < 11):    # ablations to queries
                if ablation == 'mean':
                    input += res[src].mean(0, keepdim=True) - res[src]
                elif ablation == 'zero':
                    input += -1*res[src]
                k_in = input    # only key input perturbed
                q_in = clean
                v_in = clean
            else:
                # unperturbed
                k_in = input    # CHECK THAT THIS SHOULD NOT BE CLEAN (answer: guaranteed by construction to be equal to clean)
                q_in = input
                v_in = input
            
            # attention modules
            if hasattr(block, 'sa'):
                module = block.sa
                if bidx == 3:
                    # record value channel to res directly, since there is only one channel
                    res[bidx] = run_head_with_custom_kq(module, ln_transfer(clean, v_in), ln_transfer(clean, k_in), ln_transfer(clean, q_in))
                elif bidx in (7, 8, 9):
                    # record value channels separately as these are only used for later ablations
                    value_channels[bidx] = run_head_with_custom_kq(module, ln_transfer(clean, res[lookup[bidx]]), ln_transfer(clean, k_in), ln_transfer(clean, q_in))
                elif bidx == 10:
                    # record full output of all Att2 value channels
                    res[bidx] = run_head_with_custom_kq(module, ln_transfer(clean, v_in), ln_transfer(clean, k_in), ln_transfer(clean, q_in))
            
            # MLP modules
            elif hasattr(block, 'ffwd'):
                res[bidx] = block.ffwd(ln_transfer(clean, input))
        
        # ablations of edges to unembedding (.,12)
        input = sum(res.values())
        clean = input.clone()
        # if destination is unembedding, make ablation
        if dest == 12:
            if src not in (7, 8, 9):
                if ablation == 'mean':
                    input += res[src].mean(0, keepdim=True) - res[src]
                elif ablation == 'zero':
                    input += -1*res[src]
            else:
                if ablation == 'mean':
                    input += value_channels[src].mean(0, keepdim=True) - value_channels[src]
                elif ablation == 'zero':
                    input += -1*value_channels[src]
        
        # evaluate model
        logits = model.lm_head(ln_transfer(clean, input))
        logits = logits.reshape(B*T, -1)
        targets = y.reshape(B*T)
        loss = F.cross_entropy(logits, targets)
        kl = u.KL(baseline_logits, logits)  # compute KL to control
        df['loss'].append(loss.item())
        df['kl'].append(kl)
        df['kl_norm'].append(kl/norm)
        df['ablated_edge'].append(e)

    return pd.DataFrame(df)

class CausalMediation(BaseRunner):
    def __init__(
        self,
        base_dir: str,
        get_model,
        get_task_model,
        batch_size: int,
        repeat: int,
        n_states: int,
        state_interval: int,
        data_dir_template: Callable
    ):
        super().__init__(
            base_dir, get_model, get_task_model,
            n_states, state_interval,
            data_dir_template
        )
        self.batch_size = batch_size
        self.repeat = repeat
        # storage for results
        self.model_out = []

    def write_model_out(self, df):
            df['seed'] = self.seed
            df[self.param_name] = self.param_value
            df['n'] = self.n
            df['t'] = self.t
            self.model_out.append(df)

    def chunk_and_truncate(self, T, idx):
        start = self.batch_size * idx
        end = start + self.batch_size
        self.x_train_trunc = self.x_train[start:end, -T:]
        self.y_train_trunc = self.y_train[start:end, -T:]

    def seed_step(self, seed):
        self.seed = seed

    def param_step(self, param_value):
        self.param_value = param_value

    def n_step(self, n):
        self.n = n

        # generate sequences once
        if n == min(self.nrange):
            self.config = u.load_config_and_task_pool(self.get_data_dir())
            set_seed(self.config.training.seed+1)
            self.task_model = self.get_task_model(self.config)
            self.n_sequences = self.repeat * self.batch_size
            self.x_train, self.y_train = self.task_model.get_batch(
                self.n_sequences, int(max(self.nrange))
            )
        else:
            cfg_path = self.get_data_dir() / "config.yaml"
            self.config = u.load_config(str(cfg_path))

        self.model = self.get_model(self.config)

    def t_step(self, idx):
        self.idx = idx
        state_path = self.get_data_dir() / "state" / f"{idx}.pt"
        try:
            state = torch.load(str(state_path), map_location=self.config.model.device)
        except FileNotFoundError:
            return True
        self.t = state["iter"]
        self.model.load_state_dict(state["state"])
        self.model.eval()

        for i in range(self.repeat):
            self.chunk_and_truncate(self.config.training.context_len, i)
            df = ablate_edges(self.model, (self.x_train_trunc, self.y_train_trunc), 
                   [(0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (0, 11), (0, 12), 
                    (3, 4), (3, 5), (3, 6), (3, 11), (3, 12), 
                    (4, 5), (4, 6), (4, 11), (4, 12), 
                    (7, 11), (7, 12), 
                    (8, 11), (8, 12), 
                    (9, 11), (9, 12), 
                    (11, 12)],
                ablation = 'mean')

        self.write_model_out(
            df
        )
    
    def _clean_output(self):
        self.model_out = pd.concat(self.model_out, ignore_index = True)
        self.model_out.drop_duplicates(subset = ['seed', 'n', self.param_name, 't', 'ablated_edge'], inplace = True)    # duplicate 't' entries may result from restarts (these are harmless)

    def _save_output(self):
        self._clean_output()
        self.model_out.to_csv(str(self.base_dir / "edge_causal.csv"), index = False)

    def _save_output_min(self):
        self._clean_output()
        self.model_out.to_csv(str(self.base_dir / f"{self.get_data_dir()}_edge_causal.csv"), index = False)