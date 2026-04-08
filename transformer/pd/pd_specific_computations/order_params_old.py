import sys
if '/home/cg5763/work' not in sys.path:
    sys.path.insert(0, '/home/cg5763/work')

import torch
import base.utils as u
from math import floor
import time
from base.seed import set_seed
from base.attn_patterns import get_attn_pattern

@torch.jit.script
def beta_mask(x: torch.Tensor) -> torch.Tensor:
    device = x.device
    B, T = x.shape
    x = x.unsqueeze(-1).expand(B, T, T).transpose(-1, -2)
    last_token = x.transpose(-1, -2)
    mask = (x == last_token).int()
    mask = torch.roll(mask, 1, -1)
    mask[..., 0] = 0
    mask = mask.masked_fill(torch.tril(torch.ones_like(mask, device = device)) == 0, 0).int()

    return mask

@torch.jit.script
def get_pta(attn_pat: torch.Tensor, x: torch.Tensor):
    """
    Compute mean previous token attention per sequence for all N tokens.

    Args:
    attn_pat: torch.Tensor, shape (B, T, T)
        Attention pattern.
    n: int
        Maximum number of tokens in the sequence.

    Returns:
    pta: torch.Tensor, shape (B,)
    """
    n = x.shape[-1]
    prev_wei = attn_pat.diagonal(offset = -1, dim1 = -2, dim2 = -1) # (B, T-1)
    pta = prev_wei.mean(-1) # (B, ) attention to previous token

    return pta

@torch.jit.script
def get_iha(attn_pat: torch.Tensor, x: torch.Tensor):
    """
    Compute mean induction head attention per sequence for all N tokens.

    Induction head attention is the attention to tokens that followed previous occurences of the current token in the sequence.

    Args:
    attn_pat: torch.Tensor, shape (B, T, T)
        Attention pattern.
    x: torch.Tensor, shape (B, T)
        Corresponding sequence of tokens.
    n: int
        Maximum number of tokens in the sequence.

    Returns:
    iha: torch.Tensor, shape (B,)
    """
    n = x.shape[-1]
    mask = beta_mask(x)

    ind_wei = (mask[:, -1, :] * attn_pat[:, -1, :]).sum(-1) # (B, ) total weight to previous occurences
    m = mask[:, -1, :].sum(-1) # (B, ) number of previous occurences
    iha = ind_wei/m

    return ind_wei, iha, m

class OrderParams():
    @torch.no_grad()
    def __init__(self, dir: str, get_model, TaskModel, batch_size: int, repeat: int, random_seed: int, n_states: int, state_interval: int):
        self.n_states = n_states
        self.state_interval = state_interval
        self.TaskModel = TaskModel
        self.get_model = get_model
        self.batch_size = batch_size
        self.repeat = repeat
        self.dir = dir
        self.random_seed = random_seed
        exp_config = u.load_config(f'{dir}/exp_config.yaml')
        try:
            exp_config.lmin
            self.nrange = torch.arange(exp_config.lmin, exp_config.lmax+exp_config.lstep, exp_config.lstep)
        except AttributeError:
            self.nrange = exp_config.nrange
        self.krange = exp_config.krange
        set_seed(self.random_seed)

        self.model_out = {
            'name': [],
            'k': [],
            'n': [],
            's': [],
            'idx': [],
            'pta': [],
            'iha': [],
            'iha_tot': [],
            'm': []
        }
    
    def write_model_out(self, name: str, pta: float, iha: float, iha_tot: float, m: float):
        self.model_out['name'].append(name)
        self.model_out['k'].append(self.k)
        self.model_out['n'].append(self.n)
        self.model_out['s'].append(self.s)
        self.model_out['idx'].append(self.idx)
        self.model_out['pta'].append(pta)
        self.model_out['iha'].append(iha)
        self.model_out['iha_tot'].append(iha_tot)
        self.model_out['m'].append(m)
    
    def chunk_and_truncate(self, T, idx):
        self.x_train_trunc = self.x_train[self.batch_size*idx:self.batch_size*(idx + 1), -T:]
        self.y_train_trunc = self.y_train[self.batch_size*idx:self.batch_size*(idx + 1), -T:]

    def k_step(self, k):
        self.k = k
        base_config = u.load_config_and_task_pool(f'{self.dir}/data/{self.k}_{min(self.nrange)}')
        task_model = self.TaskModel(base_config.task)
        self.task_model = task_model
        self.n_sequences = self.repeat*self.batch_size
        self.x_train, self.y_train = task_model.get_batch(self.n_sequences, floor(max(self.nrange)))

    def n_step(self, n):
        self.n = n
        self.config = u.load_config(f'{self.dir}/data/{self.k}_{self.n}/config.yaml')
        model = self.get_model(self.config)
        self.model = model
    
    def s_step(self, idx):
        self.idx = idx
        try:
            self.state = torch.load(f'{self.dir}/data/{self.k}_{self.n}/state/{idx}.pt', map_location = self.config.model.device)
        except FileNotFoundError:
            return True
        self.s = self.state['iter']
        self.model.load_state_dict(self.state['state'])
        self.model.eval()

        rep_pta = []
        rep_iha = []
        rep_iha_tot = []
        rep_m = []
        for i in range(self.repeat):
            self.chunk_and_truncate(self.config.training.context_len, i)
            attn_pats = get_attn_pattern(self.model, (self.x_train_trunc, None))
            pta = get_pta(attn_pats[0], self.x_train_trunc)
            try:
                iha_tot, iha, m = get_iha(attn_pats[2], self.x_train_trunc)
            except KeyError:
                iha_tot, iha, m = torch.zeros(1), torch.zeros(1), torch.zeros(1)    # for models with beta = 0
            rep_pta.append(pta.sum().item())
            rep_iha.append(iha.sum().item())
            rep_iha_tot.append(iha_tot.sum().item())
            rep_m.append(m.sum().item())

        self.write_model_out(self.model.name, sum(rep_pta)/self.n_sequences, sum(rep_iha)/self.n_sequences, sum(rep_iha_tot)/self.n_sequences, sum(rep_m)/self.n_sequences)
    
    @torch.no_grad()
    def run(self):
        for k in self.krange:
            start_time = time.time()
            print(f'k: {k}')
            self.k_step(k)
            for n in self.nrange:
                print(f'n: {n}')
                self.n_step(n)
                for idx in torch.arange(0, self.n_states, self.state_interval):
                    self.s_step(idx)
            print(f'Elapsed time: {time.time()-start_time}')
        
        torch.save(self.model_out, f'{self.dir}/order_params.pt')
    
    @torch.no_grad()
    def run_min(self, krange):
        self.krange = krange
        for k in self.krange:
            start_time = time.time()
            print(f'k: {k}')
            self.k_step(k)
            for n in self.nrange:
                print(f'n: {n}')
                self.n_step(n)
                for idx in torch.arange(0, self.n_states, self.state_interval):
                    flag = self.s_step(idx)
                    if flag:
                        break
            print(f'Elapsed time: {time.time()-start_time}')
        
        torch.save(self.model_out, f'{self.dir}/{k}_order_params.pt')