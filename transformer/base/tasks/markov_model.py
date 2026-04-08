import torch
from dataclasses import dataclass
from ml_collections import ConfigDict
import torch.nn as nn
from torch.distributions import Dirichlet
from math import isfinite
from base.utils import set_seed

@torch.jit.script
def inverse_transform_sample(cdf: torch.Tensor):
    """
    Inverse transform sampling
    :param cdf: torch.Tensor, shape (batch_size, n_bins)
    :return: torch.Tensor, shape (batch_size, 1)
    """
    with torch.no_grad():
        B = cdf.shape[0]
        x = torch.rand((B, 1), device = cdf.device)
        out = torch.searchsorted(cdf, x)
        out = out.squeeze(-1)

    return out

@torch.jit.script
def compute_stat_dist(
        task_pool: torch.Tensor,
        device: str,
        ) -> torch.Tensor:
    """
    Compute stationary distribution of transition matrices.

    args:
        task_pool: torch.Tensor, shape (n_tasks, n_states, n_states)
        device: str, device to use for computation

    returns:
        pi: torch.Tensor, shape (n_tasks, n_states)
    """
    with torch.no_grad():
        batch, V = task_pool.shape[:-1]
        B = torch.zeros(batch, 1, V + 1, device = device, dtype = torch.float64)
        B[..., -1] = 1
        A = torch.cat((torch.eye(V, device = device) - task_pool, torch.ones(batch, V, 1, device = device)), dim = -1)
        Q, R = torch.linalg.qr(A.transpose(-1, -2))
        Q, R = Q.double(), R.double()
        pi = torch.linalg.solve_triangular(R, Q.transpose(-1, -2)@B.transpose(-1, -2), upper = True)
        pi = pi.transpose(-1, -2)
        pi = pi.squeeze(1)
        pi = pi.float() # back to float
    
    return pi

@torch.jit.script
def compute_cdf(trans: torch.Tensor):
    """
    Compute the cdf for transition matrices
    
    args:
    trans: torch.Tensor, shape (batch_size, vocab_size, vocab_size)

    returns:
    cdf: torch.Tensor, shape (batch_size, vocab_size)
    """
    with torch.no_grad():
        cdf = trans.cumsum(dim = -1)

    return cdf

@torch.jit.script
def generate_sequence(cdf: torch.Tensor, stationary_dist: torch.Tensor, n: int, device: str):
    with torch.no_grad():
        B, A, _ = cdf.shape
        out = torch.zeros(B, n+1, device = device, dtype = torch.int64)
        out[:, 0] = inverse_transform_sample(stationary_dist)
        diag = torch.arange(B, device = device)
        for i in range(1, n+1):
            out[:, i] = inverse_transform_sample(cdf[diag, out[:, i - 1]])
    
    return out

class MarkovModel(nn.Module):
    @torch.no_grad()
    def __init__(self, config: ConfigDict):
        super(MarkovModel, self).__init__()
        self.name = 'markov_model'
        self.config = config
        self.device = self.config.device
        self.dirichlet = Dirichlet(torch.full((self.config.vocab_size,), self.config.alpha, device=self.device, dtype=float))
        self.infinite_tasks = not isfinite(self.config.n_tasks)
        assert self.config.n_tasks != 0
        if not self.infinite_tasks:
            if self.config.task_pool is None:
                self.task_pool = self.dirichlet.sample((self.config.n_tasks, self.config.vocab_size))
            else:
                self.task_pool = self.config.task_pool[:self.config.n_tasks]
            self.task_pool_cdf = compute_cdf(self.task_pool)
            self.stat_dist = compute_stat_dist(self.task_pool, self.device)
            self.stat_dist_cdf = compute_cdf(self.stat_dist)
        else:
            self.task_pool = torch.empty(0)
            self.task_pool_cdf = torch.empty(0)
            self.stat_dist = torch.empty(0)
            self.stat_dist_cdf = torch.empty(0)

        # warmup
        self.get_batch(2**4, 2**4, 'train')
        self.get_batch(2**4, 2**4, 'val')
    
    @torch.no_grad()
    def get_batch(self, batch_size: int, context_len: int, dist: str = 'train', return_tasks: bool = False):
        if self.infinite_tasks or dist == 'val':
            task_pool = self.dirichlet.sample((batch_size, self.config.vocab_size))
            task_sample_cdf = compute_cdf(task_pool)
            stat_dist = compute_stat_dist(task_pool, self.device)
            stat_dist_sample_cdf = compute_cdf(stat_dist)
        elif dist == 'train':
            task_pool = self.task_pool
            task_pool_cdf = self.task_pool_cdf
            stat_dist = self.stat_dist
            stat_dist_cdf = self.stat_dist_cdf
            task_sample = torch.randint(0, self.config.n_tasks, (batch_size, ), device=self.device)
            task_sample_cdf = task_pool_cdf[task_sample]
            stat_dist_sample_cdf = stat_dist_cdf[task_sample]
        else:
            raise ValueError(f'Invalid dist: {dist}')
        
        out = generate_sequence(task_sample_cdf, stat_dist_sample_cdf, context_len, self.device)

        x = out[:, :-1]
        y = out[:, 1:]

        if self.device == 'cpu' and self.config.system_device == 'cuda':
            x = x.pin_memory()
            y = y.pin_memory()
            x = x.cuda()
            y = y.cuda()
        
        if return_tasks:
            assert dist == 'train'
            if self.device == 'cpu' and self.config.system_device == 'cuda':
                task_sample = task_sample.pin_memory()
                task_sample = task_sample.cuda()
            task_sample = task_sample.unsqueeze(-1).expand(batch_size, context_len) # expand to (B, T)
            return x, y, task_sample
        else:
            return x, y

def get_task_model(config: ConfigDict):
    set_seed(config.task.seed)
    return MarkovModel(config.task)

def get_task_model_safe(config: ConfigDict):
    """
    Does not impact random state
    """
    return MarkovModel(config.task)