import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import ConfigDict

def seq2trans(seq: torch.Tensor, n_states: int):
    """Maps a sequence of tokens (B, T) to bigram indices (B, T-1).
    Each bigram (x,y) is encoded as: index = x * n_states + y."""
    return seq[:, :-1] * n_states + seq[:, 1:]

def row_normalize(x: torch.Tensor):
    """Row-normalizes tensor(s) over the last dimension.
    Works for tensors of shape (..., N)."""
    norm = x.sum(dim=-1, keepdim=True)
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    return x / norm

# -------------------------------
# Vectorized Estimators
# -------------------------------

class UniMem(nn.Module):
    """Retrieval with likelihood computed from unigram statistics of the context (vectorized)."""
    def __init__(self, config: ConfigDict, task_pool: torch.Tensor, stat_dist: torch.Tensor):
        super().__init__()
        self.name = '1_mem'
        self.device = config.task.device
        self.vocab_size = config.task.vocab_size
        self.stat_dist = stat_dist  # (K, V)
        self.n_tasks = config.task.n_tasks
        self.task_pool = task_pool

    def forward(self, idx, targets, reduction = 'mean', return_hatT = False):
        B, T = idx.shape
        seq = F.one_hot(idx, self.vocab_size).float()  # B, T, C
        cum_counts = torch.cumsum(seq, dim = 1) # B, T, C
        log_probs = torch.log(self.stat_dist)
        cum_loglikelihood = cum_counts @ log_probs.T     # B, T, C @ C, K -> B, T, K
        post = F.softmax(cum_loglikelihood, dim = -1)   # constant prior can be omitted
        hatT = torch.einsum('ijk,kpq-> ijpq', post, self.task_pool) # B, T, K @ K, C, C -> B, T, C, C
        pnext = torch.einsum('ijkl,ijk->ijl', hatT, seq)  # B, T, C, C @ B, T, C
        logits = torch.log(pnext)
        loss = F.cross_entropy(logits.flatten(0,1), targets.flatten(0,1), reduction = reduction)
        if reduction == 'none':
            loss = loss.reshape(B, T)

        if return_hatT:
            return logits, loss, hatT
        else:
            return logits, loss

class BiMem(nn.Module):
    """Retrieval with likelihood computed from bigram statistics of the context (vectorized)."""
    def __init__(self, config: ConfigDict, task_pool: torch.Tensor, stat_dist: torch.Tensor):
        super().__init__()
        self.name = '2_mem'
        self.device = config.task.device
        self.vocab_size = config.task.vocab_size
        self.stat_dist = stat_dist  # (K, V)
        self.task_pool = task_pool
        self.uni_mem = UniMem(config, task_pool, stat_dist)

    def forward(self, idx, targets, reduction = 'mean', return_hatT=False):
        B, T = idx.shape

        # predict first token with uni-mem
        if return_hatT:
            _logits, _loss, _hatT = self.uni_mem(idx[:, :1], targets[:, :1], reduction = 'none', return_hatT=True)      # (B, 1, V), (B, 1)
        else:
            _logits, _loss = self.uni_mem(idx[:, :1], targets[:, :1], reduction = 'none')      # (B, 1, V), (B, 1)

        # uni-mem but for transitions
        seq = F.one_hot(idx[:, 1:], self.vocab_size).float()  # B, T-1, V
        trans = seq2trans(idx, self.vocab_size)     # B, T-1
        tseq = F.one_hot(trans, self.vocab_size**2).float()  # B, T-1, V^2
        cum_counts = torch.cumsum(tseq, dim = 1) # B, T, C
        log_probs = torch.log(self.task_pool).flatten(-2, -1)           # K, V^2
        cum_loglikelihood = cum_counts @ log_probs.T     # B, T-1, V^2 @ V^2, K -> B, T-1, K
        post = F.softmax(cum_loglikelihood, dim = -1)   # constant prior can be omitted
        hatT = torch.einsum('ijk,kpq-> ijpq', post, self.task_pool) # B, T-1, K @ K, V, V -> B, T-1, V, V
        pnext = torch.einsum('ijkl,ijk->ijl', hatT, seq)  # B, T-1, V, V @ B, T-1, V -> B, T-1, V
        logits = torch.log(pnext)   # B, T-1, V
        loss = F.cross_entropy(logits.flatten(0,1), targets[:, 1:].flatten(0,1), reduction = 'none')
        loss = loss.reshape(B, T-1)

        # concatenate predictions
        logits = torch.cat((_logits, logits), dim=1)
        loss = torch.cat((_loss, loss), dim=1)

        if reduction == 'mean':
            loss = loss.mean()

        if return_hatT:
            hatT = torch.cat((_hatT, hatT), dim=1)
            return logits, loss, hatT
        else:
            return logits, loss


class UniGen(nn.Module):
    """Generalization from unigram statistics of the context (vectorized)."""
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.name = '1_gen'
        self.device = config.task.device
        self.vocab_size = config.task.vocab_size
        self.alpha = config.task.alpha

    def forward(self, idx, targets, reduction = 'mean'):
        B, T = idx.shape
        seq = F.one_hot(idx, self.vocab_size).float()  # B, T, C
        cum_counts = torch.cumsum(seq, dim = 1) # B, T, C
        cum_counts += self.alpha    # pseudocount (NOT SURE THIS IS OPTIMAL CHOICE)
        pnext = row_normalize(cum_counts)
        logits = torch.log(pnext)
        loss = F.cross_entropy(logits.flatten(0,1), targets.flatten(0,1), reduction = reduction)
        if reduction == 'none':
            loss = loss.reshape(B, T)
        return logits, loss
    

class BiGen(nn.Module):
    """Generalization from bigram statistics of the context (vectorized)."""
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.name = '2_gen'
        self.device = config.task.device
        self.vocab_size = config.task.vocab_size
        self.alpha = config.task.alpha

    def forward(self, idx, targets, reduction = 'mean'):
        B, T = idx.shape

        # uni-gen but for transitions
        seq = F.one_hot(idx[:, 1:], self.vocab_size).float()  # B, T-1, V

        trans = seq2trans(idx, self.vocab_size)     # B, T-1
        tseq = F.one_hot(trans, self.vocab_size**2).float()  # B, T-1, V^2
        cum_counts = torch.cumsum(tseq, dim = 1) # B, T-1, V^2
        cum_counts += self.alpha    # optimal pseudocount
        hatT = row_normalize(cum_counts)   # B, T-1, V^2
        hatT = hatT.reshape(B, T-1, self.vocab_size, self.vocab_size)   # B, T-1, V, V
        pnext = torch.einsum('ijkl,ijk->ijl', hatT, seq)  # B, T-1, V, V @ B, T-1, V -> B, T-1, V
        logits = torch.log(pnext)   # B, T-1, V^2
        logits = torch.cat((torch.zeros(B, 1, self.vocab_size, device=logits.device), logits), dim=1)  # prepend uniform prediction for first token
        loss = F.cross_entropy(logits.flatten(0,1), targets.flatten(0,1), reduction = 'none')
        loss = loss.reshape(B, T)

        if reduction == 'mean':
            loss = loss.mean()
        return logits, loss


def init_all_estimators(config: ConfigDict, task_pool: torch.Tensor, stat_dist: torch.Tensor):
    """Initialize and return all four estimators (vectorized versions)."""
    device = config.task.device
    bi_mem = BiMem(config, task_pool, stat_dist).to(device)
    bi_gen = BiGen(config).to(device)
    uni_mem = UniMem(config, task_pool, stat_dist).to(device)
    uni_gen = UniGen(config).to(device)
    return (uni_mem, uni_gen, bi_mem, bi_gen)

def init_generalizing_estimators(config: ConfigDict, task_pool: torch.Tensor, stat_dist: torch.Tensor):
    """Initialize and return all four estimators (vectorized versions)."""
    device = config.task.device
    bi_gen = BiGen(config).to(device)
    uni_gen = UniGen(config).to(device)
    return (uni_gen, bi_gen)