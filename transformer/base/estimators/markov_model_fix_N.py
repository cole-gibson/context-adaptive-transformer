# ------------------------------------------------------------
# Non-autoregressive "Last-step" Estimators (B,1,V logits)
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import ConfigDict
from base.estimators.markov_model import seq2trans, row_normalize

_EPS = 1e-12


class UniGenLast(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.name = '1_gen_last'
        self.device = config.task.device
        self.vocab_size = config.task.vocab_size
        self.alpha = config.task.alpha

    def forward(self, idx, targets, reduction='mean', return_hatT=False):
        B, T = idx.shape
        V = self.vocab_size

        seq = F.one_hot(idx, V).float()      # (B, T, V)
        counts = seq.sum(dim=1) + self.alpha # (B, V)
        pnext = counts / counts.sum(dim=-1, keepdim=True)
        pnext = pnext.clamp_min(_EPS)

        logits_last = torch.log(pnext).unsqueeze(1)  # (B, 1, V)

        loss = F.cross_entropy(
            logits_last.squeeze(1),
            targets[:, -1],
            reduction=reduction
        )

        if reduction == 'none':
            loss = loss.unsqueeze(1)  # (B, 1)

        return logits_last, loss


class UniMemLast(nn.Module):
    def __init__(self, config: ConfigDict, task_pool: torch.Tensor, stat_dist: torch.Tensor):
        super().__init__()
        self.name = '1_mem_last'
        self.device = config.task.device
        self.vocab_size = config.task.vocab_size
        self.stat_dist = stat_dist
        self.task_pool = task_pool

    def forward(self, idx, targets, reduction='mean', return_hatT=False):
        B, T = idx.shape
        V = self.vocab_size

        seq = F.one_hot(idx, V).float()
        counts = seq.sum(dim=1)  # (B, V)

        log_probs = torch.log(self.stat_dist.clamp_min(_EPS))  # (K, V)
        post_logits = counts @ log_probs.T                    # (B, K)
        post = F.softmax(post_logits, dim=-1)                 # (B, K)

        hatT_last = torch.einsum('bk,kpq->bpq', post, self.task_pool)  # (B, V, V)

        cur = F.one_hot(idx[:, -1], V).float()                # (B, V)
        pnext = torch.einsum('bpq,bp->bq', hatT_last, cur)    # (B, V)
        pnext = pnext.clamp_min(_EPS)

        logits_last = torch.log(pnext).unsqueeze(1)           # (B, 1, V)

        loss = F.cross_entropy(
            logits_last.squeeze(1),
            targets[:, -1],
            reduction=reduction
        )

        if reduction == 'none':
            loss = loss.unsqueeze(1)  # (B, 1)

        if return_hatT:
            return logits_last, loss, hatT_last
        return logits_last, loss


class BiGenLast(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.name = '2_gen_last'
        self.device = config.task.device
        self.vocab_size = config.task.vocab_size
        self.alpha = config.task.alpha
        self.uni_gen_last = UniGenLast(config)

    def forward(self, idx, targets, reduction='mean', return_hatT=False):
        B, T = idx.shape
        V = self.vocab_size

        if T == 1:
            return self.uni_gen_last(idx, targets, reduction=reduction)

        trans = seq2trans(idx, V)                  # (B, T-1)
        tseq = F.one_hot(trans, V * V).float()     # (B, T-1, V^2)

        cum_counts = torch.cumsum(tseq, dim=1)
        counts_last = cum_counts[:, -1, :] + self.alpha

        hatT_last = counts_last / counts_last.sum(dim=-1, keepdim=True)
        hatT_last = hatT_last.reshape(B, V, V)

        cur = F.one_hot(idx[:, -1], V).float()
        pnext = torch.einsum('bpq,bp->bq', hatT_last, cur)
        pnext = pnext.clamp_min(_EPS)

        logits_last = torch.log(pnext).unsqueeze(1)  # (B, 1, V)

        loss = F.cross_entropy(
            logits_last.squeeze(1),
            targets[:, -1],
            reduction=reduction
        )

        if reduction == 'none':
            loss = loss.unsqueeze(1)

        return logits_last, loss


class BiMemLast(nn.Module):
    def __init__(self, config: ConfigDict, task_pool: torch.Tensor, stat_dist: torch.Tensor):
        super().__init__()
        self.name = '2_mem_last'
        self.device = config.task.device
        self.vocab_size = config.task.vocab_size
        self.stat_dist = stat_dist
        self.task_pool = task_pool
        self.uni_mem_last = UniMemLast(config, task_pool, stat_dist)

    def forward(self, idx, targets, reduction='mean', return_hatT=False):
        B, T = idx.shape
        V = self.vocab_size

        if T == 1:
            return self.uni_mem_last(idx, targets, reduction=reduction, return_hatT=return_hatT)

        trans = seq2trans(idx, V)
        tseq = F.one_hot(trans, V * V).float()

        cum_counts = torch.cumsum(tseq, dim=1)
        counts_last = cum_counts[:, -1, :]  # (B, V^2)

        log_probs = torch.log(self.task_pool.clamp_min(_EPS)).flatten(-2, -1)
        post_logits = counts_last @ log_probs.T
        post = F.softmax(post_logits, dim=-1)

        hatT_last = torch.einsum('bk,kpq->bpq', post, self.task_pool)

        cur = F.one_hot(idx[:, -1], V).float()
        pnext = torch.einsum('bpq,bp->bq', hatT_last, cur)
        pnext = pnext.clamp_min(_EPS)

        logits_last = torch.log(pnext).unsqueeze(1)  # (B, 1, V)

        loss = F.cross_entropy(
            logits_last.squeeze(1),
            targets[:, -1],
            reduction=reduction
        )

        if reduction == 'none':
            loss = loss.unsqueeze(1)

        if return_hatT:
            return logits_last, loss, hatT_last
        return logits_last, loss


def init_all_estimators_last(config: ConfigDict, task_pool: torch.Tensor, stat_dist: torch.Tensor):
    device = config.task.device
    return (
        UniMemLast(config, task_pool, stat_dist).to(device),
        UniGenLast(config).to(device),
        BiMemLast(config, task_pool, stat_dist).to(device),
        BiGenLast(config).to(device),
    )


def init_generalizing_estimators_last(config: ConfigDict, task_pool=None, stat_dist=None):
    device = config.task.device
    return (
        UniGenLast(config).to(device),
        BiGenLast(config).to(device),
    )