"""
Transformer without multihead attention. Includes dropout.

Optional biases in linear layers and layernorm with bias. Can be configured with RoPE or relative positional embeddings.
Initializes with 1/d variance, and GPT-2 residual corrections. 
"""

import torch, warnings
import torch.nn as nn
from torch.nn import functional as F
from ml_collections import ConfigDict
from typing import Optional
from base.seed import set_seed
from copy import deepcopy

# same as in pd.order_params, avoid circular imports
@torch.jit.script
def beta_mask(x: torch.Tensor) -> torch.Tensor:
    device = x.device
    B, T = x.shape
    x_exp = x.unsqueeze(-1).expand(B, T, T).transpose(-1, -2)
    last_token = x.unsqueeze(-1).expand(B, T, T)
    mask = (x_exp == last_token).int()
    mask = torch.roll(mask, 1, -1)
    mask[..., 0] = 0
    mask = mask.masked_fill(
        torch.tril(torch.ones_like(mask, device=device)) == 0, 0
    ).int()
    return mask

# same as in base.utils, avoid circular import troubles
def safe_getattr(obj, path, default=None):
    for p in path.split("."):
        try:
            obj = getattr(obj, p)
        except AttributeError:
            return default
    return obj

class RotaryPositionalEmbeddings(nn.Module):
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    # All rights reserved.
    #
    # This source code is licensed under the BSD-style license found in the
    # LICENSE file in the root directory of this source tree.
    
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=config.bias """

    def __init__(self, ndim: int, weight: bool, bias: bool):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.weight = nn.Parameter(torch.ones(ndim)) if weight else None
        self.ndim = ndim

    def forward(self, input):
        return F.layer_norm(input, (self.ndim,), self.weight, self.bias, 1e-5)

@torch.jit.script
def circulant(tensor: torch.Tensor):
    device = tensor.device
    n = tensor.size(0)
    circulant_matrix = torch.zeros(n, n, device = device)
    for i in range(n):
        circulant_matrix[i] = torch.roll(tensor, i)
    return circulant_matrix

class Head(nn.Module):
    """A single head of attention"""
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd
        self.key = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.query = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value.residual_proj = 1    # to flag for 1/n_layer scaling
        self.dropout = nn.Dropout(config.dropout)
        self.device = config.device
        self.rope = config.rope
        self.repe = config.repe
        self.block_size = config.block_size

        # initializing positional encodings
        assert not (self.rope and self.repe)  # check for conflicting positional encoding specification
        if self.rope:
            assert head_size % 2 == 0   # requirement of rotary embeddings
            self.rope = RotaryPositionalEmbeddings(head_size, config.block_size, 10_000)
        elif self.repe:
            self.beta = nn.Parameter(torch.zeros((self.block_size, ), device = self.device))
        else:
            raise warnings.warn("No positional encoding, transformer is permutation invariant.")
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, hs)
        q = self.query(x) # (B, T, hs)

        if self.rope:
            # apply rotary pos encoding
            k = self.rope(k.unsqueeze(2)).squeeze(2)    # implemented for multihead attention
            q = self.rope(q.unsqueeze(2)).squeeze(2)

        wei = q @ k.transpose(-2, -1) * C**(-0.5) # (B, T, T), I believe I did scaling right?

        if self.repe:
            # pad or truncate positional bias
            if T > self.block_size:
                beta = torch.cat((self.beta, torch.zeros(T - self.block_size, device = self.device)), dim = -1)
            else:
                beta = self.beta[:T]
            # add to attention weights
            wei += circulant(beta).T

        wei = wei.masked_fill(torch.tril(torch.ones(T, T, device = self.device)) == 0, float('-inf'))
        self.wei = wei
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        self.norm_wei = wei
        
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) --> (B, T, hs)
        return out

class PTAIsolateHead(nn.Module):
    """A single head of attention with weighted pta gradients by config.pta_grad_weight"""
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd
        self.key = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.query = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value.residual_proj = 1    # to flag for 1/n_layer scaling
        self.dropout = nn.Dropout(config.dropout)
        self.device = config.device
        self.rope = config.rope
        self.repe = config.repe
        self.block_size = config.block_size
        self.w = config.pta_grad_weight

        # initializing positional encodings
        assert not (self.rope and self.repe)  # check for conflicting positional encoding specification
        if self.rope:
            assert head_size % 2 == 0   # requirement of rotary embeddings
            self.rope = RotaryPositionalEmbeddings(head_size, config.block_size, 10_000)
        elif self.repe:
            self.beta = nn.Parameter(torch.zeros((self.block_size, ), device = self.device))
        else:
            raise warnings.warn("No positional encoding, transformer is permutation invariant.")
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, hs)
        q = self.query(x) # (B, T, hs)

        if self.rope:
            # apply rotary pos encoding
            k = self.rope(k.unsqueeze(2)).squeeze(2)    # implemented for multihead attention
            q = self.rope(q.unsqueeze(2)).squeeze(2)

        wei = q @ k.transpose(-2, -1) * C**(-0.5) # (B, T, T), I believe I did scaling right?

        if self.repe:
            # pad or truncate positional bias
            if T > self.block_size:
                beta = torch.cat((self.beta, torch.zeros(T - self.block_size, device = self.device)), dim = -1)
            else:
                beta = self.beta[:T]
            # add to attention weights
            wei += circulant(beta).T

        wei = wei.masked_fill(torch.tril(torch.ones(T, T, device = self.device)) == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # reduce gradient for d = 1 tokens
        mask = torch.ones_like(wei[0])
        mask.masked_fill_(torch.diag(torch.ones(T-1, dtype=bool, device=mask.device), -1), self.w)
        wei = wei*mask + wei.detach()*(1-mask)
        
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) --> (B, T, hs)
        return out

class LastTokenHead(nn.Module):
    """A single head of attention that is not autoregressive and computes operation only for final token to reduce
    memory requirements
    """
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd
        self.key = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.query = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value.residual_proj = 1    # to flag for 1/n_layer scaling
        self.dropout = nn.Dropout(config.dropout)
        self.device = config.device
        self.rope = config.rope
        self.repe = config.repe
        self.block_size = config.block_size

        # initializing positional encodings
        assert not (self.rope and self.repe)  # check for conflicting positional encoding specification
        if self.rope:
            assert head_size % 2 == 0   # requirement of rotary embeddings
            self.rope = RotaryPositionalEmbeddings(head_size, config.block_size, 10_000)
        elif self.repe:
            self.beta = nn.Parameter(torch.zeros((self.block_size, ), device = self.device))
        else:
            raise warnings.warn("No positional encoding, transformer is permutation invariant.")
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, hs)
        q = self.query(x[:, -1, :][:, None, :]) # (B, 1, hs) last token only

        if self.rope:
            # apply rotary pos encoding
            k = self.rope(k.unsqueeze(2)).squeeze(2)    # implemented for multihead attention
            q = self.rope(q.unsqueeze(2)).squeeze(2)

        wei = q @ k.transpose(-2, -1) * C**(-0.5) # (B, T, T), I believe I did scaling right?

        if self.repe:
            # pad or truncate positional bias
            if T > self.block_size:
                beta = torch.cat((self.beta, torch.zeros(T - self.block_size, device = self.device)), dim = -1)
            else:
                beta = self.beta[:T]
            # add to attention weights
            wei += circulant(beta).T[-1, :][None, :]

        wei = wei.masked_fill(torch.tril(torch.ones(T, T, device = self.device)) == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) --> (B, T, hs)
        return out

class FeedForward(nn.Module):
    """Simple linear layer followed by GELU nonlinearity"""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.ff_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.ff_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout)
        )
        self.net[-2].residual_proj = 1    # to flag for 1/n_layer scaling
    
    def forward(self, x):
        out = self.net(x)

        return out

class DeepFeedForward(nn.Module):
    """Deep feedforward network"""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.ff_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.ff_embd, config.ff_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.ff_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout)
        )
        self.net[-2].residual_proj = 1    # to flag for 1/n_layer scaling
    
    def forward(self, x):
        out = self.net(x)

        return out

class Attention(nn.Module):
    """Attention block: attention only"""
    def __init__(self, config):
        super().__init__()
        self.sa = Head(config)
        self.ln = LayerNorm(config.n_embd, weight=config.ln_weight,  bias=config.bias)

    def forward(self, x):
        x = x + self.sa(self.ln(x))

        return x
    
class PTAIsolateAttention(nn.Module):
    """Attention block: attention only with pta gradient reweighted"""
    def __init__(self, config):
        super().__init__()
        self.sa = PTAIsolateHead(config)
        self.ln = LayerNorm(config.n_embd, weight=config.ln_weight,  bias=config.bias)

    def forward(self, x):
        x = x + self.sa(self.ln(x))

        return x

class LastTokenAttention(nn.Module):
    """Attention block: attention only, computes only for last token"""
    def __init__(self, config):
        super().__init__()
        self.sa = LastTokenHead(config)
        self.ln = LayerNorm(config.n_embd, weight=config.ln_weight,  bias=config.bias)

    def forward(self, x):
        x = x[:, -1, :][:, None, :] + self.sa(self.ln(x))   # match residual term to last token attention term

        return x

class MLP(nn.Module):
    """MLP block: classification only"""
    def __init__(self, config):
        super().__init__()
        self.ffwd = FeedForward(config)
        self.ln = LayerNorm(config.n_embd, weight=config.ln_weight,  bias=config.bias)

    def forward(self, x):
        x = x + self.ffwd(self.ln(x))

        return x
    
class DeepMLP(nn.Module):
    """Deep MLP block
    
    2 layer feedforward network

    """
    def __init__(self, config):
        super().__init__()
        self.ffwd = DeepFeedForward(config)
        self.ln = LayerNorm(config.n_embd, weight=config.ln_weight,  bias=config.bias)

    def forward(self, x):
        x = x + self.ffwd(self.ln(x))

        return x

class TaskInjectMLP(nn.Module):
    """MLP block with task information injection"""
    def __init__(self, config):
        super().__init__()
        self.ffwd = FeedForward(config)
        self.ln = LayerNorm(config.n_embd, weight=config.ln_weight,  bias=config.bias)
        self.pass_tasks = True

    def forward(self, x, task_emb):
        x = x + self.ffwd(self.ln(x + task_emb))

        return x

class PoolHead(nn.Module):
    """A single head that attends uniformly"""
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd
        self.value = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value.residual_proj = 1    # to flag for 1/n_layer scaling
        self.dropout = nn.Dropout(config.dropout)
        self.device = config.device
    
    def forward(self, x):
        B, T, C = x.shape

        wei = torch.zeros(B, T, T, device = self.device) # (B, T, T), I believe I did scaling right?
        wei = wei.masked_fill(torch.tril(torch.ones(T, T, device = self.device)) == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) --> (B, T, hs)
        return out

class Pool(nn.Module):
    """Attention block: uniform attention"""
    def __init__(self, config):
        super().__init__()
        self.sa = PoolHead(config)
        self.ln = LayerNorm(config.n_embd, weight=config.ln_weight,  bias=config.bias)

    def forward(self, x):
        x = x + self.sa(self.ln(x))

        return x

class PositionHead(nn.Module):
    """A single head that attends only according to RePE"""
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd
        self.value = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value.residual_proj = 1    # to flag for 1/n_layer scaling
        self.dropout = nn.Dropout(config.dropout)
        self.device = config.device
        self.block_size = config.block_size

        _beta = torch.zeros((self.block_size, ), device = self.device)
        _beta[1] = 0.0
        self.beta = nn.Parameter(_beta)
    
    def forward(self, x):
        B, T, C = x.shape

        wei = torch.zeros(B, T, T, device = self.device) # (B, T, T), I believe I did scaling right?
        wei = wei.masked_fill(torch.tril(torch.ones(T, T, device = self.device)) == 0, float('-inf'))
        # pad or truncate positional bias
        if T > self.block_size:
            beta = torch.cat((self.beta, torch.zeros(T - self.block_size, device = self.device)), dim = -1)
        else:
            beta = self.beta[:T]
        # add to attention weights
        wei += circulant(beta).T
        self.wei = wei
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) --> (B, T, hs)
        return out

class PositionAttention(nn.Module):
    """Attention block: featureless RePE attention"""
    def __init__(self, config):
        super().__init__()
        self.sa = PositionHead(config)
        self.ln = LayerNorm(config.n_embd, weight=config.ln_weight,  bias=config.bias)

    def forward(self, x):
        x = x + self.sa(self.ln(x))

        return x

class BetaHead(nn.Module):
    """A single head that has small beta perturbation off uniform"""
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd
        self.value = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value.residual_proj = 1    # to flag for 1/n_layer scaling
        self.dropout = nn.Dropout(config.dropout)
        self.device = config.device
        self.block_size = config.block_size
        self.beta = nn.Parameter(torch.tensor([0.0], device = self.device))
    
    def forward(self, x, idx):
        B, T, C = x.shape

        wei = torch.zeros(B, T, T, device = self.device) # (B, T, T), I believe I did scaling right?
        wei = wei.masked_fill(torch.tril(torch.ones(T, T, device = self.device)) == 0, float('-inf'))
        # add to attention weights
        wei += self.beta*beta_mask(idx)
        self.wei = wei
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) --> (B, T, hs)
        return out

class BetaAttention(nn.Module):
    """Attention block: small beta perturbation off uniform"""
    def __init__(self, config):
        super().__init__()
        self.sa = BetaHead(config)
        self.ln = LayerNorm(config.n_embd, weight=config.ln_weight,  bias=config.bias)
        self.pass_idx = True

    def forward(self, x, idx):
        x = x + self.sa(self.ln(x), idx)

        return x

class PreviousTokenHead(nn.Module):
    """A single head that attends only to the previous token"""
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd
        self.value = nn.Linear(config.n_embd, head_size, bias=config.bias)
        self.value.residual_proj = 1    # to flag for 1/n_layer scaling
        self.dropout = nn.Dropout(config.dropout)
        self.device = config.device
    
    def forward(self, x):
        B, T, C = x.shape

        wei = torch.diagonal_scatter(torch.zeros(B, T, T, device = self.device), torch.ones(B, T-1), offset=-1, dim1 = -2, dim2 = -1) # sub-diagonal all ones
        wei[:, 0, 0] = 1 # first token self attention
        wei = self.dropout(wei)
        
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) --> (B, T, hs)
        return out

class PreviousToken(nn.Module):
    """Attention block: attends only to previous token"""
    def __init__(self, config):
        super().__init__()
        self.sa = PreviousTokenHead(config)
        self.ln = LayerNorm(config.n_embd, weight=config.ln_weight,  bias=config.bias)

    def forward(self, x):
        x = x + self.sa(self.ln(x))

        return x

class Transformer(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.name = 'model'   # used when logging
        self.config = config
        self.arch_dict = {'Attention': Attention, 'MLP': MLP, 'PTAIsolateAttention': PTAIsolateAttention, 'Pool': Pool, 'PreviousToken': PreviousToken, 'LastTokenAttention': LastTokenAttention, 'TaskInjectMLP': TaskInjectMLP, 'PositionAttention': PositionAttention, 'BetaAttention': BetaAttention, 'DeepMLP': DeepMLP}        
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd) # thin wrapper of tensor
        self.inject_tasks = safe_getattr(self.config, 'inject_tasks')
        self.beta_attention = 'BetaAttention' in config.architecture
        if self.inject_tasks:
            assert 'n_tasks' in self.config 
            self.task_embedding_table = nn.Embedding(self.config.n_tasks, self.config.n_embd)
        self.layers = [self.arch_dict[l] for l in config.architecture]
        if PTAIsolateAttention in self.layers:
            assert 'pta_grad_weight' in self.config
        self.blocks = nn.Sequential(*[layer(config) for layer in self.layers]) # might need to unroll list
        self.ln_f = LayerNorm(config.n_embd, weight=config.ln_weight,  bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if self.config.tie_embd:
            self.lm_head.weight = self.token_embedding_table.weight
        
        self.n_residual_proj = 0    # initialize
        self.apply(self._count_residual_proj)   # used in initialization
        self.apply(self._init_weights)  # GPT-2 scaling for residual layers

    def _init_weights(self, module):
        std = self.config.n_embd**-0.5
        std_residual = std * self.n_residual_proj**-0.5    # no factor of 2 because a layer here != transformer block
        if isinstance(module, nn.Linear):
            if hasattr(module, 'residual_proj'):
                nn.init.normal_(module.weight, mean=0.0, std=std_residual)  # residual projection 1/(n_embd * n_layers) scaling
            else:
                nn.init.normal_(module.weight, mean=0.0, std=std)   # standard 1/n_embd scaling
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # initialize embedding weights with normal distribution
            nn.init.normal_(module.weight, mean=0.0, std=std)
        
    def _count_residual_proj(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, 'residual_proj'):
                self.n_residual_proj += 1

    def forward(self, idx, targets=None, tasks=None, last_token_loss = False, reduction = 'mean'): # x is renamed idx
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C) C is vocab_size
        x = tok_emb # (B, T, C)
        assert not (self.inject_tasks and self.beta_attention)
        if self.inject_tasks:
            assert tasks is not None
            if tasks == 'mean':
                task_emb = self.task_embedding_table.weight.mean(0, keepdim=True).unsqueeze(0).expand(B, T, -1)
            else:
                task_emb = self.task_embedding_table(tasks) # (B, C)
            for block in self.blocks:
                if hasattr(block, 'pass_tasks'):
                    x = block(x, task_emb)
                else:
                    x = block(x)
        elif self.beta_attention:
            for block in self.blocks:
                if hasattr(block, 'pass_idx'):
                    x = block(x, idx)
                else:
                    x = block(x)
        else:
            x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            if last_token_loss is True:
                logits = logits[:, -1, :][:, None, :]
                targets = targets[:, -1][:, None]
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.reshape(B*T) # changed from "view" to "reshape" due to RuntimeError
            loss = F.cross_entropy(logits, targets, reduction = reduction)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
def get_model(config: ConfigDict):
    if config.model.init_from_transformer:  # initialize reusing parameters from a 2-block transformer
        transformer_config = deepcopy(config)
        transformer_config.model.architecture = ['Attention', 'MLP', 'Attention', 'MLP']
        set_seed(config.model.seed)
        transformer = Transformer(transformer_config.model).to(transformer_config.system.device)    # full transformer for weight matching
        model = Transformer(config.model).to(config.system.device)
        model.load_state_dict(transformer.state_dict(), strict = False) # load shared parameters from transformer model
    else:
        set_seed(config.model.seed)
        model = Transformer(config.model).to(config.system.device)
    return model

def get_model_safe(config: ConfigDict):
    """
    Get model without changing random seed.
    """
    model = Transformer(config.model).to(config.system.device)
    return model