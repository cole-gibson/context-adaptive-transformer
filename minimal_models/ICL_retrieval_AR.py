# ICL_Retrieval_AR.py
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
from flax.traverse_util import unflatten_dict
from flax.serialization import from_state_dict
from functools import partial
import optax

from src.data_generator_finite import DataGenerator_finite
from src.data_generator import DataGenerator
from utils.aux import extract_nested_dict

import numpy as np
import yaml
from tqdm import tqdm
import argparse
import os, sys

"""
Autoregressive version of your training:
- batch_X: (B, T, D) token embeddings
- batch_states: (B, T) true state sequence
Train to predict s_{t+1} for all t=0..T-2 using prefix-only features (no Python loop; JIT friendly).
batch_y: (B, 1) final prediction target (kept for optional eval)

python ICL_retrieval_AR.py --mode direct_prediction --ret bi_ret --pair_embed_model mlp --pair_embed_dim 64 --K 32 --D 64 --alpha 1.0 --N 256 --seed 1 --MLP_mode match --MLP_hidden_dim 256 --Batch_size 256
"""

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='direct_prediction', choices=['task_mixture', 'direct_prediction'])
parser.add_argument('--ret', type=str, default='bi_ret', choices=['bi_ret', 'uni_ret'])
parser.add_argument('--pair_embed_model', type=str, default='linear', choices=['mlp', 'linear', 'identity'])
parser.add_argument('--pair_embed_dim', type=int, default=100)
parser.add_argument('--K', type=str, default=64)
parser.add_argument('--D', type=int, default=64)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--N', type=int, default=1000)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--MLP_mode', default='two_layer', choices=['one_layer', 'two_layer', 'match', 'match_switch'])
parser.add_argument('--load_checkpoint', action='store_true')
parser.add_argument('--MLP_hidden_dim', type=int, default=256)
parser.add_argument('--Batch_size', type=int, default=4096)
args = parser.parse_args()

mode = args.mode
ret = args.ret
K = int(args.K) if args.K != 'infinity' else args.K
D = args.D
alpha = args.alpha
N = args.N
seed = args.seed
MLP_hidden_dim = args.MLP_hidden_dim
pair_embed_dim = args.pair_embed_dim
pair_embed_model = args.pair_embed_model
MLP_mode = args.MLP_mode
batch_size = args.Batch_size
load_checkpoint = args.load_checkpoint

save_data_dir = (
    f"results_retrieval_AR_{MLP_mode}_batch_{batch_size}/{ret}_{mode}/"
    f"K_{K}_D_{D}_alpha_{alpha:0.1f}_N_{N}/"
    f"{pair_embed_model}_pair_embed_dim_{pair_embed_dim}_mlp_hidden_dim_{MLP_hidden_dim}_seed_{seed}"
)
os.makedirs(save_data_dir, exist_ok=True)

trial_name = (
    f"ret: {ret}, mode: {mode}, MLP_mode: {MLP_mode}, pair_embed_model: {pair_embed_model}, "
    f"pair_embed_dim: {pair_embed_dim}, MLP_hidden_dim: {MLP_hidden_dim}, K: {K}, D: {D}, "
    f"alpha: {alpha}, N: {N}, seed: {seed}"
)
print("Trial name:", trial_name)

default_config = yaml.safe_load(open("config/config.yaml"))
config = default_config
config["data_settings"]["seed"] = seed
config["data_settings"]["num_steps"] = N
config["data_settings"]["num_states"] = 10
config["data_settings"]["alpha"] = alpha
config["data_settings"]["K"] = K
config["learning"]["batch_size"] = batch_size
config["data_settings"]["D"] = D
config["learning"]["d"] = D
config["data_settings"]["elimate_bias"] = False

if K == 'infinity':
    data_generator = DataGenerator(**config["data_settings"])
else:
    data_generator = DataGenerator_finite(**config["data_settings"])

num_states = data_generator.num_states
key = jax.random.PRNGKey(seed)

# ---------------- Models ----------------

class MLP2(nn.Module):
    output_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(self.hidden_dim)(inputs)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        return nn.Dense(self.output_dim)(x)

class MLP1(nn.Module):
    output_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(self.hidden_dim)(inputs)
        x = nn.gelu(x)
        return nn.Dense(self.output_dim)(x)

class Linear(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, inputs):
        return nn.Dense(self.output_dim)(inputs)

class PoolingLayerAR(nn.Module):
    """
    Prefix-only pooling features for autoregressive training.
    Returns features for steps t=0..T-2 to predict s_{t+1}.

    inputs: (B, T, D)
    returns: (B, T-1, pair_embed_dim + D)
    """
    pair_embed_dim: int = 100
    pair_embed_model: str = 'linear'  # 'mlp' or 'linear' or 'identity'

    def setup(self):
        if MLP_mode in ['two_layer', 'match_switch']:
            self.mlp_head = MLP2(output_dim=self.pair_embed_dim, hidden_dim=MLP_hidden_dim)
        else:
            self.mlp_head = MLP1(output_dim=self.pair_embed_dim, hidden_dim=MLP_hidden_dim)
        self.linear_head = Linear(output_dim=self.pair_embed_dim)

    def __call__(self, inputs):
        B, T, D = inputs.shape

        # last token feature at step t is x_t, for t=0..T-2
        last_token_feats = inputs[:, 1:, :]  # (B, T-1, D)

        if ret == 'bi_ret':
            # Build STRICTLY causal pair ending at t: (x_{t-1}, x_t)
            # Start with (x_t, x_{t+1}) then shift right by 1.
            pairs = jnp.concatenate([inputs[:, :-1, :], inputs[:, 1:, :]], axis=-1)  # (B, T-1, 2D) = (x_t, x_{t+1})
            #pairs = jnp.concatenate([jnp.zeros_like(pairs[:, :1, :]), pairs[:, :-1, :]], axis=1)  # (B, T-1, 2D) = (x_{t-1}, x_t)
            pair_feats = pairs
        else:
            # uni_ret: use token x_t as the per-step feature
            pair_feats = inputs[:, :-1, :]  # (B, T-1, D)

        if self.pair_embed_model == 'mlp':
            pair_emb = self.mlp_head(pair_feats)     # (B, T-1, E)
        elif self.pair_embed_model == 'linear':
            pair_emb = self.linear_head(pair_feats)  # (B, T-1, E)
        else:
            pair_emb = pair_feats                    # (B, T-1, *)

        # Prefix mean: pooled[t] = mean_{i<=t} pair_emb[i]
        prefix_sum = jnp.cumsum(pair_emb, axis=1)  # (B, T-1, E)
        denom = jnp.arange(1, T, dtype=inputs.dtype)[None, :, None]  # (1, T-1, 1)
        pooled_prefix = prefix_sum / denom

        return jnp.concatenate([pooled_prefix, last_token_feats], axis=-1)  # (B, T-1, E + D)

class RetrievalModelAR(nn.Module):
    mode: str = 'direct_prediction'
    pair_embed_dim: int = 100
    num_states: int = 10

    def setup(self):
        self.pooling = PoolingLayerAR(
            pair_embed_dim=self.pair_embed_dim,
        )
        if MLP_mode in ['one_layer', 'match_switch']:
            self.head = MLP1(output_dim=self.num_states, hidden_dim=512)
        else:
            self.head = MLP2(output_dim=self.num_states, hidden_dim=512)

    def __call__(self, token_embeddings):
        # token_embeddings: (B, T, D)
        feats = self.pooling(token_embeddings)  # (B, T-1, F)
        logits = self.head(feats)               # (B, T-1, S)
        return logits                           # return logits (stable loss)

# ---------------- Loss / Train step ----------------

def compute_loss_ar(params, apply_fn, batch_X, batch_states):
    """
    batch_states: (B, T) true states
    targets: s_{t+1} for t=0..T-2 => (B, T-1)
    """
    targets = batch_states[:, 2:]  # (B, T-1)
    logits = apply_fn({"params": params}, batch_X)  # (B, T-1, S)
    loss_per_pos = optax.softmax_cross_entropy_with_integer_labels(logits, targets)  # (B, T-1)
    return loss_per_pos.mean()

compute_loss = compute_loss_ar

@jax.jit
def train_step(state, batch_X, batch_states):
    def loss_fn(params):
        return compute_loss(params, state.apply_fn, batch_X, batch_states)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_final_step_loss(state, batch_X, batch_y, batch_states):
    """
    Optional eval: only the final step (t=T-2) should predict s_{T-1}.
    batch_y: (B, 1)
    """
    logits_all = state.apply_fn({"params": state.params}, batch_X, batch_states[:, :-1])  # (B, T-1, S)
    logits_last = logits_all[:, -1, :]  # (B, S)
    targets_last = batch_y.squeeze(-1)  # (B,)
    return optax.softmax_cross_entropy_with_integer_labels(logits_last, targets_last).mean()

# ---------------- Data: initial batch for init ----------------

if K == 'infinity':
    keys = jax.random.split(key, batch_size)
    batch_y, batch_X, batch_states, _ = data_generator.generate_batch_markov_chains(keys)
else:
    batch_y, batch_X, batch_states, batch_ks = data_generator.generate_batch_markov_chains(
        batch_size=batch_size, sample="train"
    )

# ---------------- Init model/state ----------------

print("task diversity", config["data_settings"]["K"])
model = RetrievalModelAR(mode=mode, pair_embed_dim=pair_embed_dim, num_states=num_states)

key, subkey = jax.random.split(key)
variables = model.init(subkey, batch_X)
params = variables["params"]

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.adamw(0.001, b1=0.9, b2=0.95, weight_decay=0.001),
)

# ---------------- Checkpointing ----------------

loss_history, loss_validation_history, loss_validation_iterations, iter0 = [], [], [], 0
loss_file = os.path.join(save_data_dir, "loss_history.npz")
checkpoint_file = os.path.join(save_data_dir, "checkpoint_params.npz")

if os.path.exists(loss_file) and load_checkpoint:
    loss_history_dic = np.load(loss_file, allow_pickle=True)

    # NOTE: your previous script referenced Biloss here; leaving it out so this file runs standalone.
    # If you want that early-exit logic back, you can re-add it with a defined Biloss.

    if load_checkpoint and os.path.exists(checkpoint_file):
        iter0 = len(loss_history_dic['loss_history'])
        print(f"Loading checkpoint at iter {iter0} for {trial_name}")
        np_params = np.load(checkpoint_file, allow_pickle=True)
        dict_params = extract_nested_dict(dict(np_params))
        dict_params = jax.tree_util.tree_map(jnp.array, dict_params)

        params_flat = {tuple(k.split('/')): dict_params[k] for k in dict_params.keys()}
        params_unflat = unflatten_dict(params_flat)
        params = from_state_dict(state.params, params_unflat)
        state = state.replace(params=params)

        loss_history = list(loss_history_dic['loss_history'])
        loss_validation_history = list(loss_history_dic['loss_validation_history'])
        loss_validation_iterations = list(loss_history_dic['loss_validation_iterations'])
        print(f"Resuming training from iter {iter0}, loss {float(loss_history[-1]):.4f}")
else:
    print(f"Starting training from scratch for mode {trial_name}")

# ---------------- Train loop ----------------

for it in tqdm(range(iter0, 5 * 10**5 + 1)):
    if K == 'infinity':
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)
        batch_y, batch_X, batch_states, _ = data_generator.generate_batch_markov_chains(keys)
    else:
        batch_y, batch_X, batch_states, batch_ks = data_generator.generate_batch_markov_chains(
            batch_size=batch_size, sample="train"
        )

    # Autoregressive training uses ALL steps in batch_states
    state, loss = train_step(state, batch_X, batch_states)
    loss_history.append(loss)

    if it % 10 == 0:
        if K != 'infinity':
            batch_y, batch_X, batch_states, batch_ks = data_generator.generate_batch_markov_chains(
                batch_size=batch_size, sample="test"
            )
        else:
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, batch_size)
            batch_y, batch_X, batch_states, _ = data_generator.generate_batch_markov_chains(keys)

        # Option A (default): autoregressive validation loss over all steps
        loss_val = compute_loss(state.params, state.apply_fn, batch_X, batch_states)

        # Option B (optional): final-step-only validation loss using batch_y
        # loss_val = eval_final_step_loss(state, batch_X, batch_y, batch_states)

        loss_validation_history.append(loss_val)
        loss_validation_iterations.append(it)

    if it % 5000 == 0 and it > 0:
        loss_dict = {
            'loss_history': np.array(loss_history),
            'loss_validation_history': np.array(loss_validation_history),
            'loss_validation_iterations': np.array(loss_validation_iterations),
        }
        np.savez(os.path.join(save_data_dir, "loss_history.npz"), **loss_dict)
        np.savez(os.path.join(save_data_dir, "checkpoint_params.npz"),
                 **jax.tree_util.tree_map(np.array, state.params))
        print(f"Saved iteration {it}, Loss: {float(loss):.4f}")