import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from functools import partial
from src.data_generator_finite import DataGenerator_finite
from src.data_generator import DataGenerator
from utils.aux import extract_nested_dict
from flax.traverse_util import unflatten_dict
from flax.serialization import from_state_dict
import numpy as np
import yaml
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
import seaborn as sns
import argparse
import os, sys
import time
''' 
python ICL_retrieval_AR_find_K2.py --mode direct_prediction --ret bi_ret --MLP_mode match --pair_embed_model mlp --pair_embed_dim 64 --MLP_hidden_dim 256 --D 64 --alpha 1. --N 256  --seed 0 --Batch_size 256 --loss_threshold 2.124 --K_min 64 --K_max 32704

python ICL_retrieval_AR_find_K2.py --mode direct_prediction --ret bi_ret --MLP_mode match_switch --pair_embed_model mlp --pair_embed_dim 64 --MLP_hidden_dim 256 --D 64 --alpha 1. --N 256  --seed 0 --Batch_size 256 --loss_threshold 2.124 --K_min 64 --K_max 128

python ICL_retrieval_AR_find_K2.py --mode direct_prediction --ret bi_ret --MLP_mode one_layer --pair_embed_model mlp --pair_embed_dim 64 --MLP_hidden_dim 256 --D 64 --alpha 1. --N 256  --seed 0 --Batch_size 256 --loss_threshold 2.124 --K_min 64 --K_max 128

python ICL_retrieval_AR_find_K2.py --mode direct_prediction --ret bi_ret --MLP_mode two_layer --pair_embed_model mlp --pair_embed_dim 64 --MLP_hidden_dim 256 --D 64 --alpha 1. --N 256  --seed 0 --Batch_size 256 --loss_threshold 2.124 --K_min 64 --K_max 128
'''

time0 = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='direct_prediction', choices=['task_mixture', 'direct_prediction'], help='model mode')
parser.add_argument('--ret', type=str, default='bi_ret', choices=['bi_ret', 'uni_ret'], help='model mode')
parser.add_argument('--pair_embed_model', type=str, default='linear', choices=['mlp', 'linear', 'identity'], help='pair embedding model')
parser.add_argument('--pair_embed_dim', type=int, default=64, help='pair embedding dimension')
parser.add_argument('--K_min', type=int, default=16, help='minimum number of tasks for binary search')
parser.add_argument('--K_max', type=int, default=1024, help='maximum number of tasks for binary search')
parser.add_argument('--D', type=int, default=64, help='dimension of state embeddings')
parser.add_argument('--alpha', type=float, default=1.0, help='concentration parameter for Dirichlet prior')
parser.add_argument('--N', type=int, default=1024, help='length of the Markov chain')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--MLP_mode', default='two_layer', choices=['one_layer', 'two_layer', 'match', 'match_switch'], help='MLP architecture')
parser.add_argument('--MLP_hidden_dim', type=int, default=256, help='Hidden dimension for MLP')
parser.add_argument('--Batch_size', type=int, default=4096, help='batch size')
parser.add_argument('--loss_threshold', type=float, default=1.965, help='loss threshold for binary search')
parser.add_argument('--max_iters_per_K', type=int, default=200000, help='maximum iterations per K value')
parser.add_argument('--convergence_window', type=int, default=1000, help='window size for checking convergence')
args = parser.parse_args()   


mode = args.mode
ret = args.ret
K_min = args.K_min
K_max = args.K_max
D = args.D
alpha = args.alpha
N = args.N
seed = args.seed
MLP_hidden_dim = args.MLP_hidden_dim
pair_embed_dim = args.pair_embed_dim
pair_embed_model = args.pair_embed_model
MLP_mode = args.MLP_mode
batch_size = args.Batch_size
loss_threshold = args.loss_threshold
max_iters_per_K = args.max_iters_per_K
convergence_window = args.convergence_window

# Binary search results directory
results_base_dir = f"results_retrieval_AR_{MLP_mode}_batch_{batch_size}_critical_K2/{ret}_{mode}/D_{D}_alpha_{alpha:0.1f}_N_{N}/{pair_embed_model}_pair_embed_dim_{pair_embed_dim}_mlp_hidden_dim_{MLP_hidden_dim}_seed_{seed}"
os.makedirs(results_base_dir, exist_ok=True)

print(f"Binary search for critical K in range [{K_min}, {K_max}]")
print(f"Parameters: ret={ret}, mode={mode}, MLP_mode={MLP_mode}, pair_embed_model={pair_embed_model}")
print(f"pair_embed_dim={pair_embed_dim}, MLP_hidden_dim={MLP_hidden_dim}, D={D}, alpha={alpha}, N={N}, seed={seed}")
print(f"Loss threshold: {loss_threshold}")

default_config = yaml.safe_load(open("config/config.yaml"))
config = default_config
config["data_settings"]["seed"] = seed
config["data_settings"]["num_steps"] = N
config["data_settings"]["num_states"] = 10
config["data_settings"]["alpha"] = alpha
config["learning"]["batch_size"] = batch_size
num_states = config["data_settings"]["num_states"]
num_steps = config["data_settings"]["num_steps"]
config["data_settings"]["D"] = D    
config["learning"]["d"] = D
config["data_settings"]["elimate_bias"] = False


# Define model classes first
class MLP2(nn.Module):
    """Two-layer MLP used for embedding or classification.

    - Input: arbitrary feature vector per sample (batch_size, feature_dim)
    - Output: logits of size `output_dim` (batch_size, output_dim)
    """
    output_dim: int
    hidden_dim: int = 256
    @nn.compact
    def __call__(self, inputs):
        hidden = nn.Dense(self.hidden_dim)(inputs)
        hidden = nn.gelu(hidden)
        hidden = nn.Dense(self.hidden_dim)(hidden)
        hidden = nn.gelu(hidden)
        return nn.Dense(self.output_dim)(hidden)

class MLP1(nn.Module):
    """Single-layer MLP used for embedding or classification.

    - Input: arbitrary feature vector per sample (batch_size, feature_dim)
    - Output: logits of size `output_dim` (batch_size, output_dim)
    """
    output_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, inputs):
        hidden = nn.Dense(self.hidden_dim)(inputs)
        hidden = nn.gelu(hidden)
        return nn.Dense(self.output_dim)(hidden)

class Linear(nn.Module):
    """Single linear layer used as a lightweight embedding head."""
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
    pair_embed_model: str = 'mlp'  # 'mlp' or 'linear' or 'identity'

    def setup(self):
        try:
            assert MLP_mode in ['one_layer', 'two_layer', 'match', 'match_switch']
        except:
            raise ValueError(f"Invalid MLP_mode: {MLP_mode}")
        if MLP_mode in ['two_layer', 'match_switch']:
            self.mlp_head = MLP2(output_dim=self.pair_embed_dim, hidden_dim=MLP_hidden_dim)
        elif MLP_mode in ['one_layer', 'match']:
            self.mlp_head = MLP1(output_dim=self.pair_embed_dim, hidden_dim=MLP_hidden_dim)
        self.linear_head = Linear(output_dim=self.pair_embed_dim)

    def __call__(self, inputs):
        B, T, D = inputs.shape

        # last token feature at step t is x_t, for t=0..T-2
        last_token_feats = inputs  # (B, T-1, D)

        if ret == 'bi_ret':
            # Build STRICTLY causal pair ending at t: (x_{t-1}, x_t)
            # Start with (x_t, x_{t+1}) then shift right by 1.
            x_pre = jnp.concatenate([inputs[:, :1, :], inputs[:, :-1, :]], axis=1)  # (B, T, D) = (0, x_0, x_1, ..., x_{T-2})
            pair_feats = jnp.concatenate([x_pre, inputs], axis=-1)  # (B, T-1, 2D) = (x_t, x_{t+1})
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
        denom = jnp.arange(1, T+1, dtype=inputs.dtype)[None, :, None]  # (1, T-1, 1)
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
        try:
            assert MLP_mode in ['one_layer', 'two_layer', 'match', 'match_switch']
        except:
            raise ValueError(f"Invalid MLP_mode: {MLP_mode}")
        if MLP_mode in ['one_layer', 'match_switch']:
            self.head = MLP1(output_dim=self.num_states, hidden_dim=512)
        elif MLP_mode in ['two_layer', 'match']:
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
    targets = batch_states[:, 1:]  # (B, T-1)
    logits = apply_fn({"params": params}, batch_X)  # (B, T-1, S)
    loss_per_pos = optax.softmax_cross_entropy_with_integer_labels(logits, targets)  # (B, T-1)
    return loss_per_pos.mean()

compute_loss = compute_loss_ar
import optax

@jax.jit
def train_step(state, batch_X, batch_states):
    def loss_fn(params):
        return compute_loss(params, state.apply_fn, batch_X, batch_states)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

from flax.training import train_state

# Binary search training function
def train_for_K(K_val, key):
    """Train a fresh model for a specific K value and return final loss.
    
    For each K:
    1. Creates a NEW data generator with K tasks
    2. Initializes a NEW model from scratch
    3. Trains the model until convergence
    4. Returns the final converged loss
    """
    print(f"\n{'='*60}")
    print(f"Training NEW model with K = {K_val} tasks")
    print(f"{'='*60}")
    
 
    # Setup data generator for this K - generates batches from K different transition matrices
    config["data_settings"]["K"] = K_val
    data_generator = DataGenerator_finite(**config["data_settings"])
    print(f"  Created data generator with {K_val} transition matrices")
    
    # Generate initial batch to initialize model
    key, subkey = jax.random.split(key)
    batch_y, batch_X, batch_states, batch_ks = data_generator.generate_batch_markov_chains(batch_size=batch_size, sample="train")
    print(f"  Generated initial batch: {batch_X.shape}")
    
    # Initialize a fresh model from scratch (not reusing weights from previous K)
    model = RetrievalModelAR(mode=mode, pair_embed_dim=pair_embed_dim, num_states=num_states)

    key, subkey = jax.random.split(key)
    variables = model.init(subkey, batch_X)
    params = variables["params"]
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adamw(0.001, b1=0.9, b2=0.95, weight_decay=0.001)
    )
    
       # Check if checkpoint exists
    save_data_dir = os.path.join(results_base_dir, f"K_{K_val}")
    checkpoint_path = os.path.join(save_data_dir, "checkpoint_params.npz")
    loss_history_path = os.path.join(save_data_dir, "loss_history.npz")
    
    # If checkpoint exists, load and return early
    if os.path.exists(checkpoint_path) and os.path.exists(loss_history_path):
        print(f"  ✓ Found existing checkpoint at {checkpoint_path}")
        loss_data = np.load(loss_history_path)
        final_loss = float(loss_data['final_loss'])
        loss_history = list(loss_data['loss_history'])
        iter0 = len(loss_history)
        np_params = np.load(checkpoint_path, allow_pickle=True)
        dict_params = extract_nested_dict(dict(np_params))
        dict_params = jax.tree_util.tree_map(jnp.array, dict_params)
        params_flat = {tuple(k.split('/')): dict_params[k] for k, v in dict_params.items()}
        # Unflatten to nested dict
        params_unflat = unflatten_dict(params_flat)
         # Restore the PyTree structure
        params = from_state_dict(state.params, params_unflat)
        state = state.replace(params=params)
        print(f"  Resumed training from iter {iter0}")
        print(f"  Loaded existing results: final_loss={final_loss:.4f}, {len(loss_history)} iterations")
    else:
        loss_history = []
        iter0 = 0
    # Save results for this K
    os.makedirs(save_data_dir, exist_ok=True)
    converged = False
    # Training loop - generates new batches at each iteration
    last_mean_loss = 100.
    time_iter_start = time.time()
    for iter in range(iter0, max_iters_per_K):
        # Generate new training batch from the K transition matrices
        key, subkey = jax.random.split(key)
        batch_y, batch_X, batch_states, batch_ks = data_generator.generate_batch_markov_chains(batch_size=batch_size, sample="train")
        
        # Training step
        state, loss = train_step(state, batch_X, batch_states) 
        loss_history.append(float(loss))
        
        # Check convergence every 100 iterations
        if iter > convergence_window and iter % 1000 == 0:
            recent_losses = loss_history[-convergence_window:]
            mean_loss = np.mean(recent_losses)
            std_loss = np.std(recent_losses)
            print(f"  Iter {iter}: loss={loss:.4f} std_loss={std_loss:.4f} over last {convergence_window} iters")
            if mean_loss < loss_threshold:
                print(f"    Converged at iter {iter}: loss={mean_loss:.4f} < {loss_threshold}")
                converged = True
                break
            last_mean_loss = mean_loss
        if iter% 10000 == 0 and iter > 0:
            final_loss = np.mean(loss_history[-convergence_window:]) if len(loss_history) >= convergence_window else np.mean(loss_history[-100:])
            np.savez(f"{save_data_dir}/loss_history.npz", loss_history=np.array(loss_history), final_loss=final_loss)
            np.savez(f"{save_data_dir}/checkpoint_params.npz", **jax.tree_util.tree_map(np.array, state.params))

    final_loss = np.mean(loss_history[-convergence_window:]) if len(loss_history) >= convergence_window else np.mean(loss_history[-100:])
    np.savez(f"{save_data_dir}/loss_history.npz", loss_history=np.array(loss_history), final_loss=final_loss)
    np.savez(f"{save_data_dir}/checkpoint_params.npz", **jax.tree_util.tree_map(np.array, state.params))
    print(f"  Final loss for K={K_val}: {final_loss:.4f}")
    time_iter_end = time.time()
    return final_loss, loss_history, key, time_iter_end - time_iter_start


# Binary search state
binary_search_results = []
key = jax.random.PRNGKey(seed)

# Binary search loop
current_min = K_min
current_max = K_max

while current_min < current_max:
    K_mid = (current_min + current_max) // 2
    
    final_loss, loss_hist, key, time_elapsed = train_for_K(K_mid, key)
    
    result = {
        'K': K_mid,
        'final_loss': final_loss,
        'below_threshold': final_loss < loss_threshold,
        'time_elapsed': time_elapsed
    }
    binary_search_results.append(result)
    print(f"Binary search step: K={K_mid}, final_loss={final_loss:.4f}, time_elapsed={time_elapsed:.2f} sec")
    if final_loss > loss_threshold:
        # Success! Try smaller K
        print(f"\n✓ K={K_mid} achieved loss {final_loss:.4f} > {loss_threshold}")
        print(f"  Searching lower range: [{current_min}, {K_mid}]\n")
        current_max = K_mid
    else:
        # Failed, need larger K
        print(f"\n✗ K={K_mid} achieved loss {final_loss:.4f} <= {loss_threshold}")
        print(f"  Searching upper range: [{K_mid + 1}, {current_max}]\n")
        current_min = K_mid + 1

    if current_max - current_min < 0.1*current_min:
        print(f"Search range sufficiently small: [{current_min}, {current_max}]")
        break
    
# Final result
K_critical = current_min if current_max - current_min >0 else (current_min + current_max) // 2
print(f"\n{'='*60}")
print(f"Binary search complete!")
print(f"Critical K value: {K_critical}")
print(f"This is the minimum K where loss can reach below {loss_threshold}")
print(f"{'='*60}\n")

# Train one more time at critical K for final verification
print(f"Final verification training at K={K_critical}...")
final_loss, final_hist, key, time_elapsed = train_for_K(K_critical, key)
print(f"Final verification: K={K_critical}, loss={final_loss:.4f}\n")

# Save binary search summary
summary = {
    'K_critical': K_critical,
    'K_min_search': K_min,
    'K_max_search': K_max,
    'K_max_last': current_max,
    'K_min_last': current_min,
    'loss_threshold': loss_threshold,
    'final_loss': final_loss,
    'final_time_elapsed': time_elapsed,
    'search_history': binary_search_results
}
with open(os.path.join(results_base_dir, 'binary_search_summary.pkl'), 'wb') as f:
    pickle.dump(summary, f)

print(f"Results saved to {results_base_dir}")
print(f"\nSearch history:")
for res in binary_search_results:
    status = "✓" if res['below_threshold'] else "✗"
    print(f"  {status} K={res['K']}: loss={res['final_loss']:.4f} (time: {res['time_elapsed']:.2f} sec)")
print(f"\n Total time elapsed: {time.time() - time0:.2f} seconds")



