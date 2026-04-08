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

''' 
python ICL_Retrieval.py --mode direct_prediction --ret bi_ret --MLP_mode match --pair_embed_model mlp --pair_embed_dim 64 --MLP_hidden_dim 256 --K 1024 --D 64 --alpha 1. --N 1000 --seed 3
'''


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='direct_prediction', choices=['task_mixture', 'direct_prediction'], help='model mode')
parser.add_argument('--ret', type=str, default='bi_ret', choices=['bi_ret', 'uni_ret'], help='model mode')
parser.add_argument('--pair_embed_model', type=str, default='linear', choices=['mlp', 'linear', 'identity'], help='pair embedding model')
parser.add_argument('--pair_embed_dim', type=int, default=100, help='pair embedding dimension')
parser.add_argument('--K', type=str, default=64, help='number of tasks')
parser.add_argument('--D', type=int, default=64, help='dimension of state embeddings')
parser.add_argument('--alpha', type=float, default=1.0, help='concentration parameter for Dirichlet prior')
parser.add_argument('--N', type=int, default=1000, help='length of the Markov chain')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--MLP_mode', default='two_layer', choices=['one_layer', 'two_layer', 'match', 'match_switch'], help='MLP architecture')
parser.add_argument('--load_checkpoint', action='store_true', help='Load checkpoint')
parser.add_argument('--MLP_hidden_dim', type=int, default=256, help='Hidden dimension for MLP')
parser.add_argument('--Batch_size', type=int, default=4096, help='batch size')
args = parser.parse_args()   

mode = args.mode
ret = args.ret
if args.K !='infinity':
    K = int(args.K)
else:
    K = args.K
D = args.D
alpha = args.alpha
N = args.N
seed = args.seed
MLP_hidden_dim = args.MLP_hidden_dim
pair_embed_dim = args.pair_embed_dim
pair_embed_model = args.pair_embed_model
MLP_mode = args.MLP_mode
batch_size=args.Batch_size
save_data_dir = f"results_retrieval_{MLP_mode}_batch_{batch_size}/{ret}_{mode}/K_{K}_D_{D}_alpha_{alpha:0.1f}_N_{N}/{pair_embed_model}_pair_embed_dim_{pair_embed_dim}_mlp_hidden_dim_{MLP_hidden_dim}_seed_{seed}"
os.makedirs(save_data_dir, exist_ok=True)

trial_name = f"ret: {ret}, mode: {mode}, MLP_mode: {MLP_mode}, pair_embed_model: {pair_embed_model}, pair_embed_dim: {pair_embed_dim}, MLP_hidden_dim: {MLP_hidden_dim}, K: {K}, D: {D}, alpha: {alpha}, N: {N}, seed: {seed}"
print("Trial name:", trial_name)
load_checkpoint = args.load_checkpoint
default_config = yaml.safe_load(open("config/config.yaml"))
config = default_config
config["data_settings"]["seed"] = seed
config["data_settings"]["num_steps"] = N
config["data_settings"]["num_states"] = 10
config["data_settings"]["alpha"] = alpha
config["data_settings"]["K"] = K
config["learning"]["batch_size"] = batch_size
num_states = config["data_settings"]["num_states"]
num_steps = config["data_settings"]["num_steps"]
config["data_settings"]["D"] = D    
config["learning"]["d"] = D
config["data_settings"]["elimate_bias"] = False


if K == 'infinity':
    data_generator = DataGenerator(**config["data_settings"])
else:
    data_generator = DataGenerator_finite(**config["data_settings"])

seed = config["data_settings"]["seed"]
batch_size = config["learning"]["batch_size"]

key = jax.random.PRNGKey(seed)

if K == 'infinity':
    keys = jax.random.split(key, batch_size)
    batch_y, batch_X, batch_states, _ = data_generator.generate_batch_markov_chains(keys)
else:
    batch_y, batch_X, batch_states, batch_ks = data_generator.generate_batch_markov_chains(batch_size=batch_size, sample="train")

 
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

class PoolingLayer(nn.Module):
    """Builds a sequence-level feature from adjacent-token pair embeddings.

    Given input tokens `inputs` with shape (batch_size, seq_len, embed_dim):
    1) Form adjacent pairs by concatenation: (batch_size, seq_len-1, 2*embed_dim)
    2) Map pairs to `pair_embed_dim` either via MLP or Linear: (batch_size, seq_len-1, pair_embed_dim)
    3) Sum across the temporal axis (global pooling): (batch_size, pair_embed_dim)
    4) Concatenate the last token features `inputs[:, -1]` (batch_size, embed_dim) to provide context: 
       (batch_size, pair_embed_dim + embed_dim)
    """
    pair_embed_dim: int = 100
    pair_embed_model: str = 'linear'  # 'mlp' or 'linear' or 'none'

    def setup(self):
        if MLP_mode == 'two_layer' or MLP_mode == 'match_switch':
            self.mlp_head = MLP2(output_dim=self.pair_embed_dim, hidden_dim=MLP_hidden_dim)
        elif MLP_mode == 'one_layer' or MLP_mode == 'match':
            self.mlp_head = MLP1(output_dim=self.pair_embed_dim, hidden_dim=MLP_hidden_dim)
        self.linear_head = Linear(output_dim=self.pair_embed_dim)

    def __call__(self, inputs):
        # inputs: (batch_size, seq_len, embed_dim)
        if ret=='bi_ret':
            adjacent_pairs = jnp.concatenate([inputs[:, :-1], inputs[:, 1:]], axis=-1) # (batch_size, seq_len-1, 2*embed_dim)
        else:
            adjacent_pairs = inputs

        if self.pair_embed_model == 'mlp':
            pair_embeddings = self.mlp_head(adjacent_pairs)           # (batch_size, seq_len-1, pair_embed_dim)
        elif self.pair_embed_model == 'linear':
            pair_embeddings = self.linear_head(adjacent_pairs)        # (batch_size, seq_len-1, pair_embed_dim)
        else:
            # No embedding head: use raw concatenated pairs
            pair_embeddings = adjacent_pairs                          # (batch_size, seq_len-1, 2*embed_dim)
        
        pooled_features = jnp.sum(pair_embeddings, axis=-2)/inputs.shape[-2]           # (batch_size, pair_embed_dim) or (batch_size, 2*embed_dim)
        last_token_features = inputs[:, -1]                           # (batch_size, embed_dim)
        return jnp.concatenate([pooled_features, last_token_features], axis=-1)  # (batch_size, pair_embed_dim + embed_dim)

class RetrievalModel(nn.Module):
    """Retrieval model with two modes:

    - mode='task_mixture': learn task weights over num_tasks transition matrices.
      Produces a batch of weighted transition matrices (batch_size, num_states, num_states) and returns
      the row corresponding to each sample's `current_states` -> (batch_size, num_states) next-state distribution.

    - mode='direct_prediction': directly predict next-state distribution (batch_size, num_states) from pooled features.
    """
    # num_tasks: int = 1                         # number of tasks / transition matrices
    # transition_matrices: jnp.ndarray = jnp.zeros((1, 10, 10))       # transition matrices, shape (num_tasks, num_states, num_states)
    mode: str = 'task_mixture'              # 'task_mixture' or 'direct_prediction'
    pair_embed_dim: int = 100
    pair_embed_model: str = 'linear'        # 'mlp' or 'linear'
    num_states: int = 10                    # number of states

    def setup(self):
        self.pooling = PoolingLayer(
            pair_embed_dim=self.pair_embed_dim,
            pair_embed_model=self.pair_embed_model,
        )
        #self.task_weight_mlp = MLP(output_dim=self.num_tasks, hidden_dim=MLP_hidden_dim)      # for task weighting
        if MLP_mode == 'one_layer' or MLP_mode == 'match_switch':
            self.direct_prediction_mlp = MLP1(output_dim=self.num_states, hidden_dim=512)  # for direct next-state logits
        elif MLP_mode == 'two_layer' or MLP_mode == 'match':
            self.direct_prediction_mlp = MLP2(output_dim=self.num_states, hidden_dim=512)  # for direct next-state logits
       # self.linear_projection = Linear(output_dim=self.num_tasks)

    def __call__(self, token_embeddings, current_states):
        """Forward pass.

        Args:
            token_embeddings: token embeddings, shape (batch_size, seq_len, embed_dim)
            current_states: integer ids of current states per sample, shape (batch_size,)
        Returns:
            (batch_size, num_states) next-state distribution (probabilities).
        """
        batch_size = token_embeddings.shape[0]
        D = token_embeddings.shape[-1]
        pooled_sequence_features = self.pooling(token_embeddings)  # (batch_size, pair_embed_dim + embed_dim)

        # if self.mode == 'task_mixture':
        #     # Use only the pair embedding portion for task weighting
        #     if self.pair_embed_model == 'linear' or self.pair_embed_model == 'mlp':
        #         task_logits = self.task_weight_mlp(pooled_sequence_features[:, : self.pair_embed_dim])
        #     else:
        #         task_logits = self.task_weight_mlp(pooled_sequence_features[:, : 2*D])  # (batch_size, num_tasks)
        #     task_weights = nn.softmax(task_logits)                                                  # (batch_size, num_tasks)
        #     # Weighted sum across num_tasks transition matrices -> (batch_size, num_states, num_states)
        #     weighted_transitions = jnp.einsum('bt,tij->bij', task_weights, self.transition_matrices)
        #     batch_indices = jnp.arange(batch_size)
        #     # Select row `current_states` from each (num_states, num_states) matrix -> (batch_size, num_states)
        #     return weighted_transitions[batch_indices, current_states, :]

        if self.mode == 'direct_prediction':
            # Directly predict next-state distribution (batch_size, num_states)
            next_state_logits = self.direct_prediction_mlp(pooled_sequence_features)  # (batch_size, num_states)
            return nn.softmax(next_state_logits)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")


def compute_loss_fn(model_params, model_apply_fn, batch_token_embeddings, batch_target_states, batch_current_states, num_states):
    """Cross-entropy loss over next-state predictions.

    Expects model.apply to return (batch_size, num_states) probabilities for next state.
    """
    # Predicted next-state probabilities per sample: (batch_size, num_states)
    predicted_probs = model_apply_fn({"params": model_params}, batch_token_embeddings, batch_current_states)
    predicted_probs = jnp.clip(predicted_probs, 1e-8, 1.0)  # Numerical stability
    log_predicted_probs = jnp.log(predicted_probs)
    target_one_hot = jax.nn.one_hot(batch_target_states, num_states)  # (batch_size, num_states)
    # Mean negative log-likelihood over batch
    return -jnp.sum(target_one_hot * log_predicted_probs, axis=1).mean()

from functools import partial
# Bind the known number of states from the data generator
compute_loss = partial(compute_loss_fn, num_states=data_generator.num_states)
import optax

@jax.jit
def train_step(state, batch_X, batch_y, batch_last_state):
    loss_fn = lambda params: compute_loss(params, state.apply_fn, batch_X, batch_y, batch_last_state)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

from flax.training import train_state

print("task diversity", config["data_settings"]["K"])

model = RetrievalModel(mode=mode, pair_embed_model=pair_embed_model, pair_embed_dim=pair_embed_dim)
key, subkey = jax.random.split(key)
variables = model.init(subkey, batch_X, batch_states[:, -2])
params = variables["params"]
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.adamw(0.001, b1=0.9, b2=0.95, weight_decay=0.001)
)


loss_history, loss_validation_history, loss_validation_iterations, iter0 = [], [], [], 0

loss_file = os.path.join(save_data_dir, "loss_history.npz")
checkpoint_file = os.path.join(save_data_dir, "checkpoint_params.npz")
if os.path.exists(loss_file) and args.load_checkpoint:
    loss_history_dic = np.load(loss_file, allow_pickle=True)
    if np.abs(loss_history_dic['loss_history'][-1] - Biloss) < 0.03 or len(loss_history_dic['loss_history']) > 2*10**5:
        print(f"Data already exists and final loss is close to Biloss {trial_name}. Exiting...")
        sys.exit()
    elif args.load_checkpoint and os.path.exists(checkpoint_file):
        iter0 = len(loss_history_dic['loss_history']) 
        print(f"Loading checkpoint at iter {iter0} for mode {mode}, K={K}, alpha={alpha}, seed={seed}")
        np_params = np.load(checkpoint_file, allow_pickle=True)
        dict_params = extract_nested_dict(dict(np_params))

        dict_params = jax.tree_util.tree_map(jnp.array, dict_params)

        params_flat = {tuple(k.split('/')): dict_params[k] for k, v in dict_params.items()}
        # Unflatten to nested dict
        params_unflat = unflatten_dict(params_flat)
        # Restore the PyTree structure
        params = from_state_dict(state.params, params_unflat)
        # Update state
        state = state.replace(params=params)
        loss_history = list(loss_history_dic['loss_history'])
        loss_validation_history = list(loss_history_dic['loss_validation_history'])
        loss_validation_iterations = list(loss_history_dic['loss_validation_iterations'])
        print(f"Resuming training from iter {iter0}, loss {loss_history[-1]:.4f}")
else:
    print(f"Starting training from scratch for mode {trial_name}")


for iter in tqdm(range(iter0, 5*10**5+1)):  # Train for 5 epochs
    if K == 'infinity':
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)
        batch_y, batch_X, batch_states, _ = data_generator.generate_batch_markov_chains(keys)
    else:
        batch_y, batch_X, batch_states, batch_ks = data_generator.generate_batch_markov_chains(batch_size=batch_size, sample="train")
    batch_last_state = batch_states[:, -2]  # Get the second last state as input to predict the last state
    state, loss = train_step(state, batch_X, batch_y, batch_last_state)
    loss_history.append(loss)
    if (iter) % 10== 0:
        if K != 'infinity':
            batch_y, batch_X, batch_states, batch_ks = data_generator.generate_batch_markov_chains(batch_size=batch_size, sample="test")
        else:
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, batch_size)
            batch_y, batch_X, batch_states, _ = data_generator.generate_batch_markov_chains(keys)
        loss_val = compute_loss(state.params, state.apply_fn, batch_X, batch_y, batch_states[:, -2])
        loss_validation_history.append(loss_val)
        loss_validation_iterations.append(iter)
        #print(f"Iter {iter}, Train Loss: {loss:.4f}, Val Loss: {loss_val:.4f}")
    if iter % 5000 == 0 and iter > 0:
        loss_dict = {
            'loss_history': np.array(loss_history),
            'loss_validation_history': np.array(loss_validation_history),
            'loss_validation_iterations': np.array(loss_validation_iterations)}
            # 'loss_bi': Biloss,
            # 'loss_uni': Uniloss}
        np.savez(f"{save_data_dir}/loss_history.npz", **loss_dict)
        np.savez(f"{save_data_dir}/checkpoint_params.npz", **jax.tree_util.tree_map(np.array, state.params))
        print(f"Saved iteration {iter}, Loss: {loss:.4f}")
        # if np.abs(np.mean(loss_history[-1000:]) - Biloss) < 0.01:
        #     print("Converged! for mode ", mode, f"K={K}, D={D}, alpha={alpha}, seed={seed}")
        #     print(f"Iteration {iter}, Loss: {loss:.4f}, Biloss: {Biloss:.4f}")
        #     break




