import os
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import traverse_util
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.serialization import from_state_dict
from flax.core.frozen_dict import freeze, unfreeze
import jax.numpy as jnp
import optax
from functools import partial
from src.data_generator import DataGenerator
from src.model import TransformerModel, compute_loss_fn
from tqdm import tqdm
from utils.aux import get_largest_iteration, extract_nested_dict
import yaml, argparse, os, logging, shutil
import numpy as np
import time, os, sys
import pprint
"""
# python ICL_generalization_full.py --seed 0 --mode mlp --num_steps 1000 --D 64 --d 64 --max_distance 1000 --batch_size 256 --alpha 0.5 --RelPosBias full --num_states 10 --optimizer SGD --k 1 --elimate_bias --load_checkpoint
"""
start_time = time.time()

default_config = yaml.safe_load(open("config/config.yaml"))

parser = argparse.ArgumentParser(description='Plot syllable window')
parser.add_argument('--D', type=int, default=10, help='token embedding dimension')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--mode', type=str, default="mlp", choices=["linear", "fixed_linear", "linear_reduce", "fixed_linear_bias", "mlp", "mlp_reduce", "identity", "identity_v1"], help='mode type')
parser.add_argument('--optimizer', type=str, default="Mix", choices=["Adam", "SGD", "Mix"], help='optimizer type')
parser.add_argument('--num_steps', type=int, default=1000, help='number of steps')
parser.add_argument('--d', type=int, default=10, help='query/key/value dimension')
parser.add_argument('--num_states', type=int, default=10, help='number of states')
parser.add_argument('--alpha', type=float, default=1., help='concentration parameter')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--max_distance', type=int, default=128, help='batch size')
parser.add_argument('--RelPosBias', type=str, default='full', help='relative position bias type')
parser.add_argument('--k', type=int, default=1, help='step for mixture of optmizers')
parser.add_argument('--kinverse', action='store_true', help='Enable inverse k step')
parser.add_argument('--keep_prev', action='store_true', help='Enable inverse k step')
parser.add_argument('--elimate_bias', action='store_true', help='Elimate bias in the generating the sequence')
parser.add_argument('--load_checkpoint', action='store_true', help='Load checkpoint')
args = parser.parse_args()
config = default_config
config["data_settings"]["D"] = args.D
config["data_settings"]["seed"] = args.seed
config["data_settings"]["num_steps"] = args.num_steps
config["data_settings"]["num_states"] = args.num_states
config["data_settings"]["alpha"] = args.alpha
config["learning"]["d"] = args.d
config["learning"]["batch_size"] = args.batch_size
config["learning"]["mode"] = args.mode
config["data_settings"]["elimate_bias"] = args.elimate_bias

k = args.k
one_hot = False
num_states = config["data_settings"]["num_states"]
data_generator = DataGenerator(**config["data_settings"])
seed = args.seed
batch_size = args.batch_size
mode = args.mode


print("kinverse:", args.kinverse)

key = jax.random.PRNGKey(seed)
keys = jax.random.split(key, batch_size)
batch_y, batch_X, batch_states, _ = data_generator.generate_batch_markov_chains(keys)

if one_hot:
    batch_X = jax.nn.one_hot(batch_states[:, 0:-1], num_states)
    X = jax.nn.one_hot(jnp.arange(num_states), num_states)
else:
    X = data_generator.X

compute_loss = partial(compute_loss_fn, J=X, num_states=data_generator.num_states, lam=1.)

if one_hot:
    D, d = num_states, num_states
else:
    D, d = config["data_settings"]["D"], config["learning"]["d"]

model = TransformerModel(D=D, d=d, C=data_generator.num_states, seq_len=batch_X.shape[1], max_distance = args.max_distance, architecture=args.mode, RelPosBias=args.RelPosBias)
key, subkey = jax.random.split(key)
variables = model.init(subkey, batch_X, X)
params = variables["params"]


def label_fn(path):
    if 'attention1st' in '/'.join(path):
        return 'attention_encoder'
    elif 'attention2nd' in '/'.join(path):
        return 'attention_encoder'
    else:
        return 'mlp_head'  # fallback

def create_param_labels(params):
    flat = traverse_util.flatten_dict(params)
    labels = {path: label_fn(path) for path in flat}
    return traverse_util.unflatten_dict(labels)

if args.optimizer == "SGD":
    tx = optax.multi_transform(
        {
            'attention_encoder': optax.sgd(learning_rate=0.1, momentum=0.0),
            'mlp_head': optax.sgd(learning_rate=0.1, momentum=0.0),
        },
        param_labels=create_param_labels
    )
elif args.optimizer == "Mix":
        tx = optax.multi_transform(
        {
            'attention_encoder': optax.sgd(learning_rate=0.5, momentum=0.),
            'mlp_head': optax.adam(learning_rate=1e-2),
        },
        param_labels=create_param_labels
    )
elif args.optimizer == "Adam":
        tx = optax.multi_transform(
        {
            'attention_encoder': optax.adam(learning_rate=1e-3),
            'mlp_head': optax.adam(learning_rate=1e-2),
        },
        param_labels=create_param_labels
    )


mask_component = 'attention_encoder' if not args.kinverse else 'mlp_head'

@jax.jit
def train_step(state, batch_X, batch_y, step, k):
    loss_fn = lambda params: compute_loss(params, state.apply_fn, batch_X, batch_y)

    grads = jax.grad(loss_fn)(state.params)

    # Mask group2 gradients unless step % k == 0
    labels = create_param_labels(state.params)

    def mask_grads(grads, labels, step, k):
        def mask_leaf(g, label):
            # Use JAX's lax.cond for JIT compatibility
            return jax.lax.cond(
                (label == mask_component) & (step % k != 0),
                lambda _: jnp.zeros_like(g),
                lambda _: g,
                operand=None
            )
        flat_grads = traverse_util.flatten_dict(grads)
        flat_labels = traverse_util.flatten_dict(labels)
        masked = {k: mask_leaf(g, flat_labels[k]) for k, g in flat_grads.items()}
        return traverse_util.unflatten_dict(masked)
    grads = mask_grads(grads, labels, step, k)

    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )
    loss = loss_fn(new_params)
    return new_state, loss

state = train_state.TrainState.create(
    apply_fn=model.apply,   # model's forward function
    params=params,          # model parameters
    tx=tx                   # optax optimizer (can be multi_transform)
)

suffix = f"_k_{args.k}" if not args.kinverse else f"_kinv_{args.k}"
suffix += f"_elimate_bias" if args.elimate_bias else ""
save_data_dir = os.path.abspath(os.path.join(
    config["learning"]["save_path"],
    f"N_{args.num_steps}_D_{D}_d_{d}_C_{args.num_states}_alpha_{args.alpha}_max_distance_{args.max_distance}/{mode}_RelPosBias_{args.RelPosBias}_{args.optimizer}{suffix}/trial_{seed}"
))


os.makedirs(save_data_dir, exist_ok=True)
print(save_data_dir)
save_log_dir = os.path.abspath("./error_logs/")
os.makedirs(save_log_dir, exist_ok=True)
log_file = os.path.join(save_log_dir, f"error_log_info.txt")
# Set up logging configuration
logging.basicConfig(filename=log_file,
    filemode='a',
    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.ERROR,
)

iter0 = 0
save_loss_dir = os.path.join(save_data_dir, "loss")
save_checkpoint_dir = os.path.join(save_data_dir, "checkpoints")

# Try to load existing checkpoint if requested
if args.load_checkpoint and os.path.exists(f"{save_loss_dir}/loss_history.npy"):
    loss_history = np.load(f"{save_loss_dir}/loss_history.npy")
    thres_loss = 1.8 if args.alpha < 1 else 2.
    
    # Check if already converged
    if loss_history[-1] < thres_loss:
        print(f"Data already converged (loss={loss_history[-1]:.4f} < {thres_loss}) for D={D}, d={d}, N={args.num_steps}, seed={args.seed}")
        print(f"Exiting...")
        sys.exit()
    
    # Try to load checkpoint if exists
    iter0 = get_largest_iteration(save_checkpoint_dir)
    if iter0 > 0:
        print(f"Loading checkpoint at iter {iter0} for D={D}, d={d}, N={args.num_steps}, seed={args.seed}")
        np_params = np.load(f"{save_checkpoint_dir}/checkpoint_{iter0}_params.npz", allow_pickle=True)
        dict_params = extract_nested_dict(dict(np_params))
        dict_params = jax.tree_util.tree_map(jnp.array, dict_params)
        
        params_flat = {tuple(k.split('/')): dict_params[k] for k, v in dict_params.items()}
        params_unflat = unflatten_dict(params_flat)
        params = from_state_dict(state.params, params_unflat)
        state = state.replace(params=params)
        
        loss_history = list(loss_history)
        beta_history = list(np.load(f"{save_loss_dir}/beta_history.npy"))
        delta_history = list(np.load(f"{save_loss_dir}/delta_history.npy"))
        weight_history = list(np.load(f"{save_loss_dir}/weight_history.npy"))
    else:
        print(f"No checkpoint found (iter0={iter0}), starting fresh with new parameters")
        loss_history, beta_history, delta_history, weight_history = [], [], [], []
else:
    # No checkpoint loading or no existing data
    if not args.load_checkpoint and os.path.exists(save_data_dir):
        shutil.rmtree(save_data_dir)
        print(f"Removed existing data for D={D}, d={d}, N={args.num_steps}, seed={args.seed}")
    
    os.makedirs(save_data_dir, exist_ok=True)
    print(f"Starting fresh for D={D}, d={d}, N={args.num_steps}, seed={args.seed}")
    loss_history, beta_history, delta_history, weight_history = [], [], [], []

os.makedirs(save_loss_dir, exist_ok=True)
os.makedirs(save_checkpoint_dir, exist_ok=True)

save_token_dir = os.path.join(save_data_dir, "tokens")
os.makedirs(save_token_dir, exist_ok=True)
np.save(f"{save_token_dir}/tokens.npy", data_generator.X)


stop_count = 0


for iter in tqdm(range(iter0, iter0+config["learning"]["num_iters"]+1)):  # Train for 5 epochs
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, batch_size)
    batch_y, batch_X, batch_states, _ = data_generator.generate_batch_markov_chains(keys)
    if one_hot:
        batch_X = jax.nn.one_hot(batch_states[:, 0:-1], num_states)
    state, loss = train_step(state, batch_X, batch_y, iter, args.k)
    loss_history.append(loss)

    W1q, W1k, W1v = np.split(state.params["attention1st"]["attn"]['kernel'], [d, 2*d], axis=-1)
    W2q, W2k, W2v =  np.split(state.params["attention2nd"]["attn"]['kernel'], [2*d, 4*d], axis=-1)
    M_bar = W2q.dot(W2k.T)[:D, D:].dot(W1v.T)

    beta = np.mean(np.diag(X.dot(M_bar).dot(X.T)))

    if args.RelPosBias == 'mini':
        delta = state.params["attention1st"]["rel_pos_bias"]["rel_pos_bias"][0]
    else:
        delta = state.params["attention1st"]["rel_pos_bias"]["rel_pos_bias"][1]- np.mean(state.params["attention1st"]["rel_pos_bias"]["rel_pos_bias"][2:])

    beta_history.append(beta)
    delta_history.append(delta)
    weight_history.append(state.params["weights"])

    if (iter) % config["learning"]["save_checkpoint_frequency"]== 0:
        thres_loss = 2. if args.alpha==1.0 else 1.8
        if loss<thres_loss:
            stop_count += 1
        print(f"Iter {iter}, Loss: {loss:.6f}", f"beta: {beta:.6f}", f"delta: {delta:.6f}")
        np.savez(f"{save_checkpoint_dir}/checkpoint_{iter}_params.npz", **jax.tree_util.tree_map(np.array, state.params))
        os.makedirs(save_loss_dir, exist_ok=True)
        np.save(f"{save_loss_dir}/loss_history.npy", np.array(loss_history))
        np.save(f"{save_loss_dir}/delta_history.npy", np.array(delta_history))
        np.save(f"{save_loss_dir}/beta_history.npy", np.array(beta_history))
        np.save(f"{save_loss_dir}/weight_history.npy", np.array(weight_history))
        if stop_count==20:
            break

print(f"Time finished for D={D}, d={d}, N={args.num_steps}, seed={args.seed}: {time.time() - start_time}")


