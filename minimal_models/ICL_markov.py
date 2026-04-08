import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
from jaxlib import xla_extension
from flax.training import train_state
import optax
from functools import partial
from src.data_generator import DataGenerator
from src.model import TransformerModel, compute_loss_fn
from tqdm import tqdm
from utils.aux import get_largest_iteration
import yaml, argparse, os, logging, shutil
import numpy as np
import time, os, sys
"""
# python ICL_markov.py --seed 0 --mode "mlp" --num_steps 1000 --D 128 --max_distance 128 --batch_size 256 --alpha 1.
"""
start_time = time.time()

default_config = yaml.safe_load(open("config/config.yaml"))

parser = argparse.ArgumentParser(description='Plot syllable window')
parser.add_argument('--D', type=int, default=128, help='token embedding dimension')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--mode', type=str, default="mlp", choices=["linear", "fixed_linear", "linear_reduce", "fixed_linear_bias", "mlp", "mlp_reduce"], help='mode type')
parser.add_argument('--num_steps', type=int, default=128, help='number of steps')
parser.add_argument('--d', type=int, default=128, help='query/key/value dimension')
parser.add_argument('--num_states', type=int, default=10, help='number of states')
parser.add_argument('--alpha', type=float, default=1., help='concentration parameter')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--max_distance', type=int, default=128, help='batch size')
parser.add_argument('--RelPosBias', type=str, default='full', help='relative position bias type')
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

data_generator = DataGenerator(**config["data_settings"])

seed = config["data_settings"]["seed"]
batch_size = config["learning"]["batch_size"]
mode = config["learning"]["mode"]

key = jax.random.PRNGKey(seed)
keys = jax.random.split(key, batch_size)
batch_y, batch_X, batch_states = data_generator.generate_batch_markov_chains(keys)
X = data_generator.X

compute_loss = partial(compute_loss_fn, J=data_generator.X, num_states=data_generator.num_states, lam=1.)

@jax.jit
def train_step(state, batch_X, batch_y):
    loss_fn = lambda params: compute_loss(params, state.apply_fn, batch_X, batch_y)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

D, d = num_states, num_states

save_data_dir = os.path.abspath(os.path.join(config["learning"]["save_path"], f"N_{args.num_steps}_D_{D}_d_{d}_C_{args.num_states}_alpha_{args.alpha}_max_distance_{args.max_distance}/{mode}_RelPosBias_{args.RelPosBias}/trial_{seed}"))
os.makedirs(save_data_dir, exist_ok=True)
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


save_checkpoint_dir = os.path.join(save_data_dir, "checkpoints")
save_loss_dir = os.path.join(save_data_dir, "loss")
save_token_dir = os.path.join(save_data_dir, "tokens")

if os.path.exists(save_loss_dir):
    print(save_data_dir)
    max_iter_cur = get_largest_iteration(save_loss_dir)
    if max_iter_cur >=10000:
        print(f"Data already exists for D={D}, d={d}, N={args.num_steps}, seed={args.seed}")
        print(f"exiting...")
        sys.exit()
    else:
        shutil.rmtree(save_data_dir)
        os.makedirs(save_data_dir, exist_ok=True)
        print(f"Data removed for D={D}, d={d}, N={args.num_steps}, seed={args.seed}")

os.makedirs(save_checkpoint_dir, exist_ok=True)
os.makedirs(save_token_dir, exist_ok=True)
loss_history = []

try:
    model = TransformerModel(D=D, d=d, C=data_generator.num_states, seq_len=batch_X.shape[1], max_distance = args.max_distance, architecture=args.mode, RelPosBias=args.RelPosBias)

    key, subkey = jax.random.split(key)
    variables = model.init(subkey, batch_X, data_generator.X)
    params = variables["params"]
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(config["learning"]["lr"])
        #tx = optax.sgd(learning_rate=0.0001, momentum=0.9)
    )

    for iter in tqdm(range(config["learning"]["num_iters"]+1)):  # Train for 5 epochs
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)
        batch_y, batch_X, _ = data_generator.generate_batch_markov_chains(keys)
        state, loss = train_step(state, batch_X, batch_y)
        loss_history.append(loss)
        if (iter) % config["learning"]["save_checkpoint_frequency"]== 0:
            print(f"Iter {iter}, Loss: {loss:.4f}")
            np.savez(f"{save_checkpoint_dir}/checkpoint_{iter}_params.npz", **jax.tree_util.tree_map(np.array, state.params))  
            os.makedirs(save_loss_dir, exist_ok=True)
            np.save(f"{save_loss_dir}/loss_history_iter_{iter}.npy", np.array(loss_history))
            np.save(f"{save_token_dir}/tokens.npy", data_generator.X)
    print(f"Time finished for D={D}, d={d}, N={args.num_steps}, seed={args.seed}: {time.time() - start_time}")   
except xla_extension.XlaRuntimeError as xla_err:
    logging.error(f"""
    Training Error Summary:
    ----------------------
    Mode: {mode}
    Seed: {seed}
    D (embedding dimension): {D}
    Number of steps: {config['data_settings']['num_steps']}
    Number of states: {config['data_settings']['num_states']}
    Alpha: {config['data_settings']['alpha']}
    Batch size: {batch_size}
    Current iteration: {iter}
    Last recorded loss: {loss_history[-1] if loss_history else 'N/A'}
    Error Details:
    -------------
    {str(xla_err)}
    """, exc_info=True)