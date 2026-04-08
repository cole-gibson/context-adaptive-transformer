import argparse
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from src.data_generator import DataGenerator
import yaml, os
from tqdm import tqdm
import pickle
import numpy as np

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run Markov Chain Model")
parser.add_argument("--seed", type=int, default=10, help="Random seed for data generation")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps in the Markov chain")
parser.add_argument("--num_states", type=int, default=10, help="Number of states in the Markov chain")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
args = parser.parse_args()
tolerance = 1.8
# Use command-line arguments to configure the model
seed = args.seed
N = args.num_steps
batch_size = args.batch_size

default_config = yaml.safe_load(open("config/config_mini_2params.yaml"))
config = default_config
config["data_settings"]["seed"] = seed
config["data_settings"]["num_steps"] = N
config["data_settings"]["num_states"] = args.num_states
config["data_settings"]["alpha"] = 0.5
config["learning"]["batch_size"] = batch_size
num_states = config["data_settings"]["num_states"]
data_generator = DataGenerator(**config["data_settings"])

def model_fn(seq, params):
    """
    Define the toy model function.
    """
    # Extract sequence and labels
    N = seq.shape[0]

    # Extract parameters
    beta = params["beta"]
    delta = params["delta"]

    # Compute term1
    term1 = beta * (jnp.exp(delta)-1) / (jnp.exp(delta) + jnp.arange(1, N) - 1)
    term1 = jnp.where(seq[0:-1] == seq[-1], term1, 0)

    # Compute term2
    term2 = beta * jnp.cumsum(jnp.concatenate([jnp.array([0]), seq[0:-2] == seq[-1]])) / \
            (jnp.exp(delta) + jnp.arange(1, N) - 1)

    # Compute exp_term and its sum
    exp_term = jnp.exp(term1 + term2)
    sum_exp_term = jnp.sum(exp_term)

    # Compute states and sequence states
    states = jnp.arange(num_states)  # Array of all possible states
    seq_states = seq[1:]  # Sequence to compare against states

    # Create a mask for each state
    mask = seq_states[:, None] == states  # Shape: (N-1, num_states)

    # Apply the mask to exp_term
    masked_exp_term = mask * exp_term[:, None]  # Shape: (N-1, num_states)

    # Sum over the sequence dimension for each state
    state_sums = jnp.sum(masked_exp_term, axis=0)  # Shape: (num_states,)

    # Normalize by the total sum
    normalized_sums = state_sums / sum_exp_term
    return normalized_sums

@jax.jit
def loss_fn(params, seq, y):
    """
    Compute the loss function.
    """
    # Compute model predictions
    pred = model_fn(seq, params)
    y_one_hot = jax.nn.one_hot(y, num_states)
    # Compute the loss (negative log likelihood)
    loss = -y_one_hot.dot(jnp.log(pred + 1e-10))
    return loss

# Vectorize the loss function to handle batch data
batch_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0))

# Define a function to compute the average batch loss
def avg_batch_loss(params, batch_X, batch_y):
    return jnp.mean(batch_loss_fn(params, batch_X, batch_y))

# Compute gradients of the average batch loss
grad_fn = jax.grad(avg_batch_loss)
save_data_dir = os.path.abspath(os.path.join(
    config["learning"]["save_path"],
    f"N_{args.num_steps}_C_{args.num_states}_alpha_{config["data_settings"]["alpha"]}/trial_{seed}"
))

os.path.join(save_data_dir, "loss")
save_data = {}
save_data["loss"] = []
save_data["beta"], save_data["delta"] = [], []


key = jax.random.PRNGKey(seed)
key, key1, key2 = jax.random.split(key, 3)
# Example usage
params = {
    "beta": 0.01,  # Initialize beta
    "delta": 0.01 ,  # Initialize delta
}

learning_rate = 5.
n_steps = 100000
prev_params = None 
# Run optimization
count = 0
for step in tqdm(range(n_steps)):
    key, key1 = jax.random.split(key)
    keys = jax.random.split(key1, batch_size)
    _, _, batch_states = data_generator.generate_batch_markov_chains(keys)
    batch_X, batch_y= batch_states[:, 0:-1], batch_states[:, -1]
    # Compute gradients
    grads = grad_fn(params, batch_X, batch_y)
    # Update parameters
    new_params = {k: params[k] - learning_rate * grads[k] for k in params}
    loss = avg_batch_loss(params, batch_X, batch_y)

    # Update for the next iteration
    prev_params = params
    params = new_params
    save_data["loss"].append(loss)
    save_data["beta"].append(params["beta"])
    save_data["delta"].append(params["delta"])
    if step % 500 == 0:
        print(f"Step {step}, Loss: {save_data['loss'][-1]}, beta: {params['beta']}, delta: {params['delta']}")
        if loss < tolerance:
            count += 1
    if count > 5:
        print("Convergence achieved.")
        break
print(f"Optimized Parameters: {params}")
np.save(f"{save_loss_dir}/loss_history.npy", np.array(save_data["loss"]))
np.save(f"{save_loss_dir}/delta_history.npy", np.array(save_data["delta"]))
np.save(f"{save_loss_dir}/beta_history.npy", np.array(save_data["beta"]))
print("data saved for N=", N, "seed=", seed)