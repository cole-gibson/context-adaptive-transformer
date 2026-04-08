import numpy as np
import yaml, os
from scipy.stats import entropy
import re

config = yaml.safe_load(open("config/config_reduce.yaml"))
data_dir = config["learning"]["save_path"]
def load_data(file_dir):
    """
    Load data from the specified directory, including loss, beta, delta, and weights.

    Args:
        file_dir (str): Directory containing the data files.

    Returns:
        dict: A dictionary containing the loaded data or None if no data is found.
    """
    data = {}
    final_iter = get_largest_iteration(os.path.join(file_dir, "loss"))

    def load_file(file_name):
        return np.load(os.path.join(file_dir, "loss", file_name))

    try:
        if os.path.exists(os.path.join(file_dir, "loss", "loss_history.npy")):
            data = {
                "loss": load_file("loss_history.npy"),
                "beta": load_file("beta_history.npy"),
                "delta": load_file("delta_history.npy"),
                "weights": load_file("weight_history.npy"),
            }
        elif final_iter > 0:
            data = {
                "loss": load_file(f"loss_history_iter_{final_iter}.npy"),
                "beta": load_file(f"beta_history_iter_{final_iter}.npy"),
                "delta": load_file(f"delta_history_iter_{final_iter}.npy"),
            }
            weight_file = f"weight_history_iter_{final_iter}.npy"
            weight_path = os.path.join(file_dir, "loss", weight_file)
            if os.path.exists(weight_path):
                data["weights"] = load_file(weight_file)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None

    return data if data else None

def get_largest_iteration(save_checkpoint_dir):
    """
    Finds the largest iteration number from loss_history_iter_{iter}.npy files in the given directory.

    Args:
        save_loss_dir (str): Path to the directory containing loss history files.

    Returns:
        int: The largest iteration number found, or None if no files are found.
    """
    # Regular expression to match loss_history_iter_{iter}.npy files
    pattern = re.compile(r"checkpoint_(\d+)_params\.npz")

    iters = []
    for fname in os.listdir(save_checkpoint_dir):
        match = pattern.match(fname)
        if match:
            iters.append(int(match.group(1)))
    return max(iters) if iters else 0

def get_checkpoint_dir(N, D, d, C, alpha, max_dist=None, mode = "mlp", seed = 1, rel_pos_bias=None, optimizer=None):
    file_dir = get_file_dir(N, D, d, C, alpha, max_dist=max_dist, mode=mode, seed=seed, rel_pos_bias=rel_pos_bias, optimizer=optimizer)
    return os.path.abspath(os.path.join(file_dir, "checkpoints"))

def load_losses(file_dir, mode):
    losses_list = []
    file_mode_dir = os.path.join(file_dir, mode)
    for root, dirs, files in os.walk(file_mode_dir):
        for file in files:
            if file.endswith('.npy'):
                losses_list.append(np.load(os.path.join(root, file)))
    return np.asarray(losses_list)

def get_loss(N, D, d, C, alpha, max_dist, mode, seed, iter, rel_pos_bias=None, optimizer=None):
    file_mode_dir = get_file_dir(N, D, d, C, alpha, max_dist=max_dist, mode=mode, seed=seed, rel_pos_bias=rel_pos_bias, optimizer=optimizer)
    loss = np.load(os.path.join(file_mode_dir, "loss", f"loss_history_iter_{iter}.npy"))
    return loss
    

def get_file_dir(N, D, d, C, alpha, max_dist=None, mode=None, seed=None, rel_pos_bias=None, optimizer=None):
    """
    Constructs a directory path for saving/loading experiment results based on configuration.

    Args:
        N (int): Number of steps.
        D (int): Input dimension.
        d (int): Model dimension.
        C (int): Number of states/classes.
        alpha (float): Alpha parameter.
        max_dist (int, optional): Maximum distance parameter.
        mode (str, optional): Model mode/architecture.
        seed (int, optional): Random seed.
        rel_pos_bias (str, optional): Relative position bias type.
        optimizer (str, optional): Optimizer name.

    Returns:
        str: Constructed directory path.
    """
    base = f"N_{N}_D_{D}_d_{d}_C_{C}_alpha_{alpha}"
    if max_dist is not None:
        base += f"_max_distance_{max_dist}"
    base_path = os.path.join(data_dir, base)

    # Compose subdirectory name
    subdir = None
    if mode is not None:
        subdir = mode
        if rel_pos_bias is not None:
            subdir += f"_RelPosBias_{rel_pos_bias}"
        if optimizer is not None:
            subdir += f"_{optimizer}"

    # Add subdirectory and trial if needed
    if subdir is not None and seed is not None:
        base_path = os.path.join(base_path, subdir, f"trial_{seed}")

    return base_path

def extract_nested_dict(loaded_params):
    """Extract nested dictionary structure from loaded parameters.
    
    Args:
        loaded_params: NumPy array with dtype=object or dictionary
        
    Returns:
        Extracted nested dictionary structure
    """
    # Case 1: NumPy object array containing dictionary
    if isinstance(loaded_params, np.ndarray) and loaded_params.dtype == np.dtype('O'):
        if hasattr(loaded_params, 'keys'):
            return {key: extract_nested_dict(loaded_params[key]) for key in loaded_params.keys()}
        return loaded_params.item()  # Convert single-element array to scalar
        
    # Case 2: Regular dictionary
    elif isinstance(loaded_params, dict):
        return {key: extract_nested_dict(value) for key, value in loaded_params.items()}
        
    # Case 3: Base case - return value as is
    return loaded_params
    
def get_params(N, D, d, C, alpha, max_dist,mode = "mlp", seed = 1, iter = 0, rel_pos_bias=None,  optimizer=None):
    checkpoint_dir = get_checkpoint_dir(N, D, d, C, alpha, max_dist=max_dist, mode=mode, seed=seed, rel_pos_bias=rel_pos_bias, optimizer=optimizer)
    loaded_params = np.load(os.path.join(checkpoint_dir, f"checkpoint_{iter}_params.npz"), allow_pickle=True)
    params = {key: extract_nested_dict(value) for key, value in loaded_params.items()}
    return params

def get_moving_avg(ts, window_size = 25):
    kernel = np.ones(window_size) / window_size
    return np.convolve(ts, kernel, mode='same')


def calculate_orders(params, d, D, X, rel_pos_bias=None):
    rel_pos_bias_array = params["attention1st"]["rel_pos_bias"]["rel_pos_bias"]
    weights = np.exp(rel_pos_bias_array)
    probs = weights/weights.sum()
    H = entropy(probs, base=2) 
    if rel_pos_bias is None:
        delta = rel_pos_bias_array[1] - np.mean(rel_pos_bias_array[1:])
    elif rel_pos_bias == "full":
        delta = rel_pos_bias_array[1] - np.mean(rel_pos_bias_array[1:])
    elif rel_pos_bias == "mini":
        delta = rel_pos_bias_array[0] 
    else:
        print("Invalid rel_pos_bias")
        delta = 0
    W1q, W1k, W1v = np.split(params["attention1st"]["attn"]['kernel'], [d, 2*d], axis=-1)
    W2q, W2k, W2v =  np.split(params["attention2nd"]["attn"]['kernel'], [2*d, 4*d], axis=-1)
    M_bar = W2q.dot(W2k.T)[:D, D:]
    beta = np.diag(M_bar).mean() - (M_bar-np.diag(np.diag(M_bar))).mean()
    return H, beta, delta