import torch
import random
import numpy as np
import yaml
from ml_collections import ConfigDict
from copy import deepcopy
import importlib
import subprocess
import re
from typing import Callable, Tuple, Any, TYPE_CHECKING
from base.simple_gpt import Transformer, Attention, MLP
from pathlib import Path

from datetime import timedelta

def get_state_idx(config, t: int):
    """Get state idx for state at time t' <= t"""
    state_iters = np.array(config.log.state.state_iters)
    return len(state_iters[state_iters < t])-1

# for diffing config dicts
from ml_collections import ConfigDict
from fnmatch import fnmatchcase
from rich import print as rprint

def _is_cfgdict(x):
    return isinstance(x, ConfigDict)

def _match_exclude(path: str, patterns) -> bool:
    """Return True if path matches any of the glob patterns."""
    if not patterns:
        return False
    return any(fnmatchcase(path, pat) for pat in patterns)

def diff_configs_color(
    cfg1: ConfigDict,
    cfg2: ConfigDict,
    path: str = "",
    *,
    clean: bool = False,
    exclude: list[str] | None = None,
    _printer=rprint
):
    """
    Recursively print colorized differences between two ConfigDicts.

    Args:
        cfg1, cfg2: ConfigDicts to compare.
        path: internal; used to build dot-paths.
        clean: if True, suppress output for any differing paths that match `exclude`.
        exclude: list of glob patterns (on dot-paths) to skip. Examples:
                 ["log.state.state_iters", "log.metrics.*", "training.*"]
        _printer: print-like callable (defaults to rich.print).
    """
    keys1, keys2 = set(cfg1.keys()), set(cfg2.keys())

    for key in sorted(keys1 | keys2):
        subpath = f"{path}.{key}" if path else key

        # Only-in-first / only-in-second
        if key not in cfg2:
            if not (clean and _match_exclude(subpath, exclude)):
                _printer(f"[yellow]{subpath}[/yellow]: [green]{cfg1[key]!r}[/green] [dim]->[/dim] [red]<missing>[/red]")
            continue
        if key not in cfg1:
            if not (clean and _match_exclude(subpath, exclude)):
                _printer(f"[yellow]{subpath}[/yellow]: [red]<missing>[/red] [dim]->[/dim] [green]{cfg2[key]!r}[/green]")
            continue

        v1, v2 = cfg1[key], cfg2[key]

        # Recurse if both sides are nested configs
        if _is_cfgdict(v1) and _is_cfgdict(v2):
            diff_configs_color(v1, v2, subpath, clean=clean, exclude=exclude, _printer=_printer)
        else:
            if v1 != v2:
                if clean and _match_exclude(subpath, exclude):
                    continue
                _printer(f"[yellow]{subpath}[/yellow]: [red]{v1!r}[/red] [dim]->[/dim] [green]{v2!r}[/green]")

import torch.nn.functional as F
def KL(x: torch.Tensor, y: torch.Tensor):
    """
    Compute KL divergence D_KL(softmax(x) || softmax(y)) from logits x and y
    """
    px = F.softmax(x, dim=-1)
    py = F.softmax(y, dim=-1)
    out = (px * (px.log() - py.log())).sum(-1).mean()
    return out.item()

# get attribute if it exists, otherwise return None
def safe_getattr(obj, path, default=None):
    for p in path.split("."):
        try:
            obj = getattr(obj, p)
        except AttributeError:
            return default
    return obj

def has_key(cfg: ConfigDict, dotted_key: str) -> bool:
    """Check if a dotted key path exists in a ConfigDict."""
    parts = dotted_key.split(".")
    node = cfg
    for p in parts:
        if not isinstance(node, ConfigDict) or p not in node:
            return False
        node = node[p]
    return True

# for generic parameter setting in config
def dotted_set(obj, attr_path, value):
    parts = attr_path.split('.')
    for attr in parts[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, parts[-1], value)

def dotted_get(obj, attr_path):
    parts = attr_path.split('.')
    for attr in parts:
        obj = getattr(obj, attr)
    return obj

import os
def safe_save(obj, path):
    """
    Preemption safe
    """
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)  # Atomic on POSIX

import yaml
import os

def safe_yaml_dump(data, path):
    """
    Preemption safe
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    os.replace(tmp_path, path)

def time_display(val):
    """
    Print time rounded to nearest second
    """
    return str(timedelta(seconds = round(val)))

from datetime import datetime
from zoneinfo import ZoneInfo
import pytz

def print_eta(duration_seconds, timezone='America/New_York'):
    # Get current time in specified timezone
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    
    # Calculate end time
    eta = now + timedelta(seconds=duration_seconds)
    
    # Round to nearest minute
    if eta.second >= 30:
        eta = eta.replace(second=0, microsecond=0) + timedelta(minutes=1)
    else:
        eta = eta.replace(second=0, microsecond=0)
    
    # Format as M/D HH:MM
    return eta.strftime("%-m/%-d %H:%M")

def print_datetime():
    now_eastern = datetime.now(ZoneInfo("America/New_York"))
    return f"{now_eastern.month}/{now_eastern.day}, {now_eastern.strftime('%H:%M:%S')}"

import shutil
import importlib.util
import os
def copy_python_module(dotted_module_path: str, destination_dir: str):
    """
    Copy a python module given as a dotted module path to a directory.

    Args:
        dotted_module_path (str): The dotted path of the module to copy, e.g.,
                                  'my_package.my_module'.
        destination_dir (str): The directory where the module should be copied.
    """
    spec = importlib.util.find_spec(dotted_module_path)
    source_path = spec.origin

    os.makedirs(destination_dir, exist_ok=True)
    destination_path = os.path.join(destination_dir, os.path.basename(source_path))

    shutil.copy(source_path, destination_path)

import importlib.util
import os

def load_function_from_path(file_path, function_name):
    """
    Load a function from a Python file located at `file_path`.
    Args:
        file_path (str): The path to the Python file.
        function_name (str): The name of the function to load.
    Returns:
        Callable: The loaded function.
    """
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, function_name)

def compute_iters(compute: int, config: ConfigDict):
    d = config.model.n_embd
    l = len(config.model.architecture)//2
    n = config.model.block_size
    B = config.training.batch_size
    N = 12*d**2*l
    sigma = 6*(N+l*n*d) # FLOPs per token
    iters = int(compute//(n*B*sigma))
    return iters

def compute_FLOPs(config: ConfigDict):
    d = config.model.n_embd
    l = len(config.model.architecture)//2
    n = config.model.block_size
    B = config.training.batch_size
    iters = config.training.iters
    N = 12*d**2*l
    sigma = 6*(N+l*n*d) # FLOPS per token
    compute = iters*n*B*sigma
    return compute

def row_normalize(adj: torch.Tensor):
    """Row normalizes a single tensor or a batch of tensors with batch dimension = 0.
    If a vertex is isolated, returns 0 vector in that row"""
    norm = adj.sum(dim = -1, keepdim = True)
    out = adj/norm

    return out

def load_config_and_adj_pool(path: str) -> ConfigDict:
    """Load a config and add adj_pool from given directory"""
    with open(f'{path}/config.yaml', 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config_dict['task']['adj_pool'] = torch.load(f'{path}/adj_pool.pt')
    config = ConfigDict(config_dict)

    return config

def load_config_and_task_pool(path: str) -> ConfigDict:
    """Load a config and add task_pool from given directory"""
    with open(f'{path}/config.yaml', 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config_dict['task']['task_pool'] = torch.load(f'{path}/task_pool.pt').to(config_dict['task']['device']).double()
    config = ConfigDict(config_dict)

    return config

def load_config(path: str) -> ConfigDict:
    """Load a config located at given path; does not add the adj_pool field
    
    Made backwards compatible with folders that save config as yaml instead of python module.
    """
    path = Path(path)
    if path.suffix == '.yaml':
        if path.exists():
            with open(path, 'r') as f:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            config = ConfigDict(config_dict)
            return config
        else:
            # replace .yaml with .py
            path = path.with_suffix('.py')
    get_config = load_function_from_path(path, 'get_config')
    config = get_config()

    return config

def dynamic_import(module_path, attr_name=None):
    module = importlib.import_module(module_path)
    if attr_name:
        return getattr(module, attr_name)
    return module

import argparse
import os

class EnvDefault(argparse.Action):
    def __init__(self, envvar, required=True, default=None, **kwargs):
        if envvar in os.environ:
            default = os.environ[envvar]
            required = False
        super().__init__(default=default, required=required, **kwargs)
        self.envvar = envvar

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)

def submit_slurm_job(script_path, extra_args=None):
    """
    Submit a job to SLURM using sbatch.
    Args:
        script_path (str): Path to the SLURM script.
        extra_args (list, optional): Additional arguments for sbatch.
    Returns:
        str: Job ID if submission is successful, None otherwise.
    """
    cmd = ['sbatch']
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(script_path)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if match:
            job_id = match.group(1)
            print(f"Job submitted successfully. Job ID: {job_id}")
            return job_id
        else:
            print("Job submitted but could not parse Job ID.")
    else:
        print(f"Error submitting job: {result.stderr}")
    
    return None

def run_python_script(script_path, extra_args=None, streaming=True):
    cmd = ['python', script_path]
    if extra_args:
        cmd.extend(extra_args)
    
    if streaming:
        return run_python_script_streaming(script_path, extra_args)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("Script executed successfully.")
            return result.stdout
        else:
            print(f"Error executing script: {result.stderr}")
            print(f"Script output: {result.stdout}")
        
        return None

def run_python_script_streaming(script_path, extra_args=None):
    cmd = ['python', '-u', script_path]  # -u ensures unbuffered output
    if extra_args:
        cmd.extend(extra_args)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # combine stdout and stderr
        text=True,
        bufsize=1  # line-buffered
    )

    print("Running script...\n")
    for line in process.stdout:
        print(line, end='')  # already includes newline

    process.wait()
    if process.returncode == 0:
        print("\nScript executed successfully.")
    else:
        print(f"\nScript failed with return code {process.returncode}")

def load_all_model(dir: str, get_model: Callable, task_model: Callable) -> Tuple[Transformer, Any, ConfigDict, ConfigDict]:
    """Load a model, task_model, and val config from a directory."""
    state_dict = torch.load(f'{dir}/state_dict.pt')
    adj_pool = torch.load(f'{dir}/adj_pool.pt')
    with open(f'{dir}/config.yaml', 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config_dict['task']['adj_pool'] = adj_pool
    config = ConfigDict(config_dict)

    model = get_model(config)
    model.load_state_dict(state_dict)
    model.eval()

    task_model = task_model(config.task)
    val_config = deepcopy(config.task)
    val_config.adj_pool = None

    out = {'model': model, 'task_model': task_model, 'config': config, 'val_config': val_config}

    return out

def get_random_state():
    # with torch.random.fork_rng():
    torch_state = torch.random.get_rng_state()
    return {'numpy': np.random.get_state(),
            'torch': torch_state,
            'random': random.getstate(),
            'cuda': {str(i): torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())}
        }

def set_random_state(state):
    """Set random states for numpy, torch (forked to all devices), and random modules."""
    np.random.set_state(state['numpy'])
    # with torch.random.fork_rng():
    torch.random.set_rng_state(state['torch'])
    for device_id, cuda_state in state.get('cuda', {}).items():
        torch.cuda.set_rng_state(cuda_state, device=int(device_id))
    random.setstate(state['random'])

def load_checkpoint(checkpoint: dict, model: Transformer, optimizer: torch.optim.Optimizer):
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    set_random_state(checkpoint['random_state'])

    return model, optimizer

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

res_func_lookup = {Attention: 'sa', MLP: 'ffwd'}

def moving_average(y: torch.Tensor, window: int):
    if isinstance(y, list):
        y = torch.tensor(y)
    out = [torch.mean(y[i:(i+window)]) for i in range(len(y)-window+1)]
    out = torch.tensor(out)
    return out

def ranged_moving_average(x: torch.Tensor, y: torch.Tensor, window: tuple, range: tuple):
    if isinstance(x, list):
        x = torch.tensor(x)
    if isinstance(y, list):
        y = torch.tensor(y)
    xx = []
    yy = []
    for i, interval in enumerate(range):
        avg_min, avg_max = interval
        xx.append(x[avg_min+window[i]-1:avg_max])
        yy.append(moving_average(y[avg_min:avg_max], window[i]))
    xx = torch.cat(xx)
    yy = torch.cat(yy)
    return xx, yy

class attn_pattern_tracer(object):
    "Record the attention patterns of each attention layer"
    def __init__(self, model: Transformer):
        self._model = model
        self._patterns = {}
        self._hooks = []

    def _register_forward_hooks(self):
        def get_activations(i):
            def hook(module, args, output):
                hidden_state = output.cpu()
                self._activations[i] = hidden_state
            
            return hook
        
        for i, module in enumerate(self._model.blocks):
            if module.__class__ == MLP:
                continue
            attr = res_func_lookup[module.__class__]
            res_function = getattr(module, attr)
            h = res_function.register_forward_hook(get_activations(i), prepend=False)
            self._hooks.append(h)
    
    def __enter__(self):
        self._register_forward_hooks()

        return self
    
    def __exit__(self, *args, **kwargs):
        for h in self._hooks:
            h.remove()

class res_activations_tracer(object):
    "Record the attention patterns of each attention layer"
    def __init__(self, model: Transformer):
        self._model = model
        self._patterns = {}
        self._hooks = []

    def _register_forward_hooks(self):
        def get_activations(i):
            def hook(module, args, output):
                hidden_state = output.cpu()
                self._activations[i] = hidden_state
            
            return hook
        
        for i, module in enumerate(self._model.blocks):
            if module.__class__ == MLP:
                continue
            attr = res_func_lookup[module.__class__]
            res_function = getattr(module, attr)
            h = res_function.register_forward_hook(get_activations(i), prepend=False)
            self._hooks.append(h)
    
    def __enter__(self):
        self._register_forward_hooks()

        return self
    
    def __exit__(self, *args, **kwargs):
        for h in self._hooks:
            h.remove()