from base.trainer import Trainer
from ml_collections import ConfigDict
from base.utils import dynamic_import, load_function_from_path, dotted_get, dotted_set, safe_getattr
from pathlib import Path
import torch
import subprocess
import inspect

def name(exp_config: ConfigDict, config: ConfigDict):
    return f'S{config.model.seed}_N{config.training.context_len}_{exp_config.param_name}{dotted_get(config, exp_config.param)}'

def general_init(exp_config: ConfigDict, array_idx):
    base_dir = Path(exp_config.base_dir)    # create base_dir object

    # load config from base_dir
    config_path = base_dir / 'config.py'
    get_config = load_function_from_path(config_path, 'get_config')
    config = get_config()  # load exp_config from base_dir

    # create k list corresponding to this chunk (handles parallel dispatch of phase_diagram_parallel)
    start = exp_config.training.chunk_size*int(array_idx)
    param_list = exp_config.paramrange[start:start + exp_config.training.chunk_size]   # unused, for parallel dispatch

    project_dir = base_dir / 'data'
    config.log.project_dir = str(project_dir)   # update config project directory
    project_dir.mkdir(parents = True, exist_ok = True)    # create data directory where models are stored
    # subprocess.run(["sudo", "chmod", "777", str(project_dir)], check=True)
    config.log.tmp_dir = config.log.project_dir

    # load get_model from exp_config
    get_model = dynamic_import(exp_config.model, 'get_model')

    print(inspect.getsourcefile(get_model))

    get_task_model = dynamic_import(exp_config.task_model, 'get_task_model')

    return config, get_model, get_task_model

def set_config(exp_config, config, param_idx: int, seed_idx: int, n_idx: int):
    seed = exp_config.seedrange[seed_idx]

    if safe_getattr(exp_config, 'fix_seeds', False):
        fix_seeds = exp_config.fix_seeds
        if 'model' in fix_seeds:
            config.model.seed = seed
        if 'training' in fix_seeds:
            config.training.seed = seed
        if 'task' in fix_seeds:
            config.task.seed = seed
    else:
        config.model.seed = seed
        config.training.seed = seed
        config.task.seed = seed
    
    param_value = exp_config.paramrange[param_idx]
    n = exp_config.nrange[n_idx]
    config.training.context_len = n
    dotted_set(config, exp_config.param, param_value)
    
    if config.training.to_convergence:
        conv_dict = torch.load(config.training.convergence_dict_path)
        config.training.convergence = config.training.convergence_mult*conv_dict[n]

    config.log.model_name = name(exp_config, config) # set name

    return config

def init(exp_config, param_idx: int, seed_idx: int, n_idx: int, array_idx):
    config, get_model, get_task_model = general_init(exp_config, array_idx)
    config = set_config(exp_config, config, param_idx, seed_idx, n_idx)
    model = get_model(config)
    task_model = get_task_model(config)

    return config, model, task_model

def train_model(exp_config: ConfigDict, param_idx: int, seed_idx: int, n_idx: int, array_idx = 0):
    config, model, task_model = init(exp_config, param_idx, seed_idx, n_idx, array_idx)
    train = Trainer(model, config, task_model)
    train.begin()