from dataclasses import dataclass, field
from base.simple_gpt import Transformer
from ml_collections import ConfigDict
from base.tasks.markov_model import MarkovModel
from base.utils import safe_save, safe_yaml_dump
import torch
import os
import yaml
from pathlib import Path

def mkdir_versioned(base_name):
    version = 0
    dir_name = f'{base_name}:v{version}'
    while os.path.exists(dir_name):
        version += 1
        dir_name = f'{base_name}:v{version}'
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        dir_name = mkdir_versioned(base_name)
    return dir_name

def clean_dict(d: dict):
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            d[key] = f'torch.tensor: save manually if desired'
        elif isinstance(value, dict):
            d[key] = clean_dict(value)
    return d

class Logger():
    def __init__(self, config: ConfigDict, model: Transformer):
        self.config = config.log
        self.set_interval()
        self.full_config = clean_dict(config.to_dict())
        self.log = {'step': [], 'loss': []}
        self.model_dir = os.path.join(self.config.project_dir, self.config.model_name)
        os.makedirs(self.model_dir, exist_ok = True)
        self.state_dir = os.path.join(self.model_dir, 'state')
        state_dir = Path(self.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)

        pt_files = [
            f for f in state_dir.glob("*.pt")
            if f.stem.isdigit()
        ]

        if pt_files:
            self.state_counter = max(int(f.stem) for f in pt_files) + 1
        else:
            self.state_counter = 0

    def set_interval(self):
        if self.config.state.fix_n:
            self.state_iters = [round(i*self.config.iters/(self.config.state.n-1)) for i in range(self.config.state.n)]
            if self.state_iters[-1] != self.config.iters-1:
                self.state_iters[-1] = self.config.iters-1
        elif self.config.state.state_iters is not None:
            self.state_iters = self.config.state.state_iters + [self.config.iters-1]
        else:
            self.state_iters = []
            for idx, rate in enumerate(self.config.state.rates):
                interval = self.config.state.intervals[idx]
                if len(interval) == 2:
                    min, max = interval
                elif len(interval) == 1:
                    min, max = interval[0], self.config.iters
                self.state_iters += [i for i in range(min, max, rate)]
                self.state_iters += [self.config.iters-1]
    
    def load(self, log):
        """
        Load a log from a partially trained model
        """
        self.log = log

    def entry(self, iter: int, loss: float):
        """
        If toggle is true, record model training loss
        """
        if self.config.toggle:
            self.log['step'].append(iter)
            self.log['loss'].append(loss)
    
    def rec_state(self, iter: int, model: Transformer):
        """
        Record state
        """
        if self.config.state.toggle:
            if iter in self.state_iters:
                safe_save({'iter': iter, 'state': model.state_dict()}, f'{self.state_dir}/{self.state_counter}.pt')
                self.state_counter += 1
    
    def bundle_and_save(self, iter: int, task_model: MarkovModel):
        safe_save(task_model.task_pool, f'{self.model_dir}/task_pool.pt')
        if self.config.toggle:
            safe_save(self.log, f'{self.model_dir}/log.pt')
        safe_yaml_dump(self.full_config, f'{self.model_dir}/config.yaml')