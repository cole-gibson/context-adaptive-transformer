from pathlib import Path
import torch, time
import base.utils as u
from typing import Callable
from datetime import timedelta

class BaseRunner:
    def __init__(
        self,
        base_dir: str,
        get_model,
        get_task_model,
        n_states: int,
        state_interval: int,
        data_dir_template: Callable
    ):
        # store and normalize paths
        self.base_dir: Path = Path(base_dir)
        self.get_model = get_model
        self.get_task_model = get_task_model
        self.n_states = n_states
        self.state_interval = state_interval
        # customizable subdirectory naming: e.g. "5_10" by default
        self.data_dir_template = data_dir_template

        # load exp_config and set up krange/nrange
        exp_config = u.load_config(self.base_dir / "exp_config.yaml")
        self.nrange = exp_config.nrange
        self.nrange.sort()  # assume sorted ascending later
        self.param = exp_config.param
        self.param_name = exp_config.param_name
        self.paramrange = exp_config.paramrange
        self.seedrange = exp_config.seedrange

    def get_data_dir(self) -> Path:
        # centralizes how we build the data directory path
        return self.base_dir / "data" / self.data_dir_template(self.seed, self.n, self.param_name, self.param_value)

    # hooks for subclasses
    def seed_step(self, seed):
        raise NotImplementedError

    def param_step(self, param_value):
        raise NotImplementedError

    def n_step(self, n):
        raise NotImplementedError

    def t_step(self, idx):
        raise NotImplementedError

    @torch.no_grad()
    def run(self):
        self.iterate()
        self._save_output()

    @torch.no_grad()
    def run_min(self, paramrange, seedrange, nrange):
        self.seedrange = seedrange
        self.paramrange = paramrange
        self.nrange = nrange
        self.iterate()
        self._save_output_min()
    
    @torch.no_grad()
    def iterate(self):
        start_time = time.time()
        for seed in self.seedrange:
            print(f"seed: {seed}")
            self.seed_step(seed)
            for param_value in self.paramrange:
                print(f"{self.param_name}: {param_value}")
                self.param_step(param_value)
                for n in self.nrange:
                    print(f"n: {n}")
                    try:
                        self.n_step(n)  # for bsearch directories without contiguous in params
                    except FileNotFoundError:
                        continue
                    for idx in torch.arange(0, self.n_states, self.state_interval):
                        flag = self.t_step(idx)
                        if flag:
                            break
        print(f"runtime: {u.time_display(time.time() - start_time)} \n")

    # output hooks
    def _save_output(self):
        raise NotImplementedError

    def _save_output_min(self):
        raise NotImplementedError

    def _clean_output(self):
        raise NotImplementedError