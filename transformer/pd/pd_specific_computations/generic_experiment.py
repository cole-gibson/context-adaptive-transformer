from pathlib import Path
import torch
from base.seed import set_seed
import base.utils as u

class BaseRunner:
    def __init__(
        self,
        base_dir: str,
        get_model,
        TaskModel,
        random_seed: int,
        n_states: int,
        state_interval: int,
        data_dir_template: str = "{k}_{n}"
    ):
        # store and normalize paths
        self.base_dir: Path = Path(base_dir)
        self.get_model = get_model
        self.TaskModel = TaskModel
        self.random_seed = random_seed
        self.n_states = n_states
        self.state_interval = state_interval
        # customizable subdirectory naming: e.g. "5_10" by default
        self.data_dir_template = data_dir_template

        # load exp_config and set up krange/nrange
        exp_config = u.load_config(self.base_dir / "exp_config.yaml")
        try:
            # numeric lmin/lmax style
            exp_config.lmin
            self.nrange = torch.arange(
                exp_config.lmin,
                exp_config.lmax + exp_config.lstep,
                exp_config.lstep
            ).list()
        except AttributeError:
            # already an nrange attribute
            self.nrange = exp_config.nrange
        self.krange = exp_config.krange

        # reproducibility
        set_seed(self.random_seed)

    def get_data_dir(self, k, n) -> Path:
        # centralizes how we build the data directory path
        return self.base_dir / "data" / self.data_dir_template.format(k=k, n=n)

    # hooks for subclasses
    def k_step(self, k):
        raise NotImplementedError

    def n_step(self, n):
        raise NotImplementedError

    def s_step(self, idx):
        raise NotImplementedError

    @torch.no_grad()
    def run(self):
        for k in self.krange:
            print(f"k: {k}")
            self.k_step(k)
            for n in self.nrange:
                print(f"n: {n}")
                self.n_step(n)
                for idx in torch.arange(0, self.n_states, self.state_interval):
                    self.s_step(idx)
        self._save_output()

    @torch.no_grad()
    def run_min(self, krange):
        # same as run, but bail out of the s_step loop on FileNotFound
        self.krange = krange
        for k in self.krange:
            print(f"k: {k}")
            self.k_step(k)
            for n in self.nrange:
                print(f"n: {n}")
                self.n_step(n)
                for idx in torch.arange(0, self.n_states, self.state_interval):
                    flag = self.s_step(idx)
                    if flag:
                        break
        self._save_output_min(k)

    # output hooks
    def _save_output(self):
        raise NotImplementedError

    def _save_output_min(self, last_k):
        raise NotImplementedError