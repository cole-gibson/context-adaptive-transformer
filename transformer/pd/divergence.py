from base.estimators.markov_model_ar import init_all_estimators
from base.estimators.markov_model_fix_N import init_all_estimators_last
import base.utils as u
from pd.generic_experiment import BaseRunner
from typing import Callable
from base.seed import set_seed
from pandas import DataFrame
import torch
import torch.nn.functional as F

def kl(x: torch.Tensor, y: torch.Tensor):
    """
    Compute KL divergence D(x | y) from logits x and y
    """
    device = x.device
    B, C = x.shape
    px = F.softmax(x, dim = -1)
    py = F.softmax(y, dim = -1)

    out = px*torch.log(px/py)
    out = out.sum(-1).mean()
    return out.item()

class Divergence(BaseRunner):
    def __init__(
        self,
        base_dir: str,
        get_model,
        get_task_model,
        max_batch_size: int,
        seq_per_task: int,
        val_repeat: int,
        n_states: int,
        state_interval: int,
        data_dir_template: Callable,
        fix_N : bool
    ):
        super().__init__(
            base_dir, get_model, get_task_model,
            n_states, state_interval,
            data_dir_template
        )
        self.max_batch_size = max_batch_size
        self.seq_per_task = seq_per_task
        self.val_repeat = val_repeat
        self.n_val_sequences = self.val_repeat * self.max_batch_size
        self.fix_N = fix_N

        self.model_out = {
            "name": [], "seed": [], self.param_name: [], "n": [], "t": [], "idx": [],
            "train": [],
            "val": [],
            "est_name": []
        }

    def write_model_out(self, name, train, val, est_name):
        self.model_out["name"].append(name)
        self.model_out["seed"].append(self.seed)
        self.model_out[self.param_name].append(self.param_value)
        self.model_out["n"].append(self.n)
        self.model_out["t"].append(self.t)
        self.model_out["idx"].append(self.idx)
        self.model_out["train"].append(train)
        self.model_out["val"].append(val)
        self.model_out["est_name"].append(est_name)

    def chunk_and_truncate(self, dist: str, T, idx):
        if dist == "train":
            start = sum(self.batch_list[:idx])
            self.batch_size = self.batch_list[idx]
            self.x_train_trunc = self.x_train[start:start+self.batch_size, -T:]
            self.y_train_trunc = self.y_train[start:start+self.batch_size, -T:]
        elif dist == "val":
            start = idx * self.max_batch_size
            self.batch_size = self.max_batch_size
            self.x_val_trunc = self.x_val[start:start+self.batch_size, -T:]
            self.y_val_trunc = self.y_val[start:start+self.batch_size, -T:]

    def seed_step(self, seed):
        self.seed = seed

    def param_step(self, param_value):
        self.param_value = param_value

    def n_step(self, n):
        self.n = n

        # generate sequences once
        if self.n == min(self.nrange):
            self.config = u.load_config_and_task_pool(self.get_data_dir())
            set_seed(self.config.training.seed+1)
            self.task_model = self.get_task_model(self.config)
            total = self.seq_per_task * self.config.task.n_tasks
            self.batch_list = [self.max_batch_size] * (total // self.max_batch_size)
            if total % self.max_batch_size:
                self.batch_list.append(total % self.max_batch_size)
            self.n_sequences = sum(self.batch_list)
            self.x_train, self.y_train = self.task_model.get_batch(self.n_sequences, int(max(self.nrange)))
            self.x_val, self.y_val = self.task_model.get_batch(self.val_repeat * self.max_batch_size, int(max(self.nrange)), dist="val")
        else:
            cfg_path = self.get_data_dir() / "config.yaml"
            self.config = u.load_config(str(cfg_path))
        
        # ensure consistency with config
        assert (self.n == self.config.training.context_len) & (self.param_value == u.dotted_get(self.config, self.param))

        self.model = self.get_model(self.config)

        # initialize estimators
        if self.fix_N:
            ests = (
                init_all_estimators_last(self.config, self.task_model.task_pool.cuda().float(), self.task_model.stat_dist.cuda().float())
            )
        else:
            ests = (
                init_all_estimators(self.config, self.task_model.task_pool.cuda().float(), self.task_model.stat_dist.cuda().float())
            )
        self.repeat = len(self.batch_list)
        self.cached_est_logits = {
            "train": [],
            "val": [],
            "name": [],
        }
        for est in ests:
            tr_logits = []
            val_logits = []
            for i in range(self.repeat):
                self.chunk_and_truncate("train", self.config.training.context_len, i)
                logits, train_loss = est(self.x_train_trunc, self.y_train_trunc)
                tr_logits.append(logits.flatten(0, 1))
            for i in range(self.val_repeat):
                self.chunk_and_truncate("val", self.config.training.context_len, i)
                logits, val_loss = est(self.x_val_trunc, self.y_val_trunc)
                val_logits.append(logits.flatten(0, 1))
            self.cached_est_logits["train"].append(tr_logits)
            self.cached_est_logits["val"].append(val_logits)
            self.cached_est_logits["name"].append(est.name)

    def t_step(self, idx):
        self.idx = idx
        state_path = self.get_data_dir() / "state" / f"{idx}.pt"
        try:
            state = torch.load(str(state_path), map_location=self.config.model.device)
        except FileNotFoundError:
            return True
        self.t = state["iter"]
        self.model.load_state_dict(state["state"])
        self.model.eval()

        # model performance
        for est_idx, est in enumerate(self.cached_est_logits["name"]):
            rep = {"train": [], "val": []}
            for i in range(self.repeat):
                self.chunk_and_truncate("train", self.config.training.context_len, i)
                logits, _ = self.model(self.x_train_trunc, self.y_train_trunc, last_token_loss=self.config.training.last_token_loss)
                rep['train'].append(self.batch_size * kl(self.cached_est_logits['train'][est_idx][i], logits))

            for i in range(self.val_repeat):
                self.chunk_and_truncate("val", self.config.training.context_len, i)
                logits, _ = self.model(self.x_val_trunc, self.y_val_trunc, last_token_loss=self.config.training.last_token_loss)
                rep['val'].append(self.batch_size * kl(self.cached_est_logits['val'][est_idx][i], logits))

            self.write_model_out(
                self.model.name,
                sum(rep["train"]) / self.n_sequences,
                sum(rep["val"]) / self.n_val_sequences,
                est
            )
    
    def _clean_output(self):
        self.model_out = DataFrame(self.model_out)
        self.model_out.drop_duplicates(subset = ['name', 'est_name', self.param_name, 'n', 'seed', 't'], inplace = True)    # duplicate 's' entries may result from restarts (these are harmless)

    def _save_output(self):
        self._clean_output()
        self.model_out.to_csv(str(self.base_dir / "kl.csv"), index = False)

    def _save_output_min(self):
        self._clean_output()
        self.model_out.to_csv(str(self.base_dir / f"{self.get_data_dir()}_kl.csv"), index = False)