import torch
from base.estimators.markov_model_ar import init_all_estimators
from base.estimators.markov_model_fix_N import init_all_estimators_last as init_all_estimators_fix_N
import base.utils as u
from pd.generic_experiment import BaseRunner
from typing import Callable
from base.seed import set_seed
from pandas import DataFrame

class EvalPerf(BaseRunner):
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
        fix_N: bool
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
            "name": [], 
            "seed": [], 
            self.param_name: [], 
            "n": [], 
            "t": [], 
            "dist": [], 
            "loss": []
        }
        self.est_out = {
            "name": [], 
            "seed": [], 
            self.param_name: [], 
            "n": [], 
            "dist": [], 
            "loss": []
        }

    def write_model_out(self, name, dist, loss):
        self.model_out["name"].append(name)
        self.model_out["seed"].append(self.seed)
        self.model_out[self.param_name].append(self.param_value)
        self.model_out["n"].append(self.n)
        self.model_out["t"].append(self.t)
        self.model_out["dist"].append(dist)
        self.model_out["loss"].append(loss)

    def write_est_out(self, name, dist, loss):
        self.est_out["name"].append(name)
        self.est_out["seed"].append(self.seed)
        self.est_out[self.param_name].append(self.param_value)
        self.est_out["n"].append(self.n)
        self.est_out["dist"].append(dist)
        self.est_out["loss"].append(loss)

    def chunk_and_truncate(self, dist: str, T, idx):
        if dist == "train":
            start = sum(self.batch_list[:idx])
            self.batch_size = self.batch_list[idx]
            self.x_train_trunc = self.x_train[start:start+self.batch_size, -T:]
            self.y_train_trunc = self.y_train[start:start+self.batch_size, -T:]
            if self.inject_tasks:
                self.tasks_trunc = self.tasks[start:start+self.batch_size, -T:]
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
            self.inject_tasks = u.safe_getattr(self.config, 'model.inject_tasks')
            set_seed(self.config.training.seed+1)
            self.task_model = self.get_task_model(self.config)
            total = self.seq_per_task * self.config.task.n_tasks
            self.batch_list = [self.max_batch_size] * (total // self.max_batch_size)
            if total % self.max_batch_size:
                self.batch_list.append(total % self.max_batch_size)
            self.n_sequences = sum(self.batch_list)
            if self.inject_tasks:
                self.x_train, self.y_train, self.tasks = self.task_model.get_batch(self.n_sequences, int(max(self.nrange)), return_tasks = True)
            else:
                self.x_train, self.y_train, self.tasks = self.task_model.get_batch(self.n_sequences, int(max(self.nrange)), return_tasks = True)
            self.x_val, self.y_val = self.task_model.get_batch(self.val_repeat * self.max_batch_size, int(max(self.nrange)), dist="val")
        else:
            cfg_path = self.get_data_dir() / "config.yaml"
            self.config = u.load_config(str(cfg_path))
        
        # ensure consistency with config
        assert (self.n == self.config.training.context_len) & (self.param_value == u.dotted_get(self.config, self.param))

        self.model = self.get_model(self.config)

        # initialize estimators
        if self.fix_N:
            ests = init_all_estimators_fix_N(self.config, self.task_model.task_pool.cuda().float(), self.task_model.stat_dist.cuda().float())
            print(ests)
        else:
            ests = init_all_estimators(self.config, self.task_model.task_pool.cuda().float(), self.task_model.stat_dist.cuda().float())
        self.repeat = len(self.batch_list)
        for est in ests:
            rep_tr, rep_val = [], []

            dist = 'train'
            for i in range(self.repeat):
                self.chunk_and_truncate(dist, self.config.training.context_len, i)
                _, train_loss = est(self.x_train_trunc, self.y_train_trunc)
                rep_tr.append(self.batch_size * train_loss.item())
            self.write_est_out(est.name, dist, sum(rep_tr) / self.n_sequences)

            dist = 'val'
            for i in range(self.val_repeat):
                self.chunk_and_truncate(dist, self.config.training.context_len, i)
                _, val_loss = est(self.x_val_trunc, self.y_val_trunc)
                rep_val.append(self.batch_size * val_loss.item())
            self.write_est_out(est.name, dist, sum(rep_val) / self.n_val_sequences)
            
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
        rep = []
        dist = "train"
        for i in range(len(self.batch_list)):
            self.chunk_and_truncate(dist, self.config.training.context_len, i)
            if self.inject_tasks:
                out = self.model(self.x_train_trunc, self.y_train_trunc, self.tasks_trunc, last_token_loss = self.config.training.last_token_loss)
            else:
                out = self.model(self.x_train_trunc, self.y_train_trunc, last_token_loss = self.config.training.last_token_loss)
            rep.append(self.batch_size * out[1].item())
        
        self.write_model_out(
            self.model.name,
            dist,
            sum(rep) / self.n_sequences,
        )

        rep = []
        dist = "val"
        for i in range(self.val_repeat):
            self.chunk_and_truncate(dist, self.config.training.context_len, i)
            if self.inject_tasks:
                out = self.model(self.x_val_trunc, self.y_val_trunc, last_token_loss = self.config.training.last_token_loss)
            else:
                out = self.model(self.x_val_trunc, self.y_val_trunc, last_token_loss = self.config.training.last_token_loss)
            rep.append(self.batch_size * out[1].item())

        self.write_model_out(
            self.model.name,
            dist,
            sum(rep) / self.n_val_sequences
        )
    
    def _clean_output(self):
        self.model_out = DataFrame(self.model_out)
        self.model_out.drop_duplicates(subset = ['seed', 'n', self.param_name, "dist", 't'], inplace = True)    # duplicate 't' entries may result from restarts (these are harmless)

        self.est_out = DataFrame(self.est_out)

    def _save_output(self):
        self._clean_output()
        self.model_out.to_csv(str(self.base_dir / "model_perf.csv"), index = False)
        self.est_out.to_csv(str(self.base_dir / "est_perf.csv"), index = False)

    def _save_output_min(self):
        self._clean_output()
        self.model_out.to_csv(str(self.base_dir / f"{self.get_data_dir()}_model_perf.csv"), index = False)
        self.est_out.to_csv(str(self.base_dir / f"{self.get_data_dir()}_est_perf.csv"), index = False)