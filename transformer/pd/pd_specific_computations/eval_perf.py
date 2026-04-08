from pathlib import Path
import torch, time
from base.estimators.markov_model import init_all_estimators
from base.estimators.markov_model_ar import init_all_estimators as init_all_estimators_ar
from base.seed import set_seed
import base.utils as u
from pd.generic_experiment import BaseRunner

@torch.jit.script
def shuffle(x: torch.Tensor, y: torch.Tensor):
    device = x.device
    B, T = x.shape
    idx = torch.argsort(torch.rand(B, T-1, device=device), dim=-1)
    x_s = x[:, :-1].gather(1, idx)
    y_s = y[:, :-1].gather(1, idx)
    return torch.cat((x_s, x[:, -1:]), dim=-1), torch.cat((y_s, y[:, -1:]), dim=-1)

class EvalPerf(BaseRunner):
    def __init__(
        self,
        base_dir: str,
        get_model,
        TaskModel,
        max_batch_size: int,
        seq_per_task: int,
        val_repeat: int,
        random_seed: int,
        n_states: int,
        state_interval: int,
        data_dir_template: str = "{k}_{n}"
    ):
        super().__init__(
            base_dir, get_model, TaskModel,
            random_seed, n_states, state_interval,
            data_dir_template
        )
        self.max_batch_size = max_batch_size
        self.seq_per_task = seq_per_task
        self.val_repeat = val_repeat
        self.n_val_sequences = self.val_repeat * self.max_batch_size

        self.model_out = {
            "name": [], "k": [], "n": [], "s": [], "idx": [],
            "train": [], "val": [], "train_s": [], "val_s": [],
            "train_ar": [], "val_ar": [], "train_s_ar": [], "val_s_ar": []
        }
        self.est_out = {"name": [], "k": [], "n": [], "train": [], "val": []}

    def write_model_out(self, name, train, val, train_s, val_s, train_ar, val_ar, train_s_ar, val_s_ar):
        self.model_out["name"].append(name)
        self.model_out["k"].append(self.k)
        self.model_out["n"].append(self.n)
        self.model_out["s"].append(self.s)
        self.model_out["idx"].append(self.idx)
        self.model_out["train"].append(train)
        self.model_out["val"].append(val)
        self.model_out["train_s"].append(train_s)
        self.model_out["val_s"].append(val_s)
        self.model_out["train_ar"].append(train_ar)
        self.model_out["val_ar"].append(val_ar)
        self.model_out["train_s_ar"].append(train_s_ar)
        self.model_out["val_s_ar"].append(val_s_ar)

    def write_est_out(self, name, train, val):
        self.est_out["name"].append(name)
        self.est_out["k"].append(self.k)
        self.est_out["n"].append(self.n)
        self.est_out["train"].append(train)
        self.est_out["val"].append(val)

    def chunk_and_truncate(self, dist: str, T, idx):
        if dist == "train":
            start = sum(self.batch_list[:idx])
            self.batch_size = self.batch_list[idx]
            self.x_train_trunc = self.x_train[start:start+self.batch_size, -T:]
            self.y_train_trunc = self.y_train[start:start+self.batch_size, -T:]
            self.x_train_s_trunc, self.y_train_s_trunc = shuffle(self.x_train_trunc, self.y_train_trunc)
        else:  # "val"
            start = idx * self.max_batch_size
            self.batch_size = self.max_batch_size
            self.x_val_trunc = self.x_val[start:start+self.batch_size, -T:]
            self.y_val_trunc = self.y_val[start:start+self.batch_size, -T:]
            self.x_val_s_trunc, self.y_val_s_trunc = shuffle(self.x_val_trunc, self.y_val_trunc)

    def k_step(self, k):
        self.k = k
        base_cfg = u.load_config_and_task_pool(
            str(self.base_dir / "data" / self.data_dir_template.format(k=k, n=int(min(self.nrange))))
        )
        self.task_model = self.TaskModel(base_cfg.task)
        total = self.seq_per_task * base_cfg.task.n_tasks
        self.batch_list = [self.max_batch_size] * (total // self.max_batch_size)
        if total % self.max_batch_size:
            self.batch_list.append(total % self.max_batch_size)
        self.n_sequences = sum(self.batch_list)
        self.x_train, self.y_train = self.task_model.get_batch(self.n_sequences, int(max(self.nrange)))
        self.x_val, self.y_val = self.task_model.get_batch(self.val_repeat * self.max_batch_size, int(max(self.nrange)), dist="val")

    def n_step(self, n):
        self.n = n
        cfg_path = self.get_data_dir(self.k, self.n) / "config.yaml"
        self.config = u.load_config(str(cfg_path))
        self.model = self.get_model(self.config)
        # initialize estimators
        ests = (
            init_all_estimators(self.config, self.task_model.task_pool.cuda().float(), self.task_model.stat_dist.cuda().float())
            + init_all_estimators_ar(self.config, self.task_model.task_pool.cuda().float(), self.task_model.stat_dist.cuda().float())
        )
        self.repeat = len(self.batch_list)
        for est in ests:
            rep_tr, rep_val = [], []
            for i in range(self.repeat):
                self.chunk_and_truncate("train", self.config.training.context_len, i)
                _, train_loss = est(self.x_train_trunc, self.y_train_trunc)
                rep_tr.append(self.batch_size * train_loss.item())
            for i in range(self.val_repeat):
                self.chunk_and_truncate("val", self.config.training.context_len, i)
                _, val_loss = est(self.x_val_trunc, self.y_val_trunc)
                rep_val.append(self.batch_size * val_loss.item())
            self.write_est_out(est.name, sum(rep_tr) / self.n_sequences, sum(rep_val) / self.n_val_sequences)

    def s_step(self, idx):
        self.idx = idx
        state_path = self.get_data_dir(self.k, self.n) / "state" / f"{idx}.pt"
        try:
            state = torch.load(str(state_path), map_location=self.config.model.device)
        except FileNotFoundError:
            return True
        self.s = state["iter"]
        self.model.load_state_dict(state["state"])
        self.model.eval()

        # model performance
        rep = {key: [] for key in ["train","train_s","train_ar","train_s_ar"]}
        for i in range(len(self.batch_list)):
            self.chunk_and_truncate("train", self.config.training.context_len, i)
            out = self.model(self.x_train_trunc, self.y_train_trunc, last_token_loss=True)
            out_s = self.model(self.x_train_s_trunc, self.y_train_s_trunc, last_token_loss=True)
            out_ar = self.model(self.x_train_trunc, self.y_train_trunc, last_token_loss=False)
            out_s_ar = self.model(self.x_train_s_trunc, self.y_train_s_trunc, last_token_loss=False)
            rep["train"].append(self.batch_size * out[1].item())
            rep["train_s"].append(self.batch_size * out_s[1].item())
            rep["train_ar"].append(self.batch_size * out_ar[1].item())
            rep["train_s_ar"].append(self.batch_size * out_s_ar[1].item())

        rep_val = {key: [] for key in ["val","val_s","val_ar","val_s_ar"]}
        for i in range(self.val_repeat):
            self.chunk_and_truncate("val", self.config.training.context_len, i)
            out = self.model(self.x_val_trunc, self.y_val_trunc, last_token_loss=True)
            out_s = self.model(self.x_val_s_trunc, self.y_val_s_trunc, last_token_loss=True)
            out_ar = self.model(self.x_val_trunc, self.y_val_trunc, last_token_loss=False)
            out_s_ar = self.model(self.x_val_s_trunc, self.y_val_s_trunc, last_token_loss=False)
            rep_val["val"].append(self.batch_size * out[1].item())
            rep_val["val_s"].append(self.batch_size * out_s[1].item())
            rep_val["val_ar"].append(self.batch_size * out_ar[1].item())
            rep_val["val_s_ar"].append(self.batch_size * out_s_ar[1].item())

        self.write_model_out(
            self.model.name,
            sum(rep["train"]) / self.n_sequences,
            sum(rep_val["val"]) / self.n_val_sequences,
            sum(rep["train_s"]) / self.n_sequences,
            sum(rep_val["val_s"]) / self.n_val_sequences,
            sum(rep["train_ar"]) / self.n_sequences,
            sum(rep_val["val_ar"]) / self.n_val_sequences,
            sum(rep["train_s_ar"]) / self.n_sequences,
            sum(rep_val["val_s_ar"]) / self.n_val_sequences
        )

    def _save_output(self):
        torch.save(self.model_out, str(self.base_dir / "model_perf.pt"))
        torch.save(self.est_out, str(self.base_dir / "est_perf.pt"))

    def _save_output_min(self, last_k):
        torch.save(self.model_out, str(self.base_dir / f"{last_k}_model_perf.pt"))
        torch.save(self.est_out, str(self.base_dir / f"{last_k}_est_perf.pt"))