# order_params.py
from pathlib import Path
import torch, time
from base.attn_patterns import get_attn_pattern
from base.seed import set_seed
import base.utils as u
from pd.generic_experiment import BaseRunner

@torch.jit.script
def beta_mask(x: torch.Tensor) -> torch.Tensor:
    device = x.device
    B, T = x.shape
    x_exp = x.unsqueeze(-1).expand(B, T, T).transpose(-1, -2)
    last_token = x.unsqueeze(-1).expand(B, T, T)
    mask = (x_exp == last_token).int()
    mask = torch.roll(mask, 1, -1)
    mask[..., 0] = 0
    mask = mask.masked_fill(
        torch.tril(torch.ones_like(mask, device=device)) == 0, 0
    ).int()
    return mask

@torch.jit.script
def get_pta(attn_pat: torch.Tensor, x: torch.Tensor):
    prev_wei = attn_pat.diagonal(offset=-1, dim1=-2, dim2=-1)
    return prev_wei.mean(-1)

@torch.jit.script
def get_iha(attn_pat: torch.Tensor, x: torch.Tensor):
    mask = beta_mask(x)
    ind_wei = (mask[:, -1, :] * attn_pat[:, -1, :]).sum(-1)
    m = mask[:, -1, :].sum(-1)
    iha = ind_wei / m
    return ind_wei, iha, m

class OrderParams(BaseRunner):
    def __init__(
        self,
        base_dir: str,
        get_model,
        TaskModel,
        batch_size: int,
        repeat: int,
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
        self.batch_size = batch_size
        self.repeat = repeat
        # storage for results
        self.model_out = {
            "name": [], "k": [], "n": [], "s": [], "idx": [],
            "pta": [], "iha": [], "iha_tot": [], "m": []
        }

    def write_model_out(self, name, pta, iha, iha_tot, m):
        self.model_out["name"].append(name)
        self.model_out["k"].append(self.k)
        self.model_out["n"].append(self.n)
        self.model_out["s"].append(self.s)
        self.model_out["idx"].append(self.idx)
        self.model_out["pta"].append(pta)
        self.model_out["iha"].append(iha)
        self.model_out["iha_tot"].append(iha_tot)
        self.model_out["m"].append(m)

    def chunk_and_truncate(self, T, idx):
        start = self.batch_size * idx
        end = start + self.batch_size
        self.x_train_trunc = self.x_train[start:end, -T:]
        self.y_train_trunc = self.y_train[start:end, -T:]

    def k_step(self, k):
        self.k = k
        base_cfg = u.load_config_and_task_pool(
            str(self.base_dir / "data" / self.data_dir_template.format(k=k, n=int(min(self.nrange))))
        )
        self.task_model = self.TaskModel(base_cfg.task)
        self.n_sequences = self.repeat * self.batch_size
        self.x_train, self.y_train = self.task_model.get_batch(
            self.n_sequences, int(max(self.nrange))
        )

    def n_step(self, n):
        self.n = n
        cfg_path = self.get_data_dir(self.k, self.n) / "config.yaml"
        self.config = u.load_config(str(cfg_path))
        self.model = self.get_model(self.config)

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

        rep_pta, rep_iha, rep_iha_tot, rep_m = [], [], [], []
        for i in range(self.repeat):
            self.chunk_and_truncate(self.config.training.context_len, i)
            attn_pats = get_attn_pattern(self.model, (self.x_train_trunc, None))
            pta = get_pta(attn_pats[0], self.x_train_trunc)
            try:
                tot, iha, m = get_iha(attn_pats[2], self.x_train_trunc)
            except KeyError:
                tot, iha, m = torch.zeros(1), torch.zeros(1), torch.zeros(1)
            rep_pta.append(pta.sum().item())
            rep_iha.append(iha.sum().item())
            rep_iha_tot.append(tot.sum().item())
            rep_m.append(m.sum().item())

        self.write_model_out(
            self.model.name,
            sum(rep_pta) / self.n_sequences,
            sum(rep_iha) / self.n_sequences,
            sum(rep_iha_tot) / self.n_sequences,
            sum(rep_m) / self.n_sequences
        )

    def _save_output(self):
        torch.save(self.model_out, str(self.base_dir / "order_params.pt"))

    def _save_output_min(self, last_k):
        torch.save(self.model_out, str(self.base_dir / f"{last_k}_order_params.pt"))