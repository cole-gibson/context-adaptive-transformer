# order_params.py
import torch
from base.attn_patterns import get_attn_pattern
import base.utils as u
from pd.generic_experiment import BaseRunner
from typing import Callable
from base.seed import set_seed
from pandas import DataFrame


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
def get_lt_entropy(attn_pat: torch.Tensor, x: torch.Tensor):
    lt_attn = attn_pat[:, -1, :]
    lt_entropy = - (lt_attn * torch.log2(lt_attn + 1e-10)).sum(-1)
    return lt_entropy

@torch.jit.script
def get_iha(attn_pat: torch.Tensor, x: torch.Tensor, last_token_only: bool = False):
    mask = beta_mask(x)
    lt_ind_wei = (mask[:, -1, :] * attn_pat[:, -1, :]).sum(-1)  # iha for last token
    lt_m = mask[:, -1, :].sum(-1)   # number of match targets
    lt_iha = lt_ind_wei / lt_m      # attention per match

    if last_token_only:
        # compute only for last token
        ind_wei = lt_ind_wei
        m = lt_m
        iha = lt_iha

        return ind_wei, iha, m, lt_ind_wei, lt_iha
    else:
        # compute over all sequence lengths
        N = x.shape[-1]
        ind_wei = (mask * attn_pat).sum((-2, -1))   # iha for all tokens
        m = mask.sum((-2, -1))                  # number of match targets for all tokens
        iha = ind_wei / m                       # attention per match for all tokens
        return ind_wei/N, iha, m, lt_ind_wei, lt_iha    # ind_wei is ~N for an induction head

class OrderParams(BaseRunner):
    def __init__(
        self,
        base_dir: str,
        get_model,
        get_task_model,
        batch_size: int,
        repeat: int,
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
        self.batch_size = batch_size
        self.repeat = repeat
        self.fix_N = fix_N
        # storage for results
        self.model_out = {
            "name": [], 
            "seed": [], 
            self.param_name: [], 
            "n": [], 
            "t": [],
            "pta": [], 
            "iha": [], 
            "iha_tot": [], 
            "m": [],
            "lt_iha_tot": [],
            "1_lt_entropy": [],
            "2_lt_entropy": []
        }

    def write_model_out(self, name, pta, iha, iha_tot, m, lt_iha_tot, one_lt_entropy, two_lt_entropy):
        self.model_out["name"].append(name)
        self.model_out["seed"].append(self.seed)
        self.model_out[self.param_name].append(self.param_value)
        self.model_out["n"].append(self.n)
        self.model_out["t"].append(self.t)
        self.model_out["pta"].append(pta)
        self.model_out["iha"].append(iha)
        self.model_out["iha_tot"].append(iha_tot)
        self.model_out["m"].append(m)
        self.model_out["lt_iha_tot"].append(lt_iha_tot)
        self.model_out["1_lt_entropy"].append(one_lt_entropy)
        self.model_out["2_lt_entropy"].append(two_lt_entropy)

    def chunk_and_truncate(self, T, idx):
        start = self.batch_size * idx
        end = start + self.batch_size
        self.x_train_trunc = self.x_train[start:end, -T:]
        self.y_train_trunc = self.y_train[start:end, -T:]
        if self.inject_tasks:
            self.tasks_trunc = self.tasks[start:end, -T:]

    def seed_step(self, seed):
        self.seed = seed

    def param_step(self, param_value):
        self.param_value = param_value

    def n_step(self, n):
        self.n = n

        # generate sequences once
        if n == min(self.nrange):
            self.config = u.load_config_and_task_pool(self.get_data_dir())
            self.inject_tasks = u.safe_getattr(self.config, 'model.inject_tasks')
            set_seed(self.config.training.seed+1)
            self.task_model = self.get_task_model(self.config)
            self.n_sequences = self.repeat * self.batch_size
            if self.inject_tasks:
                self.x_train, self.y_train, self.tasks = self.task_model.get_batch(
                    self.n_sequences, int(max(self.nrange)), return_tasks = True
                )
            else:
                self.x_train, self.y_train = self.task_model.get_batch(
                self.n_sequences, int(max(self.nrange))
            )
        else:
            cfg_path = self.get_data_dir() / "config.yaml"
            self.config = u.load_config(str(cfg_path))

        self.model = self.get_model(self.config)

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

        rep_pta, rep_iha, rep_iha_tot, rep_m, rep_lt_iha_tot, rep_one_lt_entropy, rep_two_lt_entropy = [], [], [], [], [], [], []
        for i in range(self.repeat):
            self.chunk_and_truncate(self.config.training.context_len, i)
            if self.inject_tasks:
                attn_pats = get_attn_pattern(self.model, (self.x_train_trunc, None, self.tasks_trunc))
            else:
                attn_pats = get_attn_pattern(self.model, (self.x_train_trunc, None))
            pta = get_pta(attn_pats[0], self.x_train_trunc)
            one_lt_entropy = get_lt_entropy(attn_pats[0], self.x_train_trunc)
            try:
                tot, iha, m, lt_iha_tot, _ = get_iha(attn_pats[2], self.x_train_trunc, last_token_only = self.fix_N)
                two_lt_entropy = get_lt_entropy(attn_pats[2], self.x_train_trunc)
            except KeyError:
                tot, iha, m, lt_iha_tot = torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)    # for models without attn pattern in second layer
                two_lt_entropy = torch.zeros(1)
            rep_pta.append(pta.sum().item())
            # rep_pta.append(pta.sum(0).tolist())
            rep_iha.append(iha.sum().item())
            rep_iha_tot.append(tot.sum().item())
            rep_m.append(m.sum().item())
            rep_lt_iha_tot.append(lt_iha_tot.sum().item())
            rep_one_lt_entropy.append(one_lt_entropy.sum().item())
            rep_two_lt_entropy.append(two_lt_entropy.sum().item())
        self.write_model_out(
            self.model.name,
            sum(rep_pta) / self.n_sequences,
            # (torch.tensor(rep_pta).sum(0) / self.n_sequences).tolist(),
            sum(rep_iha) / self.n_sequences,
            sum(rep_iha_tot) / self.n_sequences,
            sum(rep_m) / self.n_sequences,
            sum(rep_lt_iha_tot) / self.n_sequences,
            sum(rep_one_lt_entropy) / self.n_sequences,
            sum(rep_two_lt_entropy) / self.n_sequences
        )
    
    def _clean_output(self):
        self.model_out = DataFrame(self.model_out)
        self.model_out.drop_duplicates(subset = ['seed', 'n', self.param_name, 't'], inplace = True)    # duplicate 't' entries may result from restarts (these are harmless)

    def _save_output(self):
        self._clean_output()
        self.model_out.to_csv(str(self.base_dir / "order_params.csv"), index = False)

    def _save_output_min(self):
        self._clean_output()
        self.model_out.to_csv(str(self.base_dir / f"{self.get_data_dir()}_order_params.csv"), index = False)