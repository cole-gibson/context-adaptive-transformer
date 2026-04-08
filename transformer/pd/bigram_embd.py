import torch, math
import torch.distributions as td
import base.utils as u
from pd.generic_experiment import BaseRunner
from typing import Callable
from base.tasks.markov_model import compute_stat_dist
from pandas import DataFrame
from ml_collections import ConfigDict
import pandas as pd
from sklearn.decomposition import PCA

class Utility():
    def __init__(self, model, config: ConfigDict, n_samples, apply_pca = False, n_pca_components = 0.9):
        self.model = model
        self.config = config
        self.MLP = self.model.blocks[1]
        self.attn = self.model.blocks[0]
        self.token_embedding_table = self.model.token_embedding_table
        self.n_samples = n_samples
        self.apply_pca = apply_pca
        self.n_pca_components = n_pca_components

        if self.apply_pca:
            self.pca = PCA(n_components=self.n_pca_components)
    
    def get_embds(self):
        "Computes from length 2 sequences of every possible transition"
        C = self.config.task.vocab_size
        D = self.config.model.n_embd
        device = self.config.model.device
        pairs = torch.cartesian_prod(torch.arange(C, device=device), torch.arange(C, device=device))  # (C^2, 2)
        x = self.token_embedding_table(torch.arange(C, device=device)) # C x D
        seqs = x[pairs]
        embds = self.MLP.ffwd(self.MLP.ln(self.attn(seqs)[:, -1:, :]))
        
        embds = embds.squeeze(1)   # (C^2, D) embedding of each bigram transition
        return embds
    
    def I(self, E, task_pool, full_cov):
        """
        Estimates mutual information between the tasks for embedding E.
        
        Args:
            E (torch.Tensor): Embedding matrix of shape (T, D)
            task_pool (torch.Tensor): Task pool of shape (K, T) corresponding to the stationary bigram probabilities
            full_cov (torch.Tensor): 
            n_samples (int): Number of samples for Monte Carlo estimation of GMM entropy

        Returns:
            float: Estimated mutual information in bits
        """
        E = E.T
        D, _ = E.shape
        N = self.config.training.context_len // 2   # compute midway through sequence 
        assert N <= self.config.training.context_len
        K = self.config.task.n_tasks

        covs = E @ full_cov @ E.T/N   # (D, T) @ (K, T, T) @ (T, D) -> (K, D, D) covariance of the conditional Gaussians across tasks
        # assert torch.isclose(covs, covs.mT).all()
        covs = (covs + covs.mT)/2   # ensure symmetry for numerical stability

        _, info = torch.linalg.cholesky_ex(covs)
        if (info != 0).any():
            return 0
        
        cs = task_pool @ E.T    # (K, T) @ (T, D) -> (K, D) task vectors in embedding space
        log_det_cov = covs.logdet()
        mix = td.Categorical(torch.ones(K, device=task_pool.device))
        comp = td.MultivariateNormal(cs, covs)
        gmm = td.MixtureSameFamily(mix, comp)

        # Monte Carlo estimate of GMM entropy
        samples = gmm.sample((self.n_samples, ))
        log_prob = gmm.log_prob(samples)
        mix_entropy = (-log_prob).mean()

        # Analytical entropy of the conditional Gaussians averaged across tasks
        cond_entropy = D*(1 + math.log(2*math.pi))/2 + log_det_cov.sum()/(2*K)
        out = mix_entropy - cond_entropy

        return out.item()/math.log(2), D
    
    def compute_mut_info(self, task_pool, full_cov):
        embds = self.get_embds()
        if self.apply_pca:
            embds_np = embds.detach().cpu().numpy()
            embds_proj = self.pca.fit_transform(embds_np)
            embds = torch.from_numpy(embds_proj).to(embds.device)
        
        mut_info, D = self.I(embds, task_pool, full_cov)

        return mut_info, D

class BigramEmbd(BaseRunner):
    def __init__(
        self,
        base_dir: str,
        get_model,
        get_task_model,
        repeat: int,
        n_mc_samples: int,
        apply_pca: bool,    # whether to apply PCA to the embeddings before computing mutual information
        n_pca_components: float,
        n_states: int,
        state_interval: int,
        data_dir_template: Callable
    ):
        super().__init__(
            base_dir, get_model, get_task_model,
            n_states, state_interval,
            data_dir_template
        )

        self.model_out = {
            'name': [],
            self.param_name: [],
            'n': [],
            'seed': [],
            't': [],
            'idx': [],
            'mutual_information': [],
            'd_embd': []
        }
    
        self.repeat = repeat
        self.n_mc_samples = n_mc_samples # sets n_samples for
        self.apply_pca = apply_pca
        self.n_pca_components = n_pca_components
        u.set_seed(self.seed+1)

    def write_model_out(self, name: str, mutual_information: float, D: int):
        self.model_out['name'].append(name)
        self.model_out[self.param_name].append(self.param_value)
        self.model_out['n'].append(self.n)
        self.model_out['seed'].append(self.seed)
        self.model_out['t'].append(self.t)
        self.model_out['idx'].append(self.idx)
        self.model_out['mutual_information'].append(mutual_information)
        self.model_out['d_embd'].append(D)

    def seed_step(self, seed):
        self.seed = seed

    def param_step(self, param_value):
        self.param_value = param_value

    def n_step(self, n):
        self.n = n

        self.config = u.load_config_and_task_pool(self.get_data_dir())

        t_matrix = self.config.task.task_pool.to('cuda').float()
        stat_dist = compute_stat_dist(t_matrix, 'cuda')
        self.task_pool = (stat_dist.unsqueeze(-1) * t_matrix).flatten(-2, -1)    # (K, T) stationary bigram probabilities
        self.full_cov = torch.diag_embed(self.task_pool) - torch.einsum('bi,bj->bij', self.task_pool, self.task_pool)   # (K, T, T) covariance of the bigram distribution across tasks
        
        # ensure consistency with config
        assert (self.n == self.config.training.context_len) & (self.param_value == u.dotted_get(self.config, self.param))

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
        # compute mutual information
        rep_mutual_information = []
        for _ in range(self.repeat):
            utility = Utility(self.model, self.config, self.n_mc_samples, self.apply_pca, self.n_pca_components)
            mut_info, D = utility.compute_mut_info(self.task_pool, self.full_cov)
            rep_mutual_information.append(mut_info)
        
        self.write_model_out(self.model.name, sum(rep_mutual_information)/self.repeat, D)
    
    def _clean_output(self):
        self.model_out = DataFrame(self.model_out)
        self.model_out.drop_duplicates(subset = ['seed', 'n', self.param_name, 't'], inplace = True)    # duplicate 't' entries may result from restarts (these are harmless)

    def _save_output(self):
        self._clean_output()
        self.model_out.to_csv(str(self.base_dir / "bigram_embd.csv"), index = False)

    def _save_output_min(self):
        self._clean_output()
        self.model_out.to_csv(str(self.base_dir / f"{self.get_data_dir()}_bigram_embd.csv"), index = False)