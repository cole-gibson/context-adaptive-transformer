from pd.compute import order_params, naming
from pathlib import Path
from math import floor
import pandas as pd
from ml_collections import ConfigDict
from pd.train import train_model

class BinarySearch():
    def __init__(self, exp_config: ConfigDict, n_idx: int, seed_idx: int):
        self.exp_config = exp_config
        self.n_idx = n_idx
        self.seed_idx = seed_idx
        self.low = 0
        self.high = len(self.exp_config.paramrange)-1
        self.threshold = exp_config.training.threshold
    
    def naming_helper(self, param_idx):
        return naming(self.exp_config.seedrange[self.seed_idx], self.exp_config.nrange[self.n_idx], self.exp_config.param_name, self.exp_config.paramrange[param_idx])
    
    def criteria(self, param_idx: int):
        if not (Path(self.exp_config.base_dir) / 'data' / f'{self.naming_helper(param_idx)}_order_params.csv').exists():
            order_params(self.exp_config, param_idx, self.seed_idx, self.n_idx)
        output = pd.read_csv(Path(self.exp_config.base_dir) / 'data' / f'{self.naming_helper(param_idx)}_order_params.csv')
        
        return output['iha_tot'].max() > self.threshold

    def run(self):
        while self.low <= self.high:
            mid = self.low + (self.high-self.low)//2
            train_model(self.exp_config, mid, self.seed_idx, self.n_idx)        
            above = self.criteria(mid)
            if above:
                self.high = mid-1
            else:
                self.low = mid+1
    
def run(exp_config: ConfigDict, n_idx: int, seed_idx: int):
    search = BinarySearch(exp_config, n_idx, seed_idx)
    search.run()