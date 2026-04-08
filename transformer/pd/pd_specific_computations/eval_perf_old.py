import sys
if '/home/cg5763/work' not in sys.path:
    sys.path.insert(0, '/home/cg5763/work')

import torch
from base.estimators.markov_model import init_all_estimators
from base.estimators.markov_model_ar import init_all_estimators as init_all_estimators_ar
import base.utils as u
from math import floor
import time
from base.seed import set_seed

@torch.jit.script
def shuffle(x: torch.Tensor, y: torch.Tensor):
        """
        Shuffle the input and output sequences in the same way for a batch of sequences.
        Last element of the sequence is not shuffled.

        Args:
            x: Sequence of shape (B, T)
            y: Sequence of shape (B, T)
        
        Returns:
            x_s: Shuffled sequence of x
            y_s: Shuffled sequence of y
        """
        device = x.device
        B, T = x.shape
        idx = torch.argsort(torch.rand(B, T-1, device = device), dim = -1)
        x_s = x[:, :-1].gather(1, idx)
        y_s = y[:, :-1].gather(1, idx)
        x_s = torch.cat((x_s, x[:, -1:]), dim = -1)
        y_s = torch.cat((y_s, y[:, -1:]), dim = -1)
        
        return x_s, y_s

class EvalPerf():
    @torch.no_grad()
    def __init__(self, dir: str, get_model, TaskModel, max_batch_size: int, seq_per_task: int, val_repeat: int, random_seed: int, n_states: int, state_interval: int):
        self.n_states = n_states
        self.state_interval = state_interval
        self.TaskModel = TaskModel
        self.get_model = get_model
        self.max_batch_size = max_batch_size
        self.seq_per_task = seq_per_task
        self.val_repeat = val_repeat
        self.n_val_sequences = self.val_repeat*self.max_batch_size
        self.dir = dir
        self.random_seed = random_seed
        exp_config = u.load_config(f'{dir}/exp_config.yaml')
        try:
            exp_config.lmin
            self.nrange = torch.arange(exp_config.lmin, exp_config.lmax+exp_config.lstep, exp_config.lstep)
        except AttributeError:
            self.nrange = exp_config.nrange
        self.krange = exp_config.krange
        set_seed(self.random_seed)

        self.model_out = {
            'name': [],
            'k': [],
            'n': [],
            's': [],
            'idx': [],
            'train': [],
            'val': [],
            'train_s': [],
            'val_s': [],
            'train_ar': [],
            'val_ar': [],
            'train_s_ar': [],
            'val_s_ar': [],
        }

        self.est_out = {
            'name': [],
            'k': [],
            'n': [],
            'train': [],
            'val': [],
        }
    
    def write_model_out(self, name: str, train: float, val: float, train_s: float, val_s: float, train_ar: float, val_ar: float, train_s_ar: float, val_s_ar: float):
        self.model_out['name'].append(name)
        self.model_out['k'].append(self.k)
        self.model_out['n'].append(self.n)
        self.model_out['s'].append(self.s)
        self.model_out['idx'].append(self.idx)
        self.model_out['train'].append(train)
        self.model_out['val'].append(val)
        self.model_out['train_s'].append(train_s)
        self.model_out['val_s'].append(val_s)
        self.model_out['train_ar'].append(train_ar)
        self.model_out['val_ar'].append(val_ar)
        self.model_out['train_s_ar'].append(train_s_ar)
        self.model_out['val_s_ar'].append(val_s_ar)
    
    def write_est_out(self, name: str, train: float, val: float):
        self.est_out['name'].append(name)
        self.est_out['k'].append(self.k)
        self.est_out['n'].append(self.n)
        self.est_out['train'].append(train)
        self.est_out['val'].append(val)
    
    def chunk_and_truncate(self, dist: str, T, idx):
        if dist == 'train':
            start = sum(self.batch_list[:idx])
            self.batch_size = self.batch_list[idx]
            self.x_train_trunc = self.x_train[start:start+self.batch_size, -T:]
            self.y_train_trunc = self.y_train[start:start+self.batch_size, -T:]
            self.x_train_s_trunc, self.y_train_s_trunc = shuffle(self.x_train_trunc, self.y_train_trunc)
        elif dist == 'val':
            start = idx*self.max_batch_size
            self.batch_size = self.max_batch_size
            self.x_val_trunc = self.x_val[start:start+self.batch_size, -T:]
            self.y_val_trunc = self.y_val[start:start+self.batch_size, -T:]
            self.x_val_s_trunc, self.y_val_s_trunc = shuffle(self.x_val_trunc, self.y_val_trunc)

    def k_step(self, k):
        self.k = k
        base_config = u.load_config_and_task_pool(f'{self.dir}/data/{self.k}_{min(self.nrange)}')
        task_model = self.TaskModel(base_config.task)
        self.task_model = task_model
        self.batch_list = [self.max_batch_size for _ in range((self.seq_per_task*base_config.task.n_tasks)//self.max_batch_size)]
        if (self.seq_per_task*base_config.task.n_tasks)%self.max_batch_size != 0:
            self.batch_list.append((self.seq_per_task*base_config.task.n_tasks)%self.max_batch_size)
        self.n_sequences = sum(self.batch_list)
        self.x_train, self.y_train = task_model.get_batch(sum(self.batch_list), floor(max(self.nrange)))
        self.x_val, self.y_val = task_model.get_batch(self.val_repeat*self.max_batch_size, floor(max(self.nrange)), dist = 'val')
    
    def n_step(self, n):
        self.n = n
        self.config = u.load_config(f'{self.dir}/data/{self.k}_{self.n}/config.yaml')
        model = self.get_model(self.config)
        est_list = init_all_estimators(self.config, self.task_model.task_pool.cuda().float(), self.task_model.stat_dist.cuda().float()) + init_all_estimators_ar(self.config, self.task_model.task_pool.cuda().float(), self.task_model.stat_dist.cuda().float())
        self.model = model
        self.repeat = len(self.batch_list)

        for est in est_list:
            rep_train = []
            for i in range(self.repeat):
                self.chunk_and_truncate('train', self.config.training.context_len, i)
                _, train_loss = est(self.x_train_trunc, self.y_train_trunc)
                rep_train.append(self.batch_size*train_loss.item())

            rep_val = []
            for i in range(self.val_repeat):
                self.chunk_and_truncate('val', self.config.training.context_len, i)
                _, val_loss = est(self.x_val_trunc, self.y_val_trunc)
                rep_val.append(self.batch_size*val_loss.item())
            
            self.write_est_out(est.name, sum(rep_train)/self.n_sequences, sum(rep_val)/self.n_val_sequences)
    
    def s_step(self, idx):
        self.idx = idx
        try:
            self.state = torch.load(f'{self.dir}/data/{self.k}_{self.n}/state/{idx}.pt', map_location = self.config.model.device)
        except FileNotFoundError:
            return True
        self.s = self.state['iter']
        self.model.load_state_dict(self.state['state'])
        self.model.eval()

        rep_train = []
        rep_train_s = []
        rep_train_ar = []
        rep_train_s_ar = []
        for i in range(len(self.batch_list)):
            self.chunk_and_truncate('train', self.config.training.context_len, i)
            _, train_loss = self.model(self.x_train_trunc, self.y_train_trunc, last_token_loss = True)
            _, train_loss_s = self.model(self.x_train_s_trunc, self.y_train_s_trunc, last_token_loss = True)
            _, train_loss_ar = self.model(self.x_train_trunc, self.y_train_trunc, last_token_loss = False)
            _, train_loss_s_ar = self.model(self.x_train_s_trunc, self.y_train_s_trunc, last_token_loss = False)
            rep_train.append(self.batch_size*train_loss.item())
            rep_train_s.append(self.batch_size*train_loss_s.item())
            rep_train_ar.append(self.batch_size*train_loss_ar.item())
            rep_train_s_ar.append(self.batch_size*train_loss_s_ar.item())
        
        rep_val = []
        rep_val_s = []
        rep_val_ar = []
        rep_val_s_ar = []
        for i in range(self.val_repeat):
            self.chunk_and_truncate('val', self.config.training.context_len, i)
            _, val_loss = self.model(self.x_val_trunc, self.y_val_trunc, last_token_loss = True)
            _, val_loss_s = self.model(self.x_val_s_trunc, self.y_val_s_trunc, last_token_loss = True)
            _, val_loss_ar = self.model(self.x_val_trunc, self.y_val_trunc, last_token_loss = False)
            _, val_loss_s_ar = self.model(self.x_val_s_trunc, self.y_val_s_trunc, last_token_loss = False)
            rep_val.append(self.batch_size*val_loss.item())
            rep_val_s.append(self.batch_size*val_loss_s.item())
            rep_val_ar.append(self.batch_size*val_loss_ar.item())
            rep_val_s_ar.append(self.batch_size*val_loss_s_ar.item())

        self.write_model_out(self.model.name, sum(rep_train)/self.n_sequences, sum(rep_val)/self.n_val_sequences, sum(rep_train_s)/self.n_sequences, sum(rep_val_s)/self.n_val_sequences, sum(rep_train_ar)/self.n_sequences, sum(rep_val_ar)/self.n_val_sequences, sum(rep_train_s_ar)/self.n_sequences, sum(rep_val_s_ar)/self.n_val_sequences)
    
    @torch.no_grad()
    def run(self):
        for k in self.krange:
            start_time = time.time()
            print(f'k: {k}')
            self.k_step(k)
            for n in self.nrange:
                print(f'n: {n}')
                self.n_step(n)
                for idx in torch.arange(0, self.n_states, self.state_interval):
                    self.s_step(idx)
            print(f'Elapsed time: {time.time()-start_time}')
        
        torch.save(self.model_out, f'{self.dir}/model_perf.pt')
        torch.save(self.est_out, f'{self.dir}/est_perf.pt')
    
    @torch.no_grad()
    def run_min(self, krange):
        self.krange = krange
        for k in self.krange:
            start_time = time.time()
            print(f'k: {k}')
            self.k_step(k)
            for n in self.nrange:
                print(f'n: {n}')
                self.n_step(n)
                for idx in torch.arange(0, self.n_states, self.state_interval):
                    flag = self.s_step(idx)
                    if flag:
                        break
            print(f'Elapsed time: {time.time()-start_time}')
        
        torch.save(self.model_out, f'{self.dir}/{k}_model_perf.pt')
        torch.save(self.est_out, f'{self.dir}/{k}_est_perf.pt')