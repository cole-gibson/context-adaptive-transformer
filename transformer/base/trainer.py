import torch
from ml_collections import ConfigDict
from base.logger import Logger
from base.checkpoint import Checkpoint
from typing import Callable, Optional
import time
from base.utils import set_seed, load_checkpoint, print_datetime, time_display, print_eta, safe_getattr
from collections import deque
import os
from pathlib import Path
import math

def lr_lambda(step):
    warmup_steps = 500
    total_steps = 1_000_000
    if step < warmup_steps:
        return step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1. + math.cos(math.pi * progress))  # cosine decay

class Trainer():
    """
    On initialization sets training seed
    """
    def __init__(self, model, config: ConfigDict, task_model, checkpoint = None):
        self.task_model = task_model
        self.config = config
        self.model = model
        self.iter = 0 # to train from checkpoint if needed
        self.checkpoint = checkpoint
        self.inject_tasks = safe_getattr(self.config, 'model.inject_tasks')

        # set optimizer
        if self.config.training.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay = 0.0, betas = (0.9, 0.95))
        elif self.config.training.optimizer == 'adamw_lr_schedule':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay = 0.0, betas = (0.9, 0.95))
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif self.config.training.optimizer == 'adamw_decay':
            param_dict = {pn: p for pn, p in self.model.named_parameters()}
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # filter out those not requiring grad
            decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]  # >= 2d parameters are decayed
            nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.training.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            self.optimizer = torch.optim.AdamW(optim_groups, lr=config.training.learning_rate, betas = (0.9, 0.95))
        elif self.config.training.optimizer == 'adamw_decay_icl_lr':
            param_dict = {pn: p for pn, p in self.model.named_parameters()}
            icl_params = [param_dict['blocks.2.sa.beta'], param_dict['blocks.0.sa.delta']]
            
            param_dict = {pn: p for pn, p in param_dict.items() if (p.requires_grad and pn != 'blocks.0.sa.delta' and pn != 'blocks.2.sa.beta')} # filter out those not requiring grad and not icl params
            decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]  # >= 2d parameters are decayed
            nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.training.weight_decay, 'lr': self.config.training.learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': self.config.training.learning_rate},
            {'params': icl_params, 'weight_decay': 0.0, 'lr': self.config.training.icl_learning_rate}
            ]
            self.optimizer = torch.optim.AdamW(optim_groups, betas = (0.9, 0.95))
        elif self.config.training.optimizer == 'adamw_decay_full_icl_lr':
            param_dict = {pn: p for pn, p in self.model.named_parameters()}
            icl_params = [p for pn, p in param_dict.items() if 'key' in pn or 'query' in pn]
            
            param_dict = {pn: p for pn, p in param_dict.items() if (p.requires_grad and 'key' not in pn and 'query' not in pn)} # filter out those not requiring grad and not icl params
            decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]  # >= 2d parameters are decayed
            nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.training.weight_decay, 'lr': self.config.training.learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': self.config.training.learning_rate},
            {'params': icl_params, 'weight_decay': self.config.training.weight_decay, 'lr': self.config.training.icl_learning_rate}
            ]
            self.optimizer = torch.optim.AdamW(optim_groups, betas = (0.9, 0.95))
        elif self.config.training.optimizer == 'ntt':                       # for troubleshooting
            param_dict = {pn: p for pn, p in self.model.named_parameters()}
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # filter out those not requiring grad
            decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]  # > 2d parameters are decayed
            nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
            {'params': decay_params, 'weight_decay': 0.1},
            {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            self.optimizer = torch.optim.AdamW(optim_groups, lr=0.0006, betas = (0.9, 0.95))
        elif self.config.training.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=config.training.learning_rate)
        elif self.config.training.optimizer == 'sgd_momentum':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=config.training.learning_rate, momentum = config.training.momentum)
        
        # create loss deque for convergence flagging
        try:
            self.losses = deque(maxlen = self.config.training.window)
        except AttributeError:
            self.losses = deque(maxlen=1)
        
    def checkpoint_loading(self):
        """
        If checkpoint is provided or exists in directory, load it (which implicitly sets the random state).

        Otherwise, set the random seed to config.training.seed.
        """
        checkpoint_path = Path(self.config.log.project_dir) / self.config.log.model_name / 'checkpoint.pt'
        if self.checkpoint:
            self.model, self.optimizer = load_checkpoint(self.checkpoint, self.model, self.optimizer) # try to load checkpoint provided as function argument
        elif checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, weights_only=False)

            self.model, self.optimizer = load_checkpoint(checkpoint, self.model, self.optimizer)
            self.iter = checkpoint['iter']
            self.losses = checkpoint['losses_deque']

            # Recompute loss for gradient step
            x_train = checkpoint['x_train']
            y_train = checkpoint['y_train']
            if self.inject_tasks:
                tasks = checkpoint['tasks']
                _, loss = self.model(x_train, y_train, tasks, last_token_loss=self.config.training.last_token_loss)
            else:
                _, loss = self.model(x_train, y_train, last_token_loss=self.config.training.last_token_loss)
            self._sgd_step(loss)

            print(f"Loaded checkpoint from iteration {checkpoint['iter']} at {checkpoint_path}", flush=True)
        else:
            set_seed(self.config.training.seed)
        
    def create_log(self):
        self.log = Logger(self.config, self.model)     # Make sure this is right before training to capture config accurately
        try:
            log = torch.load(os.path.join(self.config.log.project_dir, self.config.log.model_name, 'log.pt'))
            self.log.load(log)  # load log to logger if it exists
        except FileNotFoundError:
            pass
        self.checkpoint = Checkpoint(self.model, self.optimizer, self.config, self.log)
    
    def _sgd_step(self, loss):
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.config.training.optimizer == 'adamw_lr_schedule':
            self.lr_scheduler.step()
        self.iter += 1
    
    def train_step(self, x_train: torch.Tensor, y_train: torch.Tensor, tasks: torch.Tensor = None):
        if self.inject_tasks:
            _, self.loss = self.model(x_train, y_train, tasks, last_token_loss = self.config.training.last_token_loss)
        else:
            tasks = None
            _, self.loss = self.model(x_train, y_train, last_token_loss = self.config.training.last_token_loss)
        
        # write loss and model state to log
        self.log.entry(self.iter, self.loss.item())
        self.log.rec_state(self.iter, self.model)
        self.losses.append(self.loss.item())

        # check if converged or at training iters; if so, finish and return True
        if self.config.training.to_convergence and len(self.losses) > 0:
            converged = sum(self.losses)/len(self.losses) < self.config.training.convergence
        else:
            converged = False
        
        max_iters = self.iter == self.config.training.iters

        if converged or max_iters:
            self._finish(x_train, y_train, tasks = tasks)
            return True
        
        # check if over training iters; if so, just return True without finishing
        over_iters = self.iter > self.config.training.iters
        if over_iters:
            return True
        
        write_checkpoint = (self.iter % self.config.training.checkpoint == 0) or (self.iter == self.config.training.iters-1)
        
        # write checkpoint if needed
        if write_checkpoint:
            self._finish(x_train, y_train, tasks = tasks)

        # do one step of gradient descent
        self._sgd_step(self.loss)
    
    def _finish(self, x_train, y_train, tasks = None):
        self.log.bundle_and_save(self.iter, self.task_model)
        self.checkpoint.save_checkpoint(self.iter, self.loss, x_train, y_train, self.losses, tasks = tasks)

    def begin(self):
        self.checkpoint_loading()
        self.create_log()
        start_time = time.time()
        converged = False
        print(f'Starting training {print_datetime()} on task model {self.task_model.name}', flush = True)
        while self.iter < self.config.training.iters and not converged:
            if self.iter == 1000:
                iter_time = time.time() - start_time
                estimated_runtime = iter_time * self.config.training.iters / 1000
                print(f'1k iters in {time_display(time.time() - start_time)} -> Runtime of {time_display(estimated_runtime)}', flush = True)
                print(f'ETA: {print_eta(estimated_runtime)}', flush = True)
            if self.inject_tasks:
                x_train, y_train, tasks = self.task_model.get_batch(self.config.training.batch_size, self.config.training.context_len, return_tasks = True)
                converged = self.train_step(x_train, y_train, tasks)
            else:
                x_train, y_train = self.task_model.get_batch(self.config.training.batch_size, self.config.training.context_len, return_tasks = False)
                converged = self.train_step(x_train, y_train)

        end_time = time.time()
        print(f'Finished training: {print_datetime()}', flush = True)
        print(f'Training time: {time_display(end_time-start_time)}', flush = True)