import torch
from base.logger import Logger
from base.utils import get_random_state, safe_save
from collections import deque

class Checkpoint:
    def __init__(self, model, optimizer, config, log: Logger):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.log = log

    def save_checkpoint(self, iter: int, loss, x_train, y_train, losses_deque: deque, tasks = None):
        safe_save({'iter': iter, 'losses_deque': losses_deque, 'model_state_dict': self.model.state_dict(), 'loss': loss, 'x_train' : x_train, 'y_train' : y_train, 'tasks' : tasks, 'optimizer_state_dict': self.optimizer.state_dict(), 'random_state': get_random_state()}, f'{self.log.model_dir}/checkpoint.pt')