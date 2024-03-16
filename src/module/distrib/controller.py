import datetime
import numpy as np
import ray
import time
import torch
from config import cfg
from dataset import make_data_loader, split_dataset
from model import make_model, make_optimizer, make_scheduler, make_batchnorm
from metric import make_logger
from module import to_device
from .ray_server import Server
from .ray_client import Client

class Controller:
    def __init__(self, data_split, model, optimizer, scheduler, logger):
        self.data_split = data_split
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.worker = {}

    def make_worker(self, dataset):
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init()
        self.worker['server'] = Server(0, dataset, self.data_split, self.model, self.optimizer,
                                       self.scheduler, self.logger, cfg[cfg['tag']]['global'])
        local_cfg = self.make_local_cfg()
        self.worker['client'] = []
        for i in range(len(self.data_split['data'])):
            dataset_i = {k: split_dataset(dataset[k], self.data_split['data'][i][k]) for k in dataset}
            client_i = Client.remote(i, dataset_i, local_cfg)
            self.worker['client'].append(client_i)
        return

    def make_local_cfg(self):
        local_cfg = cfg[cfg['tag']]['local']
        local_cfg['profile'] = cfg['profile']
        local_cfg['data_name'] = cfg['data_name']
        local_cfg['logger_path'] = cfg['logger_path']
        return local_cfg

    def train(self):
        active_client_id, model_state_dict, optimizer_state_dict, scheduler_state_dict = self.worker['server'].train(
            self.worker['client'])
        self.worker['server'].synchronize(active_client_id, model_state_dict,
                                          optimizer_state_dict, scheduler_state_dict)
        return

    def update(self):
        self.model.load_state_dict(self.worker['server'].model.state_dict())
        self.optimizer['local'].load_state_dict(self.worker['server'].optimizer['local'].state_dict())
        self.optimizer['global'].load_state_dict(self.worker['server'].optimizer['global'].state_dict())
        self.scheduler['local'].load_state_dict(self.worker['server'].scheduler['local'].state_dict())
        self.scheduler['global'].load_state_dict(self.worker['server'].scheduler['global'].state_dict())
        self.logger.load_state_dict(self.worker['server'].logger.state_dict())
        return

    def test(self):
        if cfg['dist_mode']['eval_mode'] == 'server':
            self.worker['server'].make_batchnorm_server(None, True)
            self.worker['server'].test_server()
        elif cfg['dist_mode']['eval_mode'] == 'client':
            self.worker['server'].make_batchnorm_client(self.worker['client'], None, True)
            self.worker['server'].test_client(self.worker['client'])
        else:
            raise ValueError('Not valid test mode')
        return

    def model_state_dict(self):
        return self.model.state_dict()

    def optimizer_state_dict(self):
        return {'local': self.optimizer['local'].state_dict(), 'global': self.optimizer['global'].state_dict()}

    def scheduler_state_dict(self):
        return {'local': self.scheduler['local'].state_dict(), 'global': self.scheduler['global'].state_dict()}

    def logger_state_dict(self):
        return self.logger.state_dict()



def make_controller(data_split, model, optimizer, scheduler, logger):
    controller = Controller(data_split, model, optimizer, scheduler, logger)
    return controller
