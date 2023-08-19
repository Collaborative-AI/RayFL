import copy
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from config import cfg
from dataset import make_data_loader, collate, split_dataset
from model import make_optimizer
from module import to_device


class Controller:
    def __init__(self, data_split, model, optimizer, scheduler, metric, logger):
        self.data_split = data_split
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = metric
        self.logger = logger
        self.worker = {}

    def make_worker(self, dataset):
        self.worker['server'] = Server(0, dataset, self.data_split, self.model, self.optimizer,
                                       self.scheduler, self.metric, self.logger)
        self.worker['client'] = []
        for i in range(len(self.data_split['data'])):
            dataset_i = {k: split_dataset(dataset[k], self.data_split['data'][i][k]) for k in dataset}
            client_i = Client(i, dataset_i, self.optimizer['local'])
            self.worker['client'].append(client_i)
        return

    def train(self):
        self.worker['server'].train(self.worker['client'])
        self.worker['server'].synchronize(self.worker['client'])
        return

    def update(self):
        self.model.load_state_dict(self.worker['server'].model.state_dict())
        self.optimizer['local'].load_state_dict(self.worker['server'].optimizer['local'].state_dict())
        self.optimizer['global'].load_state_dict(self.worker['server'].optimizer['global'].state_dict())
        self.scheduler['local'].load_state_dict(self.worker['server'].scheduler['local'].state_dict())
        self.scheduler['global'].load_state_dict(self.worker['server'].scheduler['global'].state_dict())
        self.metric.load_state_dict(self.worker['server'].metric.state_dict())
        self.logger.load_state_dict(self.worker['server'].logger.state_dict())
        return

    def test(self):
        self.worker['server'].make_batchnorm()
        self.worker['server'].test(self.worker['client'])
        return

    def model_state_dict(self):
        return self.model.state_dict()

    def optimizer_state_dict(self):
        return {'local': self.optimizer['local'].state_dict(), 'global': self.optimizer['global'].state_dict()}

    def scheduler_state_dict(self):
        return {'local': self.scheduler['local'].state_dict(), 'global': self.scheduler['global'].state_dict()}

    def metric_state_dict(self):
        return self.metric.state_dict()

    def logger_state_dict(self):
        return self.logger.state_dict()


class Server:
    def __init__(self, id, dataset, data_split, model, optimizer, scheduler, metric, logger):
        self.id = id
        self.dataset = dataset
        self.data_loader = make_data_loader(self.dataset, 'global')
        self.data_split = data_split
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = metric
        self.logger = logger

    def synchronize(self, client):
        with torch.no_grad():
            valid_client = [client[i] for i in range(len(client)) if client[i].active]
            if len(valid_client) > 0:
                self.optimizer['global'].zero_grad()
                valid_data_size = [len(self.data_split['data'][client[i].id])
                                   for i in range(len(client)) if client[i].active]
                weight = torch.tensor(valid_data_size)
                weight = weight / weight.sum()
                for k, v in self.model.named_parameters():
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        for m in range(len(valid_client)):
                            tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                        v.grad = (v.data - tmp_v).detach()
                self.optimizer['global'].step()
            for i in range(len(client)):
                client[i].active = False
            self.scheduler['local'].step()
        return

    def train(self, client):
        start_time = time.time()
        lr = self.optimizer['local'].param_groups[0]['lr']
        num_active_clients = int(np.ceil(cfg['comm_mode']['active_ratio'] * len(client)))
        active_client_idx = torch.randperm(len(client))[:num_active_clients]
        active_client_id = [client[i].id for i in active_client_idx]
        for i in range(len(client)):
            if i in active_client_idx:
                client[i].active = True
            else:
                client[i].active = False
        for i in range(num_active_clients):
            m = active_client_idx[i]
            client[m].train(copy.deepcopy(self.model), lr, self.metric, self.logger)
            if i % int((num_active_clients * cfg['log_interval']) + 1) == 0:
                _time = (time.time() - start_time) / (i + 1)
                epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_clients - i - 1))
                exp_finished_time = epoch_finished_time + datetime.timedelta(
                    seconds=round((cfg['global']['num_epochs'] - cfg['epoch']) * _time * num_active_clients))
                exp_progress = 100. * i / num_active_clients
                info = {'info': ['Model: {}'.format(cfg['model_tag']),
                                 'Train Epoch (C): {}({:.0f}%)'.format(cfg['epoch'], exp_progress),
                                 'Learning rate: {:.6f}'.format(lr),
                                 'ID: {}({}/{})'.format(active_client_id[i], i + 1, num_active_clients),
                                 'Epoch Finished Time: {}'.format(epoch_finished_time),
                                 'Experiment Finished Time: {}'.format(exp_finished_time)]}
                self.logger.append(info, 'train')
                print(self.logger.write('train', self.metric.metric_name['train']))
        return

    def make_batchnorm(self):
        def make_batchnorm_(m, momentum, track_running_stats):
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.momentum = momentum
                m.track_running_stats = track_running_stats
                m.register_buffer('running_mean', torch.zeros(m.num_features, device=m.weight.device))
                m.register_buffer('running_var', torch.ones(m.num_features, device=m.weight.device))
                m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=m.weight.device))
            return m

        with torch.no_grad():
            model = copy.deepcopy(self.model)
            model = model.to(cfg['device'])
            model.apply(lambda m: make_batchnorm_(m, momentum=None, track_running_stats=True))
            model.train(True)
            for i, input in enumerate(self.data_loader['test']):
                input = collate(input)
                input = to_device(input, cfg['device'])
                model(input)
        self.model.load_state_dict(model.state_dict())
        return

    def test(self, client):
        with torch.no_grad():
            model = copy.deepcopy(self.model)
            model = model.to(cfg['device'])
            model.train(False)
            for i, input in enumerate(self.data_loader['test']):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model(input)
                evaluation = self.metric.evaluate('test', 'batch', input, output)
                self.logger.append(evaluation, 'test', input_size)
            evaluation = self.metric.evaluate('test', 'full')
            self.logger.append(evaluation, 'test', input_size)
            info = {
                'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100.)]}
            self.logger.append(info, 'test')
            print(self.logger.write('test', self.metric.metric_name['test']))
        return


class Client:
    def __init__(self, id, dataset, optimizer):
        self.id = id
        self.dataset = dataset
        self.data_loader = make_data_loader(self.dataset, 'local')
        self.model_state_dict = {}
        self.optimizer_state_dict = make_state_dict(optimizer)
        self.active = False

    def train(self, model, lr, metric, logger):
        model = model.to(cfg['device'])
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = make_optimizer(model.parameters(), 'local')
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)
        for epoch in range(1, cfg['comm_mode']['local_update'] + 1):
            for i, input in enumerate(self.data_loader['train']):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                evaluation = metric.evaluate('train', 'batch', input, output)
                logger.append(evaluation, 'train', n=input_size)
        self.model_state_dict = make_state_dict(model)
        self.optimizer_state_dict = make_state_dict(optimizer)
        return

    def test(self, model, metric, logger):
        with torch.no_grad():
            model = model.to(cfg['device'])
            model.train(False)
            for i, input in enumerate(self.data_loader['test']):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model(input)
                evaluation = metric.evaluate('test', 'batch', input, output)
                logger.append(evaluation, 'test', input_size)
            evaluation = metric.evaluate('test', 'full')
            logger.append(evaluation, 'test', input_size)
        return


def make_state_dict(input):
    state_dict = input.state_dict()
    state_dict_ = {}
    for k, v in state_dict.items():
        if isinstance(state_dict[k], torch.Tensor):
            state_dict_[k] = copy.deepcopy(to_device(state_dict[k], 'cpu'))
        else:
            state_dict_[k] = copy.deepcopy(state_dict[k])
    return state_dict_


def make_controller(data_split, model, optimizer, scheduler, metric, logger):
    controller = Controller(data_split, model, optimizer, scheduler, metric, logger)
    return controller
