import copy
import datetime
import numpy as np
import ray
import time
import torch
import torch.nn as nn
from config import cfg
from dataset import make_data_loader, collate, split_dataset
from model import make_model, make_optimizer
from module import to_device
from metric import make_metric, make_logger


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
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init()
        self.worker['server'] = Server(0, dataset, self.data_split, self.model, self.optimizer,
                                       self.scheduler, self.metric, self.logger)
        self.worker['client'] = []
        for i in range(len(self.data_split['data'])):
            dataset_i = {k: split_dataset(dataset[k], self.data_split['data'][i][k]) for k in dataset}
            client_i = Client.remote(i, dataset_i, cfg)
            self.worker['client'].append(client_i)
        return

    def train(self):
        active_client_id, model_state_dict = self.worker['server'].train(self.worker['client'])
        self.worker['server'].synchronize(active_client_id, model_state_dict)
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
        if cfg['test_mode'] == 'server':
            self.worker['server'].make_batchnorm_server()
            self.worker['server'].test_server()
        elif cfg['test_mode'] == 'client':
            self.worker['server'].make_batchnorm_client(self.worker['client'])
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

    def metric_state_dict(self):
        return self.metric.state_dict()

    def logger_state_dict(self):
        return self.logger.state_dict()


class Server:
    def __init__(self, id, dataset, data_split, model, optimizer, scheduler, metric, logger):
        self.id = id
        self.dataset = dataset
        self.data_loader = make_data_loader(self.dataset, cfg['global']['batch_size'], cfg['global']['shuffle'])
        self.data_split = data_split
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = metric
        self.logger = logger

    def synchronize(self, active_client_id, model_state_dict):
        with torch.no_grad():
            if len(model_state_dict) > 0:
                self.optimizer['global'].zero_grad()
                valid_data_size = [len(self.data_split['data'][active_client_id[i]])
                                   for i in range(len(active_client_id))]
                weight = torch.tensor(valid_data_size)
                weight = weight / weight.sum()
                for k, v in self.model.named_parameters():
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        for i in range(len(active_client_id)):
                            tmp_v += weight[i] * model_state_dict[i][k]
                        v.grad = (v.data - tmp_v).detach()
                self.optimizer['global'].step()
            self.optimizer['local'].step()
            self.scheduler['local'].step()
        return

    def train(self, client):
        start_time = time.time()
        lr = self.optimizer['local'].param_groups[0]['lr']
        num_active_clients = int(np.ceil(cfg['comm_mode']['active_ratio'] * len(client)))
        active_client_id = torch.randperm(len(client))[:num_active_clients]
        active_client = [client[i] for i in range(len(client)) if i in active_client_id]
        result = []
        for i in range(len(active_client)):
            result_i = active_client[i].train.remote(self.model.state_dict(), lr)
            result.append(result_i)
        result = ray.get(result)
        model_state_dict = [result[i]['model_state_dict'] for i in range(len(result))]
        logger_state_dict = [result[i]['logger_state_dict'] for i in range(len(result))]
        for i in range(len(logger_state_dict)):
            self.logger.update_state_dict(logger_state_dict[i])
        _time = (time.time() - start_time)
        exp_finished_time = datetime.timedelta(
            seconds=round((cfg['global']['num_epochs'] - cfg['epoch']) * _time * num_active_clients))
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Train Epoch (C): {}'.format(cfg['epoch']),
                         'Learning rate: {:.6f}'.format(lr),
                         'Experiment Finished Time: {}'.format(exp_finished_time)]}
        self.logger.append(info, 'train')
        print(self.logger.write('train', self.metric.metric_name['train']))
        return active_client_id, model_state_dict

    def make_batchnorm_server(self):
        flag = False
        def make_batchnorm_(m, momentum, track_running_stats):
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                flag = True
                m.momentum = momentum
                m.track_running_stats = track_running_stats
                m.register_buffer('running_mean', torch.zeros(m.num_features, device=m.weight.device))
                m.register_buffer('running_var', torch.ones(m.num_features, device=m.weight.device))
                m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=m.weight.device))
            return m

        with torch.no_grad():
            self.model.apply(lambda m: make_batchnorm_(m, momentum=None, track_running_stats=True))
            if flag:
                self.model.train(True)
                for i, input in enumerate(self.data_loader['train']):
                    input = collate(input)
                    input = to_device(input, cfg['device'])
                    self.model(input)
        return

    def test_server(self):
        with torch.no_grad():
            self.model.train(False)
            for i, input in enumerate(self.data_loader['test']):
                input = collate(input)
                input_size = input['data'].size(0)
                output = self.model(input)
                evaluation = self.metric.evaluate('test', 'batch', input, output)
                self.logger.append(evaluation, 'test', input_size)
            evaluation = self.metric.evaluate('test', 'full')
            self.logger.append(evaluation, 'test', input_size)
            info = {
                'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100.)]}
            self.logger.append(info, 'test')
            print(self.logger.write('test', self.metric.metric_name['test']))
        return

    def make_batchnorm_client(self, client):
        # result = []
        # for i in range(len(client)):
        #     result_i = active_client[i].train.remote(self.model.state_dict(), lr)
        #     result.append(result_i)
        # result = ray.get(result)

        flag = False
        def make_batchnorm_(m, momentum, track_running_stats):
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                flag = True
                m.momentum = momentum
                m.track_running_stats = track_running_stats
                m.register_buffer('running_mean', torch.zeros(m.num_features, device=m.weight.device))
                m.register_buffer('running_var', torch.ones(m.num_features, device=m.weight.device))
                m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=m.weight.device))
            return m

        with torch.no_grad():
            self.model.apply(lambda m: make_batchnorm_(m, momentum=None, track_running_stats=True))
            if flag:
                self.model.train(True)
                for i, input in enumerate(self.data_loader['train']):
                    input = collate(input)
                    input = to_device(input, cfg['device'])
                    self.model(input)
        return

    def test_client(self):
        with torch.no_grad():
            self.model.train(False)
            for i, input in enumerate(self.data_loader['test']):
                input = collate(input)
                input_size = input['data'].size(0)
                output = self.model(input)
                evaluation = self.metric.evaluate('test', 'batch', input, output)
                self.logger.append(evaluation, 'test', input_size)
            evaluation = self.metric.evaluate('test', 'full')
            self.logger.append(evaluation, 'test', input_size)
            info = {
                'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100.)]}
            self.logger.append(info, 'test')
            print(self.logger.write('test', self.metric.metric_name['test']))
        return

@ray.remote
class Client:
    def __init__(self, id, dataset, cfg):
        self.id = id
        self.dataset = dataset
        self.cfg = cfg
        self.data_loader = make_data_loader(self.dataset, cfg['local']['batch_size'], cfg['local']['shuffle'])

    def train(self, model_state_dict, lr):
        model = make_model(cfg)
        model.load_state_dict(model_state_dict)
        optimizer = make_optimizer(model.parameters(), cfg['local'])
        optimizer_state_dict = optimizer.state_dict()
        optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer.load_state_dict(optimizer_state_dict)
        metric = make_metric(cfg['data_name'], {'train': ['Loss'], 'test': ['Loss']})
        logger = make_logger()
        model.train(True)
        for epoch in range(1, self.cfg['local']['num_update'] + 1):
            for i, input in enumerate(self.data_loader['train']):
                input = collate(input)
                input_size = input['data'].size(0)
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                evaluation = metric.evaluate('train', 'batch', input, output)
                logger.append(evaluation, 'train', n=input_size)
        result = {'model_state_dict': model.state_dict(), 'logger_state_dict': logger.state_dict()}
        return result

    # def test(self, model, metric, logger):
    #     with torch.no_grad():
    #         model = model.to(cfg['device'])
    #         model.train(False)
    #         for i, input in enumerate(self.data_loader['test']):
    #             input = collate(input)
    #             input_size = input['data'].size(0)
    #             input = to_device(input, cfg['device'])
    #             output = model(input)
    #             evaluation = metric.evaluate('test', 'batch', input, output)
    #             logger.append(evaluation, 'test', input_size)
    #         evaluation = metric.evaluate('test', 'full')
    #         logger.append(evaluation, 'test', input_size)
    #     return


def make_controller(data_split, model, optimizer, scheduler, metric, logger):
    controller = Controller(data_split, model, optimizer, scheduler, metric, logger)
    return controller
