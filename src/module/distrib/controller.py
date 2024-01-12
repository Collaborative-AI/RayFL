import datetime
import numpy as np
import ray
import time
import torch
from config import cfg
from dataset import make_data_loader, split_dataset
from model import make_model, make_optimizer, make_batchnorm
from metric import make_metric, make_logger
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
        self.parallel_step_size = 10

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
            self.worker['server'].make_batchnorm_server(None, True)
            self.worker['server'].test_server()
        elif cfg['test_mode'] == 'client':
            self.worker['server'].make_batchnorm_client(self.worker['client'], self.parallel_step_size, None, True)
            self.worker['server'].test_client(self.worker['client'], self.parallel_step_size)
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

    def make_batchnorm_server(self, momentum, track_running_stats):
        with torch.no_grad():
            flag = make_batchnorm(self.model, momentum, track_running_stats=track_running_stats)
            if flag and track_running_stats:
                self.model.train(True)
                for i, input in enumerate(self.data_loader['train']):
                    input = collate(input)
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
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Test Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100.)]}
            self.logger.append(info, 'test')
            print(self.logger.write('test', self.metric.metric_name['test']))
        return

    def make_batchnorm_client(self, client, parallel_step_size, momentum, track_running_stats):
        result = []
        for i in range(0, len(client), parallel_step_size):
            result_i = []
            upper_bound = min(i + parallel_step_size, len(client))
            for j in range(i, upper_bound):
                result_i_j = client[i + j].make_batchnorm.remote(self.model.state_dict(), momentum, track_running_stats)
                result_i.append(result_i_j)
            result_i = ray.get(result_i)
            result.extend(result_i)
        model_state_dict = [result[i]['model_state_dict'] for i in range(len(result)) if result[i] is not None]

        with torch.no_grad():
            if len(model_state_dict) > 0:
                data_size = [len(self.data_split['data'][i]) for i in range(len(client))]
                model_state_dict_ = self.model.state_dict()
                for k, v in model_state_dict_.items():
                    parameter_type = k.split('.')[-1]
                    if 'running_mean' in parameter_type:
                        global_running_mean = v.data.new_zeros(v.size())
                        for i in range(len(model_state_dict)):
                            global_running_mean += data_size[i] * model_state_dict[i][k]
                        global_running_mean = global_running_mean / sum(data_size)
                        v.data.copy_(global_running_mean)
                    if 'running_var' in parameter_type:
                        running_mean_k = k.replace('running_var', 'running_mean')
                        global_running_var = v.data.new_zeros(v.size())
                        for i in range(len(model_state_dict)):
                            global_running_var += model_state_dict[i][k] * (data_size[i] - 1) + data_size[i] * (
                                    model_state_dict[i][running_mean_k] - global_running_mean) ** 2
                        global_running_var = global_running_var / (sum(data_size) - len(data_size))
                        v.data.copy_(global_running_var)
                self.model.load_state_dict(model_state_dict_)
        return

    def test_client(self, client, parallel_step_size):
        result = []
        for i in range(0, len(client), parallel_step_size):
            result_i = []
            upper_bound = min(i + parallel_step_size, len(client))
            for j in range(i, upper_bound):
                result_i_j = client[i + j].test.remote(self.model.state_dict())
                result_i.append(result_i_j)
            result_i = ray.get(result_i)
            result.extend(result_i)
        logger_state_dict = [result[i]['logger_state_dict'] for i in range(len(result))]
        for i in range(len(logger_state_dict)):
            self.logger.update_state_dict(logger_state_dict[i])
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(cfg['epoch'], 100.)]}
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
        model = make_model(self.cfg).to(self.cfg['device'])
        model.load_state_dict(model_state_dict)
        optimizer = make_optimizer(model.parameters(), self.cfg['local'])
        optimizer_state_dict = optimizer.state_dict()
        optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer.load_state_dict(optimizer_state_dict)
        metric = make_metric(self.cfg['data_name'], {'train': ['Loss'], 'test': ['Loss']})
        logger = make_logger()
        model.train(True)
        for epoch in range(1, self.cfg['local']['num_update'] + 1):
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
        model = model.to('cpu')
        result = {'model_state_dict': model.state_dict(), 'logger_state_dict': logger.state_dict()}
        return result

    def make_batchnorm(self, model_state_dict, momentum, track_running_stats):
        with torch.no_grad():
            model = make_model(self.cfg)
            flag = make_batchnorm(model, momentum, track_running_stats)
            if flag and track_running_stats:
                model = model.to(self.cfg['device'])
                model.load_state_dict(model_state_dict)
                model.train(True)
                for i, input in enumerate(self.data_loader['train']):
                    input = collate(input)
                    input = to_device(input, cfg['device'])
                    model(input)
                model = model.to('cpu')
                result = {'model_state_dict': model.state_dict()}
            else:
                result = None
        return result

    def test(self, model_state_dict):
        with torch.no_grad():
            model = make_model(self.cfg).to(self.cfg['device'])
            model.load_state_dict(model_state_dict)
            metric = make_metric(self.cfg['data_name'], {'train': ['Loss'], 'test': ['Loss']})
            logger = make_logger()
            model.train(False)
            for i, input in enumerate(self.data_loader['test']):
                input = collate(input)
                input = to_device(input, cfg['device'])
                input_size = input['data'].size(0)
                output = model(input)
                evaluation = metric.evaluate('test', 'batch', input, output)
                logger.append(evaluation, 'test', input_size)
            evaluation = metric.evaluate('test', 'full')
            logger.append(evaluation, 'test', input_size)
        result = {'logger_state_dict': logger.state_dict()}
        return result


def make_controller(data_split, model, optimizer, scheduler, metric, logger):
    controller = Controller(data_split, model, optimizer, scheduler, metric, logger)
    return controller
