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
        active_client_id, model_state_dict, lr = self.worker['server'].train(self.worker['client'])
        self.worker['server'].synchronize(active_client_id, model_state_dict, lr)
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


class Server:
    def __init__(self, id, dataset, data_split, model, optimizer, scheduler, logger, cfg):
        self.id = id
        self.dataset = dataset
        self.data_split = data_split
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.cfg = cfg
        self.data_loader = make_data_loader(self.dataset, self.cfg['optimizer']['batch_size'], shuffle=False)

    def synchronize(self, active_client_id, model_state_dict, lr):
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
                self.scheduler['global'].step()
                self.optimizer['local'].param_groups[0]['lr'] = lr
        return

    def train(self, client):
        start_time = time.time()
        num_active_clients = int(np.ceil(cfg['dist_mode']['active_ratio'] * len(client)))
        active_client_id = torch.randperm(len(client))[:num_active_clients]
        active_client = [client[i] for i in range(len(client)) if i in active_client_id]
        lr = self.optimizer['local'].param_groups[0]['lr']
        result = []
        for i in range(len(active_client)):
            result_i = active_client[i].train.remote(self.model.state_dict(), lr)
            result.append(result_i)
        result = ray.get(result)
        model_state_dict = [result[i]['model_state_dict'] for i in range(len(result))]
        logger_state_dict = [result[i]['logger_state_dict'] for i in range(len(result))]
        lr = result[0]['lr']
        step = result[0]['step']
        for i in range(len(logger_state_dict)):
            self.logger.update_state_dict(logger_state_dict[i])
        step_time = (time.time() - start_time)
        exp_finished_time = datetime.timedelta(
            seconds=round((cfg['num_steps'] - (cfg['step'] + 1)) * step_time))
        info = {'info': ['Model: {}'.format(cfg['tag']),
                         'Train Epoch (C): {}'.format(cfg['step'] + 1),
                         'Learning rate: {:.6f}'.format(lr),
                         'Experiment Finished Time: {}'.format(exp_finished_time)]}
        self.logger.append(info, 'train')
        print(self.logger.write('train'))
        cfg['step'] += step
        return active_client_id, model_state_dict, lr

    def make_batchnorm_server(self, momentum, track_running_stats):
        with torch.no_grad():
            flag = make_batchnorm(self.model, momentum, track_running_stats=track_running_stats)
            if flag and track_running_stats:
                self.model.train(True)
                for i, input in enumerate(self.data_loader['train']):
                    self.model(input)
        return

    def test_server(self):
        with torch.no_grad():
            self.model.train(False)
            for i, input in enumerate(self.data_loader['test']):
                input_size = input['data'].size(0)
                output = self.model(input)
                evaluation = self.logger.evaluate('test', 'batch', input, output)
                self.logger.append(evaluation, 'test', input_size)
            evaluation = self.logger.evaluate('test', 'full')
            self.logger.append(evaluation, 'test', input_size)
            info = {'info': ['Model: {}'.format(cfg['tag']),
                             'Test Epoch (S): {}'.format(cfg['step'])]}
            self.logger.append(info, 'test')
            print(self.logger.write('test'))
        return

    def make_batchnorm_client(self, client, momentum, track_running_stats):
        num_active_clients = int(np.ceil(cfg['dist_mode']['active_ratio'] * len(client)))
        result = []
        for i in range(0, len(client), num_active_clients):
            result_i = []
            upper_bound = min(i + num_active_clients, len(client))
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

    def test_client(self, client):
        num_active_clients = int(np.ceil(cfg['dist_mode']['active_ratio'] * len(client)))
        result = []
        for i in range(0, len(client), num_active_clients):
            result_i = []
            upper_bound = min(i + num_active_clients, len(client))
            for j in range(i, upper_bound):
                result_i_j = client[i + j].test.remote(self.model.state_dict())
                result_i.append(result_i_j)
            result_i = ray.get(result_i)
            result.extend(result_i)
        logger_state_dict = [result[i]['logger_state_dict'] for i in range(len(result))]
        for i in range(len(logger_state_dict)):
            self.logger.update_state_dict(logger_state_dict[i])
        info = {'info': ['Model: {}'.format(cfg['tag']),
                         'Test Epoch (C): {}'.format(cfg['step'])]}
        self.logger.append(info, 'test')
        print(self.logger.write('test'))
        return


@ray.remote
class Client:
    def __init__(self, id, dataset, cfg):
        self.id = id
        self.dataset = dataset
        self.cfg = cfg
        self.data_loader = make_data_loader(self.dataset, self.cfg['optimizer']['batch_size'],
                                            self.cfg['optimizer']['num_local_steps'])

    def train(self, model_state_dict, lr, step):
        self.cfg['optimizer']['num_step'] = step
        model = make_model(self.cfg['model']).to(self.cfg['device'])
        model.load_state_dict(model_state_dict)
        optimizer = make_optimizer(model.parameters(), self.cfg['optimizer'])
        optimizer_state_dict = optimizer.state_dict()
        optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler = make_scheduler(optimizer['local'], self.cfg['optimizer'])
        logger = make_logger(self.cfg['logger_path'], data_name=self.cfg['data_name'])
        model.train(True)
        with logger.profiler:
            for i, input in enumerate(self.data_loader['train']):
                if i % cfg['step_period'] == 0 and cfg['profile']:
                    logger.profiler.step()
                input_size = input['data'].size(0)
                input = to_device(input, self.cfg['device'])
                output = model(input)
                loss = 1 / cfg['step_period'] * output['loss']
                loss.backward()
                if (i + 1) % cfg['step_period'] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1
                evaluation = logger.evaluate('train', 'batch', input, output)
                logger.append(evaluation, 'train', n=input_size)
        model = model.to('cpu')
        lr = optimizer_state_dict['param_groups'][0]['lr']
        result = {'model_state_dict': model.state_dict(), 'logger_state_dict': logger.state_dict(), 'lr': lr,
                  'step': step}
        return result

    def make_batchnorm(self, model_state_dict, momentum, track_running_stats):
        with torch.no_grad():
            model = make_model(self.cfg['model']).to(self.cfg['device'])
            flag = make_batchnorm(model, momentum, track_running_stats)
            if flag and track_running_stats:
                model = model.to(self.cfg['device'])
                model.load_state_dict(model_state_dict)
                model.train(True)
                for i, input in enumerate(self.data_loader['train']):
                    input = to_device(input, self.cfg['device'])
                    model(input)
                model = model.to('cpu')
                result = {'model_state_dict': model.state_dict()}
            else:
                result = None
        return result

    def test(self, model_state_dict):
        with torch.no_grad():
            model = make_model(self.cfg['model']).to(self.cfg['device'])
            model.load_state_dict(model_state_dict)
            logger = make_logger(self.cfg['logger_path'], data_name=self.cfg['data_name'])
            model.train(False)
            for i, input in enumerate(self.data_loader['test']):
                input = to_device(input, self.cfg['device'])
                input_size = input['data'].size(0)
                output = model(input)
                evaluation = logger.evaluate('test', 'batch', input, output)
                logger.append(evaluation, 'test', input_size)
            evaluation = logger.evaluate('test', 'full')
            logger.append(evaluation, 'test', input_size)
        result = {'logger_state_dict': logger.state_dict()}
        return result


def make_controller(data_split, model, optimizer, scheduler, logger):
    controller = Controller(data_split, model, optimizer, scheduler, logger)
    return controller
