
import ray

import torch
from config import cfg
from dataset import make_data_loader, split_dataset
from model import make_model, make_optimizer, make_scheduler, make_batchnorm
from metric import make_logger
from module import to_device



@ray.remote
class Client:
    def __init__(self, id, dataset, cfg):
        self.id = id
        self.dataset = dataset
        self.cfg = cfg
        self.data_loader = make_data_loader(self.dataset, self.cfg['optimizer']['batch_size'],
                                            self.cfg['optimizer']['num_local_steps'])

    def train(self, model_state_dict, optimizer_state_dict, scheduler_state_dict):
        model = make_model(self.cfg['model']).to(self.cfg['device'])
        model.load_state_dict(model_state_dict)

        optimizer = make_optimizer(model.parameters(), self.cfg['optimizer'])
        # optimizer_state_dict_ = optimizer.state_dict()
        # optimizer_state_dict_['param_groups'][0]['lr'] = optimizer_state_dict['param_groups'][0]['lr']
        optimizer.load_state_dict(optimizer_state_dict)

        scheduler = make_scheduler(optimizer, self.cfg['optimizer'])
        scheduler.load_state_dict(scheduler_state_dict)

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
                evaluation = logger.evaluate('train', 'batch', input, output)
                logger.append(evaluation, 'train', n=input_size)
        model = model.to('cpu')
        result = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(), 'logger_state_dict': logger.state_dict()}
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
