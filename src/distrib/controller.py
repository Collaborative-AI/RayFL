import copy
import torch
import torch.nn.functional as F
from model import make_model
from config import cfg
from dataset import make_data_loader, split_dataset, make_batchnorm_stats
from module import to_device, make_optimizer, collate


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
        self.worker['server'] = Server(0, self.model_state_dict, self.optimizer_state_dict,
                                       self.scheduler_state_dict, self.metric_state_dict, self.logger_state_dict)
        self.worker['client'] = []
        for i in range(len(self.data_split['data'])):
            dataset_i = {k: split_dataset(dataset[k], self.data_split['data'][i]) for k in dataset}
            client_i = Client(i, dataset_i)
            self.worker['client'].append(client_i)
        return



class Server:
    def __init__(self, id, model_state_dict, optimizer_state_dict, scheduler_state_dict, metric_state_dict,
                 logger_state_dict):
        self.id = id
        self.model_state_dict = model_state_dict
        self.optimizer_state_dict = optimizer_state_dict
        self.scheduler_state_dict = scheduler_state_dict
        self.metric_state_dict = metric_state_dict
        self.logger_state_dict = logger_state_dict

    def distribute(self, client):
        for m in range(len(client)):
            if client[m].active:
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
        return

    def update(self, client):
        with torch.no_grad():
            valid_client = [client[i] for i in range(len(client)) if client[i].active]
            if len(valid_client) > 0:
                model = eval('models.{}()'.format(cfg['model_name']))
                model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
                model.load_state_dict(self.model_state_dict)
                global_optimizer = make_optimizer(model, 'global')
                global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                global_optimizer.zero_grad()
                weight = torch.ones(len(valid_client))
                weight = weight / weight.sum()
                for k, v in model.named_parameters():
                    parameter_type = k.split('.')[-1]
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        tmp_v = v.data.new_zeros(v.size())
                        for m in range(len(valid_client)):
                            tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                        v.grad = (v.data - tmp_v).detach()
                global_optimizer.step()
                self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                self.model_state_dict = save_model_state_dict(model.state_dict())
            for i in range(len(client)):
                client[i].active = False
        return

    def train(self, dataset, lr, metric, logger):
        data_loader = make_data_loader({'train': dataset}, 'server')['train']
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
        model.load_state_dict(self.model_state_dict)
        self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        optimizer = make_optimizer(model, 'local')
        optimizer.load_state_dict(self.optimizer_state_dict)
        model.train(True)
        for epoch in range(1, cfg['server']['num_epochs'] + 1):
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                logger.append(evaluation, 'train', n=input_size)
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        return


class Client:
    def __init__(self, client_id, model, data_split):
        self.client_id = client_id
        self.data_split = data_split
        self.model_state_dict = save_model_state_dict(model.state_dict())
        optimizer = make_optimizer(model, 'local')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.active = False
        self.beta = torch.distributions.beta.Beta(torch.tensor([cfg['alpha']]), torch.tensor([cfg['alpha']]))

    def make_hard_pseudo_label(self, soft_pseudo_label):
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_p.ge(cfg['threshold'])
        return hard_pseudo_label, mask

    def make_dataset(self, dataset, metric, logger):
        if 'sup' in cfg['loss_mode']:
            return dataset
        elif 'fix' in cfg['loss_mode']:
            with torch.no_grad():
                data_loader = make_data_loader({'train': dataset}, 'global', shuffle={'train': False})['train']
                model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
                model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=True))
                model.load_state_dict(self.model_state_dict)
                model.train(False)
                output = []
                target = []
                for i, input in enumerate(data_loader):
                    input = {'data': input['data'], 'target': input['target']}
                    input = collate(input)
                    input = to_device(input, cfg['device'])
                    output_ = model(input)
                    output_i = output_['target']
                    target_i = input['target']
                    output.append(output_i.cpu())
                    target.append(target_i.cpu())
                output_, input_ = {}, {}
                output_['target'] = torch.cat(output, dim=0)
                input_['target'] = torch.cat(target, dim=0)
                output_['target'] = F.softmax(output_['target'], dim=-1)
                new_target, mask = self.make_hard_pseudo_label(output_['target'])
                output_['mask'] = mask
                evaluation = metric.evaluate(['PAccuracy', 'MAccuracy', 'LabelRatio'], input_, output_)
                logger.append(evaluation, 'train', n=len(input_['target']))
                if torch.any(mask):
                    fix_dataset = copy.deepcopy(dataset)
                    fix_dataset.target = new_target.tolist()
                    mask = mask.tolist()
                    fix_dataset.data = list(compress(fix_dataset.data, mask))
                    fix_dataset.target = list(compress(fix_dataset.target, mask))
                    fix_dataset.id = list(compress(fix_dataset.id, mask))
                    if 'mix' in cfg['loss_mode']:
                        mix_dataset = copy.deepcopy(dataset)
                        mix_dataset.target = new_target.tolist()
                        mix_dataset = MixDataset(len(fix_dataset), mix_dataset)
                    else:
                        mix_dataset = None
                    return fix_dataset, mix_dataset
                else:
                    return None
        else:
            raise ValueError('Not valid loss mode')
        return

    def train(self, dataset, lr, metric, logger):
        if cfg['loss_mode'] == 'sup':
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model, 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            for epoch in range(1, cfg['local']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
        elif cfg['loss_mode'] == 'fix':
            fix_dataset, _ = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model, 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            for epoch in range(1, cfg['local']['num_epochs'] + 1):
                for i, input in enumerate(fix_data_loader):
                    input = {'aug_data': input['aug_data'], 'aug_target': input['target']}
                    input = collate(input)
                    input_size = input['aug_data'].size(0)
                    input = to_device(input, cfg['device'])
                    input['loss_mode'] = cfg['loss_mode']
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    input_ = {'target': input['aug_target']}
                    output_ = {'loss': output['loss'], 'target': output['aug_target']}
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input_, output_)
                    logger.append(evaluation, 'train', n=input_size)
        elif cfg['loss_mode'] == 'fix-mix':
            fix_dataset, mix_dataset = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            mix_data_loader = make_data_loader({'train': mix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=False))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model, 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            for epoch in range(1, cfg['local']['num_epochs'] + 1):
                for i, (fix_input, mix_input) in enumerate(zip(fix_data_loader, mix_data_loader)):
                    fix_input = collate(fix_input)
                    mix_input = collate(mix_input)
                    lam = self.beta.sample()[0]
                    fix_input['mix_data'] = (lam * fix_input['data'] + (1 - lam) * mix_input['data']).detach()
                    fix_input['mix_target'] = torch.stack([fix_input['target'], mix_input['target']], dim=-1)
                    input = {'aug_data': fix_input['aug_data'], 'aug_target': fix_input['target'],
                             'mix_data': fix_input['mix_data'], 'mix_target': fix_input['mix_target'], 'lam': lam}
                    input_size = input['aug_data'].size(0)
                    input = to_device(input, cfg['device'])
                    input['loss_mode'] = cfg['loss_mode']
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    input_ = {'target': input['aug_target']}
                    output_ = {'loss': output['loss'], 'target': output['aug_target']}
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input_, output_)
                    logger.append(evaluation, 'train', n=input_size)
        else:
            raise ValueError('Not valid loss mode')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        return


def make_controller(data_split, model, optimizer, scheduler, metric, logger):
    controller = Controller(data_split, model, optimizer, scheduler, metric, logger)
    return controller


def save_state_dict(state_dict):
    state_dict_ = {}
    for k, v in state_dict.items():
        if isinstance(state_dict_[k], torch.Tensor):
            state_dict_[k] = to_device(state_dict_[k], 'cpu')
        else:
            state_dict_[k] = copy.deepcopy(state_dict_[k])
    return state_dict_
