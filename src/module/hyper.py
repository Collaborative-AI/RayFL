from config import cfg


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['batch_size'] = int(cfg['control']['batch_size'])
    cfg['step_period'] = int(cfg['control']['step_period'])
    cfg['num_steps'] = int(cfg['control']['num_steps'])
    cfg['eval_period'] = int(cfg['control']['eval_period'])
    cfg['optimizer_name'] = cfg['control']['optimizer_name']
    cfg['lr'] = float(cfg['control']['lr'])
    cfg['momentum'] = [float(x) for x in cfg['control']['momentum'].split('-')]
    cfg['scheduler_name'] = cfg['control']['scheduler_name']
    # cfg['num_epochs'] = 400

    cfg['collate_mode'] = 'dict'

    cfg['model'] = {}
    cfg['model']['model_name'] = cfg['model_name']
    data_shape = {'MNIST': [1, 28, 28], 'FashionMNIST': [1, 28, 28], 'SVHN': [3, 32, 32], 'CIFAR10': [3, 32, 32],
                  'CIFAR100': [3, 32, 32]}
    target_size = {'MNIST': 10, 'FashionMNIST': 10, 'SVHN': 10, 'CIFAR10': 10, 'CIFAR100': 100}
    cfg['model']['data_shape'] = data_shape[cfg['data_name']]
    cfg['model']['target_size'] = target_size[cfg['data_name']]
    cfg['model']['linear'] = {}
    cfg['model']['mlp'] = {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'}
    cfg['model']['cnn'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['resnet9'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['model']['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}

    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = cfg['optimizer_name']
    cfg[tag]['optimizer']['lr'] = cfg['lr']
    cfg[tag]['optimizer']['momentum'] = cfg['momentum'][0] if cfg['optimizer_name'] in ['SGD'] else cfg['momentum']
    cfg[tag]['optimizer']['weight_decay'] = 5e-4
    cfg[tag]['optimizer']['nesterov'] = True if cfg[tag]['optimizer']['momentum'] > 0 else False
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    cfg[tag]['optimizer']['scheduler_name'] = cfg['scheduler_name']

    if 'data_mode' in cfg['control']:
        cfg['data_mode'] = cfg['control']['data_mode']
        cfg['data_mode']['num_splits'] = int(cfg['data_mode']['num_splits'])

        cfg['dist_mode'] = cfg['control']['dist_mode']
        cfg['dist_mode']['active_ratio'] = float(cfg['dist_mode']['active_ratio'])
        cfg['dist_mode']['num_steps'] = int(cfg['dist_mode']['num_steps'])
        cfg['dist_mode']['optimizer_name'] = cfg['dist_mode']['optimizer_name']
        cfg['dist_mode']['lr'] = float(cfg['dist_mode']['lr'])
        cfg['dist_mode']['momentum'] = [float(x) for x in cfg['dist_mode']['momentum'].split('-')]
        cfg['dist_mode']['scheduler_name'] = cfg['dist_mode']['scheduler_name']


        cfg[tag]['local'] = {}
        cfg[tag]['local']['device'] = cfg['device']
        cfg[tag]['local']['model'] = cfg['model']
        cfg[tag]['local']['dist_mode'] = cfg['dist_mode']
        cfg[tag]['local']['optimizer'] = {}
        cfg[tag]['local']['optimizer']['optimizer_name'] = cfg['dist_mode']['optimizer_name']
        cfg[tag]['local']['optimizer']['lr'] = cfg['dist_mode']['lr']
        cfg[tag]['local']['optimizer']['momentum'] = cfg['dist_mode']['momentum'][0] \
            if cfg['dist_mode']['optimizer_name'] in ['SGD'] else cfg['dist_mode']['momentum']
        cfg[tag]['local']['optimizer']['weight_decay'] = 5e-4
        cfg[tag]['local']['optimizer']['nesterov'] = True if cfg[tag]['local']['optimizer']['momentum'] > 0 else False
        cfg[tag]['local']['optimizer']['batch_size'] = {'train': cfg['batch_size'],
                                                        'test': cfg['batch_size']}
        cfg[tag]['local']['optimizer']['step_period'] = cfg['step_period']
        cfg[tag]['local']['optimizer']['num_steps'] = cfg['num_steps']
        cfg[tag]['local']['optimizer']['num_local_steps'] = cfg['dist_mode']['num_steps']
        cfg[tag]['local']['optimizer']['scheduler_name'] = 'CosineAnnealingLR'

        cfg[tag]['global'] = {}
        cfg[tag]['global']['optimizer'] = {}
        cfg[tag]['global']['optimizer']['optimizer_name'] = cfg['optimizer_name']
        cfg[tag]['global']['optimizer']['lr'] = cfg['lr']
        cfg[tag]['global']['optimizer']['momentum'] = cfg['momentum'][0] \
            if cfg['optimizer_name'] in ['SGD'] else cfg['momentum']
        cfg[tag]['global']['optimizer']['weight_decay'] = 0
        cfg[tag]['global']['optimizer']['nesterov'] = True if cfg[tag]['global']['optimizer']['momentum'] > 0 else False
        cfg[tag]['global']['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
        cfg[tag]['global']['optimizer']['step_period'] = cfg['step_period']
        cfg[tag]['global']['optimizer']['num_steps'] = cfg['num_steps']
        cfg[tag]['global']['optimizer']['scheduler_name'] = 'None'
    return