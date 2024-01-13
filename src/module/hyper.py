from config import cfg


def process_control():
    cfg['collate_mode'] = 'dict'
    data_shape = {'MNIST': [1, 28, 28], 'FashionMNIST': [1, 28, 28], 'SVHN': [3, 32, 32], 'CIFAR10': [3, 32, 32],
                  'CIFAR100': [3, 32, 32]}
    target_size = {'MNIST': 10, 'FashionMNIST': 10, 'SVHN': 10, 'CIFAR10': 10, 'CIFAR100': 100}

    cfg['linear'] = {}
    cfg['mlp'] = {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'}
    cfg['cnn'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['resnet9'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}
    cfg['data_name'] = cfg['control']['data_name']
    cfg['data_shape'] = data_shape[cfg['data_name']]
    cfg['target_size'] = target_size[cfg['data_name']]

    cfg['model_name'] = cfg['control']['model_name']
    model_name = cfg['model_name']
    cfg[model_name]['shuffle'] = {'train': True, 'test': False}
    cfg[model_name]['optimizer_name'] = 'SGD'
    cfg[model_name]['lr'] = 1e-1
    cfg[model_name]['momentum'] = 0.9
    cfg[model_name]['weight_decay'] = 5e-4
    cfg[model_name]['nesterov'] = True
    cfg[model_name]['scheduler_name'] = 'CosineAnnealingLR'
    cfg[model_name]['num_epochs'] = 400
    cfg[model_name]['batch_size'] = {'train': 250, 'test': 250}

    cfg['batch_size'] = 250

    cfg['step_period'] = 1
    cfg['num_steps'] = 80000
    cfg['eval_period'] = 200
    # cfg['num_epochs'] = 400

    if 'data_mode' in cfg['control']:
        cfg['data_mode'] = cfg['control']['data_mode']
        cfg['data_mode']['num_splits'] = int(cfg['data_mode']['num_splits'])
        cfg['comm_mode'] = cfg['control']['comm_mode']
        cfg['comm_mode']['active_ratio'] = float(cfg['comm_mode']['active_ratio'])
        cfg['comm_mode']['num_steps'] = int(cfg['comm_mode']['num_steps'])
        cfg['test_mode'] = cfg['control']['test_mode']
        cfg['local'] = {}
        cfg['local']['shuffle'] = {'train': True, 'test': False}
        cfg['local']['optimizer_name'] = 'SGD'
        cfg['local']['lr'] = 3e-2
        cfg['local']['momentum'] = 0.9
        cfg['local']['weight_decay'] = 5e-4
        cfg['local']['nesterov'] = True
        cfg['local']['num_epochs'] = 800
        cfg['local']['batch_size'] = {'train': 250, 'test': 500}
        cfg['local']['scheduler_name'] = 'CosineAnnealingLR'
        cfg['local']['num_steps'] = cfg['comm_mode']['num_steps']
        cfg['global'] = {}
        cfg['global']['shuffle'] = {'train': True, 'test': False}
        cfg['global']['optimizer_name'] = 'SGD'
        cfg['global']['lr'] = 1
        cfg['global']['momentum'] = 0
        cfg['global']['weight_decay'] = 0
        cfg['global']['nesterov'] = False
        cfg['global']['num_steps'] = cfg['num_steps']
        cfg['global']['batch_size'] = {'train': 250, 'test': 500}
        cfg['global']['scheduler_name'] = 'None'
    return
