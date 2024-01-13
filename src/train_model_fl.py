# uncomment the following 2 lines to run it locally
import sys
sys.path.insert(0, '/RayFL/src')


import argparse
import os
import shutil
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_split
from dataset import process_dataset
from dataset import make_dataset
from metric import make_logger
from model import make_model, make_optimizer, make_scheduler
from module import check, resume, process_control, make_controller

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['tag'] = '_'.join([x for x in tag_list if x])
        print('Experiment: {}'.format(cfg['tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    path = os.path.join('output', 'exp')
    tag_path = os.path.join(path, cfg['tag'])
    checkpoint_path = os.path.join(tag_path, 'checkpoint')
    best_path = os.path.join(tag_path, 'best')
    dataset = make_dataset(cfg['data_name'])
    dataset = process_dataset(dataset)
    model = make_model(cfg)
    data_split = make_split(dataset, cfg['data_mode']['num_splits'], cfg['data_mode']['split_mode'],
                            stat_mode=cfg['data_mode']['stat_mode'])
    result = resume(os.path.join(checkpoint_path, 'model'), resume_mode=cfg['resume_mode'])
    if result is None:
        cfg['iteration'] = 0
        model = model.to(cfg['device'])
        optimizer = {'local': make_optimizer([torch.nn.Parameter(torch.tensor([0.]))], cfg['local']),
                     'global': make_optimizer(model.parameters(), cfg['global'])}
        scheduler = {'local': make_scheduler(optimizer['local'], cfg['local']),
                     'global': make_scheduler(optimizer['global'], cfg['global'])}
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['tag'])))
    else:
        cfg['iteration'] = result['cfg']['iteration']
        model = model.to(cfg['device'])
        optimizer = {'local': make_optimizer([torch.nn.Parameter(torch.tensor([0.]))], cfg['local']),
                     'global': make_optimizer(model.parameters(), cfg['global'])}
        scheduler = {'local': make_scheduler(optimizer['local'], cfg['local']),
                     'global': make_scheduler(optimizer['global'], cfg['global'])}
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['tag'])))
        model.load_state_dict(result['model'])
        optimizer['local'].load_state_dict(result['optimize']['local'])
        optimizer['global'].load_state_dict(result['optimize']['global'])
        scheduler['local'].load_state_dict(result['scheduler']['local'])
        scheduler['global'].load_state_dict(result['scheduler']['global'])
        logger.load_state_dict(result['logger'])
    controller = make_controller(data_split, model, optimizer, scheduler, logger)
    controller.make_worker(dataset)
    while cfg['iteration'] < cfg['num_steps']:
        controller.train()
        controller.test()
        controller.update()
        result = {'cfg': cfg, 'iteration': cfg['iteration'], 'data_split': data_split,
                  'model': controller.model_state_dict(),
                  'optimizer': controller.optimizer_state_dict(),
                  'scheduler': controller.scheduler_state_dict(),
                  'logger': controller.logger_state_dict()}
        check(result, checkpoint_path)
        if logger.compare('test'):
            shutil.copytree(checkpoint_path, best_path, dirs_exist_ok=True)
        logger.reset()
    return


if __name__ == "__main__":
    main()
