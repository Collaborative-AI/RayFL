import argparse
import datetime
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, collate, make_split
from metric import make_metric, make_logger
from model import make_model, make_optimizer, make_scheduler
from distrib import make_controller
from module import save, to_device, process_control, resume, makedir_exist_ok

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiment']))
    for i in range(cfg['num_experiment']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    model_path = os.path.join('output', 'model')
    model_tag_path = os.path.join(model_path, cfg['model_tag'])
    checkpoint_path = os.path.join(model_tag_path, 'checkpoint')
    best_path = os.path.join(model_tag_path, 'best')
    dataset = make_dataset(cfg['data_name'])
    dataset = process_dataset(dataset)
    model = make_model(cfg['model_name'])
    data_split = make_split(dataset, cfg['data_mode']['num_split'], cfg['data_mode']['split_mode'],
                            stat_mode=cfg['data_mode']['stat_mode'])
    result = resume(os.path.join(checkpoint_path, 'model'), resume_mode=cfg['resume_mode'])
    cfg['epoch'] = 1
    optimizer = {'local': make_optimizer(model.parameters(), 'local'),
                 'global': make_optimizer(model.parameters(), 'global')}
    scheduler = {'local': make_scheduler(optimizer['local'], 'local'),
                 'global': make_scheduler(optimizer['global'], 'global')}
    metric = make_metric({'train': ['Loss'], 'test': ['Loss']})
    logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    if result is not None:
        cfg['epoch'] = result['epoch']
        model.load_state_dict(result['model_state_dict'])
        optimizer['local'].load_state_dict(result['optimizer_state_dict']['local'])
        optimizer['global'].load_state_dict(result['optimizer_state_dict']['global'])
        scheduler['local'].load_state_dict(result['scheduler_state_dict']['local'])
        scheduler['global'].load_state_dict(result['scheduler_state_dict']['global'])
        metric.load_state_dict(result['metric_state_dict'])
        logger.load_state_dict(result['logger_state_dict'])
    controller = make_controller(data_split, model, optimizer, scheduler, metric, logger)
    controller.make_worker(dataset)
    for epoch in range(cfg['epoch'], cfg[cfg['model_name']]['num_epochs'] + 1):
        cfg['epoch'] = epoch
        controller.train()
        controller.test()
        controller.update()
        result = {'cfg': cfg, 'epoch': cfg['epoch'] + 1, 'model_state_dict': controller.model_state_dict(),
                  'optimizer_state_dict': controller.optimizer_state_dict(),
                  'scheduler_state_dict': controller.scheduler_state_dict(),
                  'metric_state_dict': controller.metric_state_dict(),
                  'logger_state_dict': controller.logger_state_dict()}
        save(result, os.path.join(checkpoint_path, 'model'))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            makedir_exist_ok(best_path)
            shutil.copy(os.path.join(checkpoint_path, 'model'), os.path.join(best_path, 'model'))
        logger.save(True)
        logger.reset()
    return


if __name__ == "__main__":
    main()
