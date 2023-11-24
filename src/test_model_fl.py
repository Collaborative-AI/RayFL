import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, collate
from metric import make_metric, make_logger
from model import make_model
from module import save, to_device, process_control, resume, make_controller

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
    result_path = os.path.join('output', 'result')
    model_tag_path = os.path.join(model_path, cfg['model_tag'])
    checkpoint_path = os.path.join(model_tag_path, 'checkpoint')
    best_path = os.path.join(model_tag_path, 'best')
    dataset = make_dataset(cfg['data_name'])
    dataset = process_dataset(dataset)
    model = make_model(cfg)
    metric = make_metric(cfg['data_name'], {'train': ['Loss'], 'test': ['Loss']})
    result = resume(os.path.join(best_path, 'model'))
    data_split = result['data_split']
    model.load_state_dict(result['model_state_dict'])
    cfg['epoch'] = result['epoch']
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    controller = make_controller(data_split, model, None, None, metric, test_logger)
    controller.make_worker(dataset)
    controller.test()
    result = resume(os.path.join(checkpoint_path, 'model'))
    result = {'cfg': cfg, 'epoch': cfg['epoch'], 'logger_state_dict': {'train': result['logger_state_dict'],
                                                                       'test': test_logger.state_dict()}}
    save(result, os.path.join(result_path, cfg['model_tag']))
    return


if __name__ == "__main__":
    main()
