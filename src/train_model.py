import argparse
import datetime
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset
from metric import make_logger
from model import make_model, make_optimizer, make_scheduler
from module import check, resume, to_device, process_control

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
    result = resume(checkpoint_path, resume_mode=cfg['resume_mode'])
    if result is None:
        cfg['iteration'] = 0
        model = model.to(cfg['device'])
        optimizer = make_optimizer(model.parameters(), cfg[cfg['model_name']])
        scheduler = make_scheduler(optimizer, cfg[cfg['model_name']])
        logger = make_logger(os.path.join(tag_path, 'logger', 'train', 'runs'))
    else:
        cfg['iteration'] = result['cfg']['iteration']
        model = model.to(cfg['device'])
        optimizer = make_optimizer(model.parameters(), cfg[cfg['model_name']])
        scheduler = make_scheduler(optimizer, cfg[cfg['model_name']])
        logger = make_logger(os.path.join(tag_path, 'logger', 'train', 'runs'))
        model.load_state_dict(result['model'])
        optimizer.load_state_dict(result['optimizer'])
        scheduler.load_state_dict(result['scheduler'])
        logger.load_state_dict(result['logger'])
        logger.reset()
    data_loader = make_data_loader(dataset, cfg[cfg['model_name']]['batch_size'], cfg['num_steps'], cfg['iteration'],
                                   cfg['step_period'])
    data_iterator = enumerate(data_loader['train'])
    while cfg['iteration'] < cfg['num_steps']:
        train(data_iterator, model, optimizer, scheduler, logger)
        test(data_loader['test'], model, logger)
        result = {'cfg': cfg, 'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                  'logger': logger.state_dict()}
        check(result, checkpoint_path)
        if logger.compare('test'):
            shutil.copytree(checkpoint_path, best_path, dirs_exist_ok=True)
        logger.reset()
    return


def train(data_loader, model, optimizer, scheduler, logger):
    model.train(True)
    start_time = time.time()
    with logger.profiler:
        for i, input in data_loader:
            if i % cfg['step_period'] == 0 and cfg['profile']:
                logger.profiler.step()
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
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
            idx = cfg['iteration'] % cfg['eval_period']
            if idx % int(cfg['eval_period'] * cfg['log_interval']) == 0 and (i + 1) % cfg['step_period'] == 0:
                step_time = (time.time() - start_time) / (idx + 1)
                lr = optimizer.param_groups[0]['lr']
                epoch_finished_time = datetime.timedelta(
                    seconds=round((cfg['eval_period'] - (idx + 1)) * step_time))
                exp_finished_time = datetime.timedelta(
                    seconds=round((cfg['num_steps'] - (cfg['iteration'] + 1)) * step_time))
                info = {'info': ['Model: {}'.format(cfg['tag']),
                                 'Train Epoch: {}({:.0f}%)'.format((cfg['iteration'] // cfg['eval_period']) + 1,
                                                                   100. * idx / cfg['eval_period']),
                                 'Learning rate: {:.6f}'.format(lr),
                                 'Epoch Finished Time: {}'.format(epoch_finished_time),
                                 'Experiment Finished Time: {}'.format(exp_finished_time)]}
                logger.append(info, 'train')
                print(logger.write('train'))
            if (i + 1) % cfg['step_period'] == 0:
                cfg['iteration'] += 1
            if (idx + 1) % cfg['eval_period'] == 0 and (i + 1) % cfg['step_period'] == 0:
                break
    return


def test(data_loader, model, logger):
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            evaluation = logger.evaluate('test', 'batch', input, output)
            logger.append(evaluation, 'test', input_size)
        evaluation = logger.evaluate('test', 'full')
        logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['tag']),
                         'Test Epoch: {}({:.0f}%)'.format(cfg['iteration'] // cfg['eval_period'], 100.)]}
        logger.append(info, 'test')
        print(logger.write('test'))
        logger.save(True)
    return


if __name__ == "__main__":
    main()
