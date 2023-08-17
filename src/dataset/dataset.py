import copy
import dataset
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from config import cfg

data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))}


def make_dataset(data_name, verbose=True):
    dataset_ = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = os.path.join('data', data_name)
    if data_name in ['MNIST', 'FashionMNIST']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['train'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset_['test'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['CIFAR10', 'CIFAR100']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['train'].transform = dataset.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset_['test'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['SVHN']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['train'].transform = dataset.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset_['test'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset_


def input_collate(batch):
    return {key: [b[key] for b in batch] for key in batch[0]}


def make_data_collate(collate_mode):
    if collate_mode == 'dict':
        return input_collate
    elif collate_mode == 'default':
        return default_collate
    else:
        raise ValueError('Not valid collate mode')


def make_data_loader(dataset, tag, batch_size=None, shuffle=None, sampler=None):
    data_loader = {}
    cfg['num_steps'] = {}
    for k in dataset:
        batch_size_ = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        shuffle_ = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size_, shuffle=shuffle_,
                                        pin_memory=cfg['pin_memory'], num_workers=cfg['num_worker'],
                                        collate_fn=make_data_collate(cfg['collate_mode']),
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size_, sampler=sampler[k],
                                        pin_memory=cfg['pin_memory'], num_workers=cfg['num_worker'],
                                        collate_fn=make_data_collate(cfg['collate_mode']),
                                        worker_init_fn=np.random.seed(cfg['seed']))
        cfg['num_steps'][k] = len(data_loader[k])
    return data_loader


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input


def process_dataset(dataset):
    processed_dataset = dataset
    cfg['data_size'] = {k: len(processed_dataset[k]) for k in processed_dataset}
    cfg['target_size'] = processed_dataset['train'].target_size
    return processed_dataset


def make_split(dataset, num_split, split_mode, **kwargs):
    if split_mode == 'horiz':
        stat_mode = kwargs['stat_mode']
        if stat_mode == 'iid':
            data_split, target_split = iid(dataset, num_split)
        elif 'noniid' in stat_mode:
            data_split, target_split = noniid(dataset, num_split, stat_mode)
        else:
            raise ValueError('Not valid data split mode')
    else:
        raise ValueError('Not valid data split mode')
    split = {'data': data_split, 'target': target_split}
    return split


def split_dataset(dataset, idx):
    dataset_ = copy.deepcopy(dataset)
    dataset_.data = [dataset.data[s] for s in idx]
    dataset_.target = [dataset.target[s] for s in idx]
    dataset_.id = list(range(len(dataset_.data)))
    return dataset_


def iid(dataset, num_splits):
    data_split = [{k: None for k in dataset} for _ in range(num_splits)]
    target_split = [{k: None for k in dataset} for _ in range(num_splits)]
    for k in dataset:
        idx_k = torch.randperm(len(dataset[k]))
        data_split_k = torch.tensor_split(idx_k, num_splits)
        for i in range(num_splits):
            data_split[i][k] = data_split_k[i].tolist()
            target_i_k = torch.tensor(dataset[k].target)[data_split[i][k]]
            if k == 'train':
                unique_target_i_k, num_target_i = torch.unique(target_i_k, sorted=True, return_counts=True)
                target_split[i][k] = {unique_target_i_k[m].item(): num_target_i[m].item()
                                      for m in range(len(unique_target_i_k))}
            else:
                target_split[i][k] = {target: int((target_i_k == target).sum()) for target in target_split[i]['train']}
    return data_split, target_split


def noniid(dataset, num_splits, stat_mode):
    data_split_mode_list = stat_mode.split('~')
    data_split_mode_tag = data_split_mode_list[-2]
    target_size = len(torch.unique(torch.tensor(dataset['train'].target)))
    if data_split_mode_tag == 'c':
        data_split = [{k: [] for k in dataset} for _ in range(num_splits)]
        shard_per_user = int(data_split_mode_list[-1])
        shard_per_class = int(np.ceil(shard_per_user * num_splits / target_size))
        target_idx_split = [{k: None for k in dataset} for _ in range(target_size)]
        for k in dataset:
            target = torch.tensor(dataset[k].target)
            for target_i in range(target_size):
                target_idx = torch.where(target == target_i)[0]
                num_leftover = len(target_idx) % shard_per_class
                leftover = target_idx[-num_leftover:] if num_leftover > 0 else []
                target_idx = target_idx[:-num_leftover] if num_leftover > 0 else target_idx
                target_idx = target_idx.reshape((shard_per_class, -1)).tolist()
                for i, leftover_target_idx in enumerate(leftover):
                    target_idx[i].append(leftover_target_idx.item())
                target_idx_split[target_i][k] = target_idx
        target_split_key = []
        for i in range(shard_per_class):
            target_split_key.append(torch.randperm(target_size))
        target_split_key = torch.cat(target_split_key, dim=0)
        target_split = [{k: None for k in dataset} for _ in range(num_splits)]
        exact_size = shard_per_user * num_splits
        exact_target_split, leftover_target_split = target_split_key[:exact_size].tolist(), \
            {k: target_split_key[exact_size:].tolist() for k in dataset}
        for i in range(0, exact_size, shard_per_user):
            target_split_i = exact_target_split[i:i + shard_per_user]
            for j in range(len(target_split_i)):
                target_i_j = target_split_i[j]
                for k in dataset:
                    idx = torch.randint(len(target_idx_split[target_i_j][k]), (1,)).item()
                    data_split[i // shard_per_user][k].extend(target_idx_split[target_i_j][k].pop(idx))
                    if target_i_j in leftover_target_split[k]:
                        idx = torch.randint(len(target_idx_split[target_i_j][k]), (1,)).item()
                        data_split[i // shard_per_user][k].extend(target_idx_split[target_i_j][k].pop(idx))
                        leftover_idx = leftover_target_split[k].index(target_i_j)
                        leftover_target_split[k].pop(leftover_idx)
                    target_i_j_k = torch.tensor(dataset[k].target)[data_split[i // shard_per_user][k]]
                    if k == 'train':
                        unique_target_i_k, num_target_i = torch.unique(target_i_j_k, sorted=True, return_counts=True)
                        target_split[i // shard_per_user][k] = {unique_target_i_k[m].item(): num_target_i[m].item()
                                                                for m in range(len(unique_target_i_k))}
                    else:
                        target_split[i // shard_per_user][k] = {x: int((target_i_j_k == x).sum()) for x in
                                                                target_split[i // shard_per_user]['train']}
    elif data_split_mode_tag == 'd':
        data_split, target_split = None, None
        min_size = 0
        required_min_size = 10
        while min_size < required_min_size:
            beta = float(data_split_mode_list[-1])
            dir = torch.distributions.dirichlet.Dirichlet(torch.tensor(beta).repeat(num_splits))
            data_split = [{k: [] for k in dataset} for _ in range(num_splits)]
            for target_i in range(target_size):
                proportions = dir.sample()
                for k in dataset:
                    target = torch.tensor(dataset[k].target)
                    target_idx = torch.where(target == target_i)[0]
                    proportions = torch.tensor([p * (len(data_split_idx[k]) < (len(target) / num_splits))
                                                for p, data_split_idx in zip(proportions, data_split)])
                    proportions = proportions / proportions.sum()
                    split_idx = (torch.cumsum(proportions, dim=-1) * len(target_idx)).long().tolist()[:-1]
                    split_idx = torch.tensor_split(target_idx, split_idx)
                    for i in range(len(split_idx)):
                        data_split[i][k].extend(split_idx[i].tolist())
            min_size = min([len(data_split[i]['train']) for i in range(len(data_split))])
            target_split = [{k: None for k in dataset} for _ in range(num_splits)]
            for i in range(num_splits):
                for k in dataset:
                    target_i_k = torch.tensor(dataset[k].target)[data_split[i][k]]
                    if k == 'train':
                        unique_target_i_k, num_target_i = torch.unique(target_i_k, sorted=True, return_counts=True)
                        target_split[i][k] = {unique_target_i_k[m].item(): num_target_i[m].item() for m in
                                              range(len(unique_target_i_k))}
                    else:
                        target_split[i][k] = {x: (target_i_k == x).sum().item() for x in target_split[i]['train']}
    else:
        raise ValueError('Not valid data split mode tag')
    return data_split, target_split
