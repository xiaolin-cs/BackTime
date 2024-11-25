#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import os
import random
from dataset import load_raw_data
from trainer import Trainer
import yaml
from easydict import EasyDict as edict

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parser_args():
    # load configs/default_config.yaml
    default_config = yaml.load(open('configs/default_config.yaml', 'r'), Loader=yaml.FullLoader)

    # load training config
    config = yaml.load(open('configs/train_config.yaml'), Loader=yaml.FullLoader)['Train']

    # load dataset config
    config['Dataset'] = default_config['Dataset'][config['dataset']]
    config['Target_Pattern'] = default_config['Target_Pattern'][config['pattern_type']]

    config['Model'] = default_config['Model'][config['model_name']]
    config['Model']['c_out'] = config['Dataset']['num_of_vertices']
    config['Model']['enc_in'] = config['Dataset']['num_of_vertices']
    config['Model']['dec_in'] = config['Dataset']['num_of_vertices']

    config['Surrogate'] = default_config['Model'][config['surrogate_name']]
    config['Surrogate']['c_out'] = config['Dataset']['num_of_vertices']
    config['Surrogate']['enc_in'] = config['Dataset']['num_of_vertices']
    config['Surrogate']['dec_in'] = config['Dataset']['num_of_vertices']

    config = edict(config)
    return config


def main(config):
    # set gpu
    gpuid = config.gpuid
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda:0')
    print("CUDA:", USE_CUDA, DEVICE)

    seed_torch()

    data_config = config.Dataset
    if not data_config.use_timestamps:
        train_mean, train_std, train_data_seq, test_data_seq = load_raw_data(data_config)
    else:
        train_mean, train_std, train_data_seq, test_data_seq, train_data_stamps, test_data_stamps = load_raw_data(data_config)

    # set attacked variables
    spatial_poison_num = max(int(round(train_data_seq.shape[1] * config.alpha_s)), 1)

    atk_vars = np.arange(train_data_seq.shape[1])
    atk_vars = np.random.choice(atk_vars, size=spatial_poison_num, replace=False)
    atk_vars = torch.from_numpy(atk_vars).long().to(DEVICE)
    print('shape of attacked_variables', atk_vars.shape)

    # load target pattern
    target_pattern = config.Target_Pattern
    target_pattern = torch.tensor(target_pattern).float().to(DEVICE) * train_std

    exp_trainer = Trainer(config, atk_vars, target_pattern, train_mean, train_std, train_data_seq, test_data_seq, DEVICE)

    save_file = f'./checkpoints/attacker_{config.dataset}.pth'
    if os.path.exists(save_file):
        state = torch.load(save_file)
        exp_trainer.load_attacker(state)
        print('load attacker from', save_file)
    else:
        print('*' * 40)
        print('start training surrogate model and attacker')
        exp_trainer.train()

        state = exp_trainer.save_attacker()
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(state, save_file)

    print('*' * 40)
    print('start evaluating attack performance on a new model')
    exp_trainer.test()


if __name__ == "__main__":
    config = parser_args()
    main(config)

