import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm
from timefeatures import time_features
import pandas as pd


def load_raw_data(dataset_config):
    if 'PEMS' in dataset_config.dataset_name:
        raw_data = np.load(dataset_config.data_filename)['data']
        train_data_seq = raw_data[:int(0.6 * raw_data.shape[0])]
        val_data_seq = raw_data[int(0.6 * raw_data.shape[0]):int(0.8 * raw_data.shape[0])]
        test_data_seq = raw_data[int(0.8 * raw_data.shape[0]):]

        train_mean = np.mean(train_data_seq, axis=(0, 1))
        train_std = np.std(train_data_seq, axis=(0, 1))
        if len(train_mean.shape) == 1:
            train_mean = train_mean[0]
            train_std = train_std[0]

        return train_mean, train_std, train_data_seq, test_data_seq

    elif dataset_config.dataset_name == 'ETTm1' or dataset_config.dataset_name == 'Weather':
        raw_data = pd.read_csv(dataset_config.data_filename)
        raw_data_feats = raw_data.values[:, 1:]
        raw_data_stamps = raw_data.values[:, 0]
        raw_data_stamps = pd.to_datetime(raw_data_stamps)

        # raw_data_stamps = raw_data_stamps.to_numpy()

        train_data_seq = raw_data_feats[:int(0.6 * raw_data_feats.shape[0])]
        val_data_seq = raw_data_feats[int(0.6 * raw_data_feats.shape[0]):int(0.8 * raw_data_feats.shape[0])]
        test_data_seq = raw_data_feats[int(0.8 * raw_data_feats.shape[0]):]

        train_data_stamps = raw_data_stamps[:int(0.6 * raw_data_stamps.shape[0])]
        val_data_stamps = raw_data_stamps[int(0.6 * raw_data_stamps.shape[0]):int(0.8 * raw_data_stamps.shape[0])]
        test_data_stamps = raw_data_stamps[int(0.8 * raw_data_stamps.shape[0]):]

        train_mean = np.mean(train_data_seq, axis=(0, 1))
        train_std = np.std(train_data_seq, axis=(0, 1))
        if len(train_mean.shape) == 1:
            train_mean = train_mean[0]
            train_std = train_std[0]

        return train_mean, train_std, train_data_seq, test_data_seq, train_data_stamps, test_data_stamps

    else:
        raise ValueError('Dataset not supported')


class TimeDataset(Dataset):
    def __init__(self, raw_data, mean, std, device, num_for_hist=12, num_for_futr=12, timestamps=None):
        # todo: use config to replace all the parameters
        """
        :param raw_data: for input, the raw data, shape (T, n, c), T: time span, n: sensor number, c: channels number
            for storage, the target pattern, shape (n, c, T)
        :param mean: the mean value of the raw data
        :param std: the standard deviation of the raw data
        :data: the clean data
        :poisoned_data: the poisoned data for attack
        """
        self.device = device
        self.data = raw_data
        self.use_timestamp = timestamps is not None
        if self.use_timestamp:
            self.timestamps = time_features(timestamps)
            self.timestamps = self.timestamps.transpose(1, 0)
            self.timestamps = torch.from_numpy(self.timestamps).float().to(self.device)
        else:
            self.timestamps = None

        # permutate the data to (n, c, T)
        if len(self.data.shape) == 2:
            self.data = self.data.reshape(self.data.shape[0], self.data.shape[1], 1)
        self.data = np.transpose(self.data, (1, 2, 0)).astype(np.float32)
        self.data = torch.from_numpy(self.data).float().to(self.device)

        self.init_poison_data()

        self.std = float(std)
        self.mean = float(mean)
        self.num_for_hist = num_for_hist
        self.num_for_futr = num_for_futr

    def __len__(self):
        return self.data.shape[-1] - self.num_for_hist - self.num_for_futr + 1

    def __getitem__(self, idx):
        """
        :param idx: the index of the data
        :return:
        """
        data = self.poisoned_data[:, 0:1, idx:idx + self.num_for_hist]
        data = self.normalize(data)

        poisoned_target = self.poisoned_data[:, 0, idx + self.num_for_hist:idx + self.num_for_hist + self.num_for_futr]

        clean_target = self.data[:, 0, idx + self.num_for_hist:idx + self.num_for_hist + self.num_for_futr]
        if not self.use_timestamp:
            return data, poisoned_target, clean_target, idx
        else:
            input_stamps = self.timestamps[idx:idx + self.num_for_hist]
            target_stamps = self.timestamps[idx + self.num_for_hist:idx + self.num_for_hist + self.num_for_futr]
            return data, poisoned_target, clean_target, input_stamps, target_stamps, idx

    def init_poison_data(self):
        self.poisoned_data = torch.clone(self.data).detach().to(self.device)

    def normalize(self, data):
        return (data - self.mean) / self.std

    def denormalize(self, data):
        return data * self.std + self.mean


class AttackEvaluateSet(TimeDataset):
    def __init__(self, attacker, raw_data, mean, std, device, num_for_hist=12, num_for_futr=12, timestamps=None):
        super(AttackEvaluateSet, self).__init__(raw_data, mean, std, device, num_for_hist, num_for_futr, timestamps)
        self.attacker = attacker

    def collate_fn(self, data):
        """
        :param data: the input data
        :return: the attacked data by the attacker
        """
        if self.use_timestamp:
            features, target, clean_target, input_stamps, target_stamps, idx = zip(*data)
        else:
            features, target, clean_target, idx = zip(*data)

        features = torch.stack(features, dim=0)
        clean_target = torch.stack(clean_target, dim=0)

        features = self.denormalize(features)

        data_bef = features[:, self.attacker.atk_vars, 0,
                   -self.attacker.trigger_len - self.attacker.bef_tgr_len:-self.attacker.trigger_len]
        triggers = self.attacker.predict_trigger(data_bef)[0]
        triggers = triggers.reshape(-1, self.attacker.atk_vars.shape[0], 1, self.attacker.trigger_len)
        features[:, self.attacker.atk_vars, :, -self.attacker.trigger_len:] = triggers

        target = clean_target.clone().detach().to(self.device)
        target[:, self.attacker.atk_vars, :self.attacker.pattern_len] = \
            self.attacker.target_pattern + features[:, self.attacker.atk_vars, :, -self.attacker.trigger_len - 1]

        features = self.normalize(features)
        return features, target, clean_target, idx
