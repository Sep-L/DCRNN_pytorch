"""
DCRNN
Description: 
Author: LQZ
Time: 2022/3/22 9:46 
"""


import os
import torch
import numpy as np


# 数据归一化处理
class StandardScaler:
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(dataset_dir, train_batch_size, val_batch_size, test_batch_size):
    data = {}
    for category in ['train', 'val', 'test']:
        # 拼出数据文件路径并读取
        data_path = os.path.join(dataset_dir, category + '.npz')
        cat_data = np.load(data_path)
        # 全部放在data中
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    # 以训练集数据为基准做标准化
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # 数据归一
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])

    # 用torch把数据分批次，训练集打乱
    torch_data = {}
    # train-----------------------------------------------------------------------------------------------------------
    train_dataset = torch.utils.data.TensorDataset(data['x_train'], data['y_train'])
    torch_data['train_loader'] = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    # val-------------------------------------------------------------------------------------------------------------
    train_dataset = torch.utils.data.TensorDataset(data['x_val'], data['y_val'])
    torch_data['val_loader'] = torch.utils.data.DataLoader(train_dataset, batch_size=val_batch_size, shuffle=False)
    # test------------------------------------------------------------------------------------------------------------
    train_dataset = torch.utils.data.TensorDataset(data['x_tset'], data['y_test'])
    torch_data['test_loader'] = torch.utils.data.DataLoader(train_dataset, batch_size=test_batch_size, shuffle=False)
    # regular---------------------------------------------------------------------------------------------------------
    torch_data['scaler'] = scaler

    del data
    return torch_data
