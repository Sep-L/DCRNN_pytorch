"""
DCRNN
Description: 
Author: LQZ
Time: 2022/3/22 10:37 
"""
import logging
import os
import pickle
import sys
import scipy.sparse as sp
import numpy as np


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def get_logger(log_dir, name, log_filename, log_level):
    logger = logging.getLogger(name)
    # 设置日志级别
    logger.setLevel(log_level)
    # 添加文件处理程序, 格式: 时间 - 名字 - 级别 - 信息
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # 添加标准输出处理
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    # 添加文件处理和标准输出处理
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # 先写入日志文件目录, 可有可无
    logger.info('Log directory: %s', log_dir)
    return logger


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj: 邻接矩阵 A
    :return: 正则化的拉普拉斯矩阵 (坐标格式的稀疏矩阵)
    """
    '''
    转换成坐标格式稀疏矩阵便于计算
    row  = np.array([0, 3, 1, 0])
    col  = np.array([0, 3, 1, 2])
    data = np.array([4, 5, 7, 9])
    scipy.sparse.coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    array([[4, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]])
    row, col 保存 data 值所对应的行和列
    '''
    adj = sp.coo_matrix(adj)
    # 按列相加得到度矩阵 D
    d = np.array(adj.sum(axis=1))
    # 求出 D^-1/2
    d_inv_sqrt = np.power(d, -0.5).flatten()
    # 上一步矩阵中可能会出现 0 的倒数无穷大的情况, 置 0
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # 从对角线构造一个稀疏矩阵, 前面的 D 都是数组形式只保留了对角线的数组, 这里变回矩阵
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # 计算正则化拉普拉斯矩阵 L
    # 稀疏单位矩阵 I
    i_mat = sp.eye(adj.shape[0])
    # L = I - (D^-1/2) * A * (D^-1/2), 因为 D A 均为对称矩阵, 所以也可以写成 L = I - (A * (D^-1/2))^T * (D^-1/2)
    normalized_laplacian = i_mat - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max, undirected=True):
    if undirected:
        # 取 adj_mx 和 adj_mx.T 每个位置上的最大值, reduce 默认 axis=0
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    # 计算得出坐标格式的正则化拉普拉斯矩阵
    l_mat = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        # 求 L 的 k 个特征值和特征向量 'LM': 最大值
        lambda_max, _ = sp.linalg.eigsh(l_mat, k=1, which='LM')
        lambda_max = lambda_max[0]
    # csr_matrix 比起 coo_matrix 可以进一步压缩记录稀疏矩阵时占用的空间, 详细看官网
    l_mat = sp.csr_matrix(l_mat)
    n, _ = l_mat.shape
    # scipy.sparse.identity 不用传入具体shape, 他是单位 n 阶矩阵传入 n 即可
    i_mat = sp.identity(n, format='csr', dtype=l_mat.dtype)
    # 使用切比雪夫多项式来降低复杂度
    l_mat = (2 / lambda_max * l_mat) - i_mat
    return l_mat.astype(np.float32)


def calculate_random_walk_matrix(adj_mx):
    return None
