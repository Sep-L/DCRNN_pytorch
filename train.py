"""
DCRNN
Description: 
Author: LQZ
Time: 2022/3/22 10:22 
"""
import argparse

import yaml

from model.dcrnn_supervisor import DCRNNSupervisor
from utils import load_pickle


def main(args):
    # 读取配置文件
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        # 读取道路信息
        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(graph_pkl_filename)
        # 初始化 DCRNNSupervisor
        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str, help='Configuration filename for model')
    main_args = parser.parse_args()
    main(main_args)