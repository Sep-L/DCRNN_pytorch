"""
DCRNN
Description: 
Author: LQZ
Time: 2022/3/22 12:24 
"""
import torch

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter((torch.empty(*shape, device=device)))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)), nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)), biases)

        return self._biases_dict[length]


class DCGRUCell(torch.nn.Module):
    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, nonlinear='tanh',
                 filter_type="laplacian", use_gc_for_ru=True):
        super().__init__()
        self._activation = torch.tanh if nonlinear == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._support_matrix = []
        if filter_type == "laplacian":
            self._support_matrix.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            self._support_matrix.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            self._support_matrix.append(utils.calculate_random_walk_matrix(adj_mx).T)
            self._support_matrix.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        else:
            self._support_matrix.append(utils.calculate_scaled_laplacian(adj_mx))

        # rnn_network: self, 即 DCGRUCell
        # layer_type: 标明是哪一层
        self._fc_params = LayerParams(self, 'fc')
        self._gconv_params = LayerParams(self, 'gconv')

