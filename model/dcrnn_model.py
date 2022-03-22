"""
DCRNN
Description: 
Author: LQZ
Time: 2022/3/22 11:35 
"""
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """
    查询训练参数的个数
    :param model: 模型
    :return: 参数个数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        # 设置模型参数, 后面的是未设置时的默认参数
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        # 输入维度及 encoder 序列长度
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder

        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes, filter_type=self.filter_type)
             for _ in range(self.num_rnn_layers)]
        )

    def forward(self, inputs, hidden_state=None):
        """
        编码器
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: shape (num_layers, batch_size, self.hidden_state_size)
        :return:
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow