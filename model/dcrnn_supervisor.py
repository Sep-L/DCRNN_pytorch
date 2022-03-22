"""
DCRNN
Description: 
Author: LQZ
Time: 2022/3/22 10:22 
"""
import time

from torch.utils.tensorboard import SummaryWriter

import data_load


class DCRNNSupervisor:
    def __init__(self, adj_mx, **kwargs):
        # 所有参数
        self._kwargs = kwargs
        # 设置日志相关配置
        self._log_dir = self._get_log_dir(kwargs)
        self._writer = SummaryWriter('runs/' + self._log_dir)
        log_level = self._kwargs['log'].get('log_level')
        log_filename = self._kwargs['log'].get('log_filename')
        self._logger = utils.get_logger(self._log_dir, __name__, log_filename, log_level)

        # 各个部分的参数
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        # 数据集读取
        self._data = data_load.load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        # 读取配置
        self.num_nodes = int(self._model_kwargs.get('num_nodes'))
        self.input_dim = int(self._model_kwargs.get('input_dim'))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim'))
        self.use_curriculum_learning = bool(self._model_kwargs.get('use_curriculum_learning'))
        self.horizon = int(self._model_kwargs.get('horizon'))  # for the decoder

        # setup model
        dcrnn_model = DCRNNModel(adj_mx, self._logger, **self._model_kwargs)
        self.dcrnn_model = dcrnn_model.cuda() if torch.cuda.is_available() else dcrnn_model
        self._logger.info("Model created")

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model()

        def _get_log_dir(kwargs):
            log_dir = kwargs['train'].get('log_dir')
            if log_dir is not None:
                # 基础参数
                batch_size = kwargs['data'].get('batch_size')
                learning_rate = kwargs['train'].get('base_lr')
                max_diffusion_step = kwargs['model'].get('max_diffusion_step')
                num_rnn_layers = kwargs['model'].get('num_rnn_layers')
                rnn_units = kwargs['model'].get('rnn_units')
                horizon = kwargs['model'].get('horizon')
                filter_type = kwargs['model'].get('filter_type')
                # 空间卷积形式 'L'普通 'R'随机游走 'DR'双向随机游走
                filter_type_abbr = 'L'
                if filter_type == 'random_walk':
                    filter_type_abbr = 'R'
                elif filter_type == 'dual_random_walk':
                    filter_type_abbr = 'DR'
                # 标明当前运行所对应配置
                run_id = 'dcrnn_type%s_step%d_h_%d_lr_%g_bs_%d_%s/' \
                         % (filter_type_abbr, max_diffusion_step, horizon, learning_rate, batch_size, time.strftime('%m%d%H%M%S'))
                log_base_dir = kwargs.get('log_base_dir')
                log_dir = os.path.join(log_base_dir, run_id)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
            return log_dir