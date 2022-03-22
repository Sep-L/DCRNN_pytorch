"""
DCRNN
Description: 读取原始数据并对数据作预处理
Author: LQZ
Time: 2022/3/18 16:26 
"""
import argparse
import os
import numpy as np
import pandas as pd


def load_data(args):
    # 读取文件数据
    data_frame = pd.read_hdf(args.traffic_filename)
    # 当前时刻为 0, 用过去的 x_offsets 来预测未来的 y_offsets 时间段的信息
    x_offsets = (np.arange(-11, 1, 1))
    # 要预测的时间段
    y_offsets = np.arange(1, 13, 1)
    # 生成一一对应的历史数据和要预测的数据
    x, y = generate_graph_seq2seq_io_data(data_frame, x_offsets, y_offsets, args)
    print("x shape: ", x.shape, ", y shape: ", y.shape)

    # 训练: 验证: 测试 = 7: 1: 2
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    # train
    x_train, y_train = x[: num_train], y[: num_train]
    # val
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    x_offsets = x_offsets.reshape(list(x_offsets.shape) + [1])
    y_offsets = y_offsets.reshape(list(y_offsets.shape) + [1])

    # 将数据写入 npz 文件
    for cat in ["train", "val", "test"]:
        # locals() 可以获取当前所有的局部变量, 从而得到上面的数据
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        # 保存时, x, y, x_offsets, y_offsets 应该是一一对应的
        # 但是 x,y 都是列的形式, x_offsets, y_offsets 还是行, 所以要转换下 x_offsets 和 y_offsets 的形式
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            # x_offsets: (t) -> list: [t] -> +[1]: [t, 1] -> reshape: (t, 1)
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            # y_offsets: (t) -> list: [t] -> +[1]: [t, 1] -> reshape: (t, 1)
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def generate_graph_seq2seq_io_data(data_frame, x_offsets, y_offsets, args, scaler=None):
    """
    Generate samples from
    :param data_frame: 读取的数据
    :param x_offsets: 历史数据时间段相对当前时刻的偏移量
    :param y_offsets: 历史数据时间段相对当前时刻的偏移量
    :param args: 设置 add_time_in_day 以及 add_day_in_week 的参数
    :param scaler: 是否归一化
    :return: x: (epoch_size, num_samples + min(x_offsets) - max(y_offsets), num_nodes, features)
             y: (epoch_size, num_samples + min(x_offsets) - max(y_offsets), num_nodes, features)
    """
    # num_samples 样本总数, num_nodes 节点数
    num_samples, num_nodes = data_frame.shape
    # 扩展一维表示特征
    data = np.expand_dims(data_frame.values, axis=-1)

    # 方便接下来拼接时间关系
    data_list = [data]
    # 是否添加一天中的每个时间信息
    if args.add_time_in_day:
        # time_index: (num_samples) 每个样本在当天的时刻 (0.xxx天)
        time_index = (data_frame.index.values - data_frame.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        # 扩展到每个节点 (1, num_nodes, num_samples) 转置调整格式 -> (num_samples, num_nodes, 1)
        time_in_day = np.tile(time_index, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    # 是否添加一周中的每天时间信息
    if args.add_day_in_week:
        # day_in_week: (num_samples, num_nodes, 7)
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        '''
        把每个样本所对应的周标成 1, 如星期三则第三维的对应位置置 1
        等效如下, 但是运算速度快非常多
        for i in range(num_samples):
            res[i, :, data_frame.index.dayofweek[i]] = 1
        '''
        day_in_week[np.arange(num_samples), :, data_frame.index.dayofweek] = 1
        data_list.append(day_in_week)
    # data: (num_samples, num_nodes, 1 + 1 + 7), 第三维 1(自身的速度数据) + 1(天的时间信息) + 7(周的时间信息)
    data = np.concatenate(data_list, axis=-1)

    x, y = [], []
    '''
    t 代表最后一次观察的时刻, 所以
    t 之前至少要有 abs(min(x_offsets)) 个用来预测, 即 min_t = abs(min(x_offsets))
    t 之后至少要有 abs(max(y_offsets)) 个用来验证预测结果, 即 max_t = num_samples - abs(max(y_offsets))
    可以应用的 t 的长度 length = num_samples + min(x_offsets) - max(y_offsets)
    '''
    min_t = abs(min(x_offsets))
    max_t = num_samples - abs(max(y_offsets))
    for t in range(min_t, max_t):
        # 注意这里 x_offsets 和 y_offsets 都还是数组, x_t 即历史时间段, y_t即预测时间段
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        # 每次划分出一组数据, 添加到列表中
        x.append(x_t)
        y.append(y_t)
    # x: (length, len(x_offset), num_nodes, 1 + 1 + 7)
    x = np.stack(x, axis=0)
    # x: (length, len(y_offset), num_nodes, 1 + 1 + 7)
    y = np.stack(y, axis=0)
    return x, y


# 这里单纯为了解决 args 同名会有警告的问题, 不影响
def main(args):
    print("Generating training data")
    load_data(args)
    print("Generating training data completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 生成的 npz 文件所在目录
    parser.add_argument("--output_dir", type=str, default="dataset/METR-LA", help="Output directory.")
    # 交通文件所在目录
    parser.add_argument("--traffic_filename", type=str, default="dataset/METR-LA/METR-LA.h5", help="Raw traffic readings")
    # 是否添加时间信息
    parser.add_argument("--add_time_in_day", type=bool, default=True, help="Add time in day")
    parser.add_argument("--add_day_in_week", type=bool, default=True, help="Add day in week")
    main_args = parser.parse_args()
    main(main_args)
