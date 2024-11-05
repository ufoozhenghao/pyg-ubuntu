import os
import matplotlib.pyplot as plt
import numpy as np

# search_data 旨在从时间序列数据中提取一组索引对，用于预测模型的输入。它通过指定的历史依赖关系（num_of_depend）和其他参数来确定这些索引对。
def search_data(sequence_length, num_of_depend, label_start_idx, num_for_predict, units, points_per_hour):
    """
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int, 依赖的历史时间段数。例如，如果你想依赖过去的 3 小时数据，那么 num_of_depend 就是 3
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, 每个样本要预测的点数。例如，如果你要预测未来 12 个时间点的数据，那么 num_for_predict 就是 12
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data, default 12
    Returns
    ----------
    list[(start_idx, end_idx)]
    """

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None
    # 返回索引对列表的逆序版本。这样可以确保最早的依赖在列表的最前面。
    # x_idx[::-1] 等价于 x_idx[None:None:-1]，表示从列表的末尾开始，以步长 -1 逐步向前，直到列表的开头。
    return x_idx[::-1]

# get_sample_indices 旨在从时间序列数据中提取样本，用于预测模型的输入。它根据指定的周、天和小时的依赖关系，返回相应的样本和预测目标。
def get_sample_indices(data_sequence, num_of_depend, label_start_idx, num_for_predict, points_per_hour=1):
    """
    Parameters
    ----------
    data_sequence: np.ndarray shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_depend: int，依赖的小时数。例如，如果你想依赖过去的 3 小时数据，那么 num_of_hours 就是 3
    label_start_idx: int, 预测目标的第一个索引。即你要预测的目标数据在时间序列中的起始位置
    num_for_predict: int, 每个样本要预测的点数。例如，如果你要预测未来 12 个时间点的数据，那么num_for_predict 就是 12。
    points_per_hour: int, default 1, 每小时的数据点数
    Returns
    ----------
    week_sample: np.ndarray shape is (num_of_weeks * points_per_hour, num_of_vertices, num_of_features)
    day_sample: np.ndarray shape is (num_of_days * points_per_hour,  num_of_vertices, num_of_features)
    hour_sample: np.ndarray   shape is (num_of_hours * points_per_hour, num_of_vertices, num_of_features)
    target: np.ndarray shape is (num_for_predict, num_of_vertices, num_of_features)
    """
    hour_sample = None
    # 检查预测的结束索引是否超出了数据序列的长度。如果超出，返回 None。
    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return hour_sample, None

    if num_of_depend > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_depend, label_start_idx, num_for_predict, 1, points_per_hour)
        if not hour_indices:
            return None, None
        hour_sample = np.concatenate([data_sequence[i: j] for i, j in hour_indices], axis=0)
        print('hour_indices：', hour_indices)
    
    # 超过 10 小时的数据依赖是不合理的或不需要的
    if num_of_depend > 10:
        return 1
    # 提取预测目标
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return hour_sample, target

# 从图信号矩阵文件中读取数据，并生成用于训练、验证和测试的数据集。
def read_and_generate_dataset(graph_signal_matrix_filename, num_of_depend, num_for_predict, points_per_hour=1):
    """
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_depend: int， 依赖的小时数。例如，如果你想依赖过去的 3 小时数据，那么 num_of_hours 就是 3
    num_for_predict: int
    points_per_hour: int, default 1, depends on data
    Returns
    ----------
    feature: np.ndarray, shape is (num_of_samples, num_of_depend * points_per_hour, num_of_vertices, num_of_features)
    target: np.ndarray, shape is (num_of_samples, num_of_vertices, num_for_predict)
    """
    # Read original data
    data_seq = np.load(graph_signal_matrix_filename)['data']  # (sequence_length, num_of_vertices, num_of_features) (16992, 307, 3)

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_depend, idx, num_for_predict,points_per_hour)
        if (sample[0] is None) and (sample[1] is None):
            continue

        hour_sample, target = sample
        print(target.shape)  # hour_sample and target 
        sample = []  # [(hour_sample),target,time_sample]

        if num_of_depend > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(hour_sample)

        # [:, :, 0, :] 第二个轴选择0个元素==>>切片操作后，形状将从 (1, N, F, T) 变为 (1, N, T)
        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        sample.append(target)
        # 创建一个包含当前索引的时间样本，并添加到 sample 列表中
        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)
        all_samples.append(sample)  #sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

    # 按比例将 all_samples 划分为训练集（60%）、验证集（20%）和测试集（20%）。使用 zip 和 np.concatenate 将样本拼接成一个数组。
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    # zip 函数用于将多个可迭代对象打包成一个元组的迭代器。*subset_samples 表示对 subset_samples 进行解包操作。
    training_set = [np.concatenate(i, axis=0) for i in
                    zip(*all_samples[:split_line1])]  #[(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]

    """
    zipped_samples = zip(*subset_samples)
    结果是一个迭代器，内容如下：
    [
      (array1_1, array1_2, ..., array1_10),
      (array2_1, array2_2, ..., array2_10),
      (array3_1, array3_2, ..., array3_10),
      (array4_1, array4_2, ..., array4_10),
      (array5_1, array5_2, ..., array5_10)
    ]

    np.concatenate 后 training_set 中每个数组的形状：(B * split_line1, N, F, T)
    """

    return training_set, validation_set, testing_set

graph_signal_matrix_filename = 'data/62/62_quarter_single_dataset.npz'
data = np.load(graph_signal_matrix_filename)

# 查看文件中的键
print("Keys in the .npz file:", data.files)
# 查看每个数组的形状和数据类型
for key in data.files:
    array = data[key]
    print(f"Array name: {key}")
    print(f"Shape: {array.shape}") # Shape: (100, 186, 2)
    print(f"Data type: {array.dtype}")
    print(array[:5])
    print('====================')


node_temperature = data['data'][:, 61, 0]  # (100, 1，,1)
print('节点序列长度： ', node_temperature.shape)
print('节点第一组数据前5： ', node_temperature[:5])
fig_temperature = plt.figure(figsize=(15, 5))
plt.title('node_temperature')
plt.xlabel('hours')
plt.ylabel('temperature')
plt.plot(np.arange(len(node_temperature)), node_temperature, linestyle='-')
fig_temperature.autofmt_xdate(rotation=45)
plt.show()

# graph_signal_matrix_filename = 'dataset.npz'
data_seq = np.load(graph_signal_matrix_filename)['data']
print('data_seq.shape[0]: ', data_seq.shape[0])
num_of_depend = 5       # 依赖的历史时间段数
num_for_predict = 3     # 每个样本要预测的点数 
units = 1
points_per_hour = 1
label_start_idx = 6     # 从第 6 小时开始预测


training_set, validation_set, testing_set = read_and_generate_dataset(graph_signal_matrix_filename, num_of_depend, num_for_predict, points_per_hour);

print("training_set 的形状:", [x.shape for x in training_set])
print(training_set[0][0].shape)
print([_x.shape for _x in training_set[:-2]])

def normalization(train, val, test):
    """
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std 包含均值和标准差的字典
    train_norm, val_norm,
    test_norm: np.ndarray, shape is the same as original 归一化后的训练数据
    """

    # 判断train 和 val的(N, F, T)形状；即在节点数量、特征数量和时间步长上的形状是相同的 train.shape[1:]=(307, 3, 12)
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    # mean：沿轴 (0, 1, 3) 计算均值，即对批量、节点和时间步进行平均，保留特征维度。
    # std：沿轴 (0, 1, 3) 计算标准差，同样保留特征维度。
    mean = train.mean(axis=(0, 1, 3), keepdims=True)  # train (B,N,F,T')
    std = train.std(axis=(0, 1, 3), keepdims=True)
    print('mean.shape:', mean.shape)
    print('std.shape:', std.shape)

    # 零均值和单位标准差归一化函数
    def normalize(x):
        return (x - mean) / std

    # 归一化训练集、验证集和测试集
    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm

# 从 training_set 中提取从第一个元素到倒数第三个元素（包括倒数第三个元素）的所有元素==>>在这里其实就是第一个元素(55, 186, 2, 15)，然后使用 np.concatenate 函数沿着指定的轴（这里是 axis=-1，即T'轴）将这些元素连接起来。
# training_set 的形状: [(55, 186, 2, 15), (55, 186, 3), (55, 1)]

train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T) (55, 186, 2, 15)
val_x = np.concatenate(validation_set[:-2], axis=-1)
test_x = np.concatenate(testing_set[:-2], axis=-1)

_train_x = training_set[0]
assert np.array_equal(_train_x, train_x), "Contents are not equal"

train_target = training_set[-2]  # (B,N,T) (55, 186, 3)
val_target = validation_set[-2]
test_target = testing_set[-2]

train_timestamp = training_set[-1]  # (B,1) (55, 1)
val_timestamp = validation_set[-1]
test_timestamp = testing_set[-1]

(stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)

print('train_x_norm.shape:', train_x_norm.shape)

all_data = {'train': {'x': train_x_norm, 'target': train_target, 'timestamp': train_timestamp},
            'val': {'x': val_x_norm, 'target': val_target, 'timestamp': val_timestamp},
            'test': {'x': test_x_norm, 'target': test_target, 'timestamp': test_timestamp},
            'stats': {'_mean': stats['_mean'], '_std': stats['_std']}}

print('train x:', all_data['train']['x'].shape)
print('train target:', all_data['train']['target'].shape)
print('train timestamp:', all_data['train']['timestamp'].shape)
print()
print('val x:', all_data['val']['x'].shape)
print('val target:', all_data['val']['target'].shape)
print('val timestamp:', all_data['val']['timestamp'].shape)
print()
print('test x:', all_data['test']['x'].shape)
print('test target:', all_data['test']['target'].shape)
print('test timestamp:', all_data['test']['timestamp'].shape)
print()
print('train data _mean :', all_data['stats']['_mean'].shape, '\n', all_data['stats']['_mean'])
print('train data _std :', all_data['stats']['_std'].shape, '\n', all_data['stats']['_std'])

# 输出
file = os.path.basename(graph_signal_matrix_filename).split('.')[0]  # 获取文件名

sensor_number = '62'
model_scale='quarter' # 模型比例
layer = 'single' # 混凝土层数 单层
filename = f"{sensor_number}_{model_scale}_{layer}_dataset_astcgn"
print('save file:', filename)
"""
使用 np.savez_compressed 函数将数据保存到一个压缩的 .npz 文件中。
filename 是保存文件的路径。
传递多个关键字参数，将不同的数据保存到 .npz 文件中。每个关键字参数对应一个数组，具体包括：
train_x：训练数据的输入特征。
train_target：训练数据的目标值。
train_timestamp：训练数据的时间戳。
val_x：验证数据的输入特征。
val_target：验证数据的目标值。
val_timestamp：验证数据的时间戳。
test_x：测试数据的输入特征。
test_target：测试数据的目标值。
test_timestamp：测试数据的时间戳。
mean：数据的均值（用于数据标准化）。
std：数据的标准差（用于数据标准化）。
"""
np.savez_compressed(filename,
                    train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                    train_timestamp=all_data['train']['timestamp'],
                    val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                    val_timestamp=all_data['val']['timestamp'],
                    test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                    test_timestamp=all_data['test']['timestamp'],
                    mean=all_data['stats']['_mean'], std=all_data['stats']['_std'])

