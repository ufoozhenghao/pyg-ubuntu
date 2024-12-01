# 这是一个示例 Python 脚本。
import numpy as np


# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。
    graph_signal_matrix_file_array = ['./data/38/38_quarter_single_16t.npz','./data/38/38_quarter_single_18t.npz','./data/38/38_quarter_single_20t.npz']
    # for i in range(len(graph_signal_matrix_file_array)):
    data_seq = np.load(graph_signal_matrix_file_array[0])['data']   # (500,38,2)
    print(data_seq[0])

def test():
    import re

    # 文件路径数组
    graph_signal_matrix_file_array = [
        './data/38/38_quarter_single_16t.npz',
        './data/38/38_quarter_single_18t.npz',
        './data/38/38_quarter_single_20t.npz'
    ]

    # 提取数字的正则表达式模式
    pattern = re.compile(r'_(\d+)t\.npz')

    # 提取数字并存储到列表中
    numbers = []
    for file_path in graph_signal_matrix_file_array:
        match = pattern.search(file_path)
        if match:
            numbers.append(match.group(1))

    print(numbers)

def tt():
    graph_signal_matrix_filename = './data/38/38_quarter_single_16_18_20_dataset_astcgn.npz'
    file_data = np.load(graph_signal_matrix_filename)
    print(file_data)


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # print_hi('PyCharm')
    tt()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
