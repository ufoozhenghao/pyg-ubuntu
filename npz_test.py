"""
@File    : npz_test.py
@Author  : zhenghaoxuan
@Date    : 2024/10/16 10:42
@Feature : Enter feature description here
"""

import numpy as np

graph_signal_matrix_filename = 'dataset.npz'
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
    print()

import matplotlib.pyplot as plt

flow = data['data'][:, 62, 0]  # (100, 1，,1)
print(flow.shape)
print(flow[:5])
fig_flow = plt.figure(figsize=(15, 5))
plt.title('traffic Flow')
plt.xlabel('minute')
plt.ylabel('flow')
plt.plot(np.arange(len(flow)), flow, linestyle='-')
fig_flow.autofmt_xdate(rotation=45)
plt.show()


