"""
@File    : transfer2vertices.py
@Author  : zhenghaoxuan
@Date    : 2024/10/16 10:34
@Feature : 将 节点温度数据.csv 转换为 dataset.json 和 dataset.npz
"""

import csv
import json
import numpy as np


def convert_csv_to_json(temperature, input_file, output_file, output_npz):
    data = []

    with open(input_file, newline='', encoding='utf-8-sig') as csvfile:
        csvreader = csv.reader(csvfile)
        all_data = list(csvreader)
        # 处理数据
        for row in all_data:
            processed_row = [[float(value), temperature] for value in row]
            data.append(processed_row)

    output = {"data": data}
    # print(output)
    with open(output_file, 'w') as jsonfile:
        json.dump(output, jsonfile, indent=4)

    # # 转换为 NumPy 数组
    np_data = np.array(data, dtype=np.float64)
    # 保存为 .npz 文件
    np.savez(output_npz, data=np_data)

if __name__ == '__main__':
    water_temperature = 24
    convert_csv_to_json(water_temperature,'../data/38/38_quarter_single_24t.csv',
                        '../data/38/38_quarter_single_24t.json',
                        '../data/38/38_quarter_single_24t.npz')
    print('done')