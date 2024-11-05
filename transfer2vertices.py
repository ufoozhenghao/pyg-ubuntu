"""
@File    : transfer2vertices.py
@Author  : zhenghaoxuan
@Date    : 2024/10/16 10:34
@Feature : 将 one_quarter_single.csv 转换为 dataset.json 和 dataset.npz
"""

import csv
import json
import numpy as np

def convert_csv_to_json(input_file, output_file, output_npz):
    data = []
    group_attributes = [10, 12, 14, 16, 18, 20]  # 每组的隐藏属性

    with open(input_file, newline='', encoding='utf-8-sig') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            row_data = []
            for i in range(6):
                group_data = row[i*62:(i+1)*62]
                for value in group_data:
                    # 去除空格并转换为浮点数
                    cleaned_value = float(value.strip())
                    row_data.append([cleaned_value, group_attributes[i]])
            data.append(row_data)

    output = {"data": data}

    with open(output_file, 'w') as jsonfile:
        json.dump(output, jsonfile, indent=4)

    # 转换为 NumPy 数组
    np_data = np.array(data, dtype=np.float64)
    # 保存为 .npz 文件
    np.savez(output_npz, data=np_data)

if __name__ == '__main__':
    convert_csv_to_json('./62/62_quarter_single.csv', './62/62_quarter_single_dataset.json', './62/62_quarter_single_dataset.npz')
    print('done')