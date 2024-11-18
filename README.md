## 数据文件说明

62：混凝土中的温度节点数量（不包含进出水管内的温度）

62_id_filename.csv： 62个节点midas编号 从低到高排列

62_1.4_single.csv： 62个节点温度，模型为1/4单层混凝土

62_node_distance.csv: 62个节点之间的距离

---

62_quarter_single.csv: 62个节点，1/4模型，单层混凝土

62_quarter_single_dataset.json: `62_quarter_single.csv`通过`transfer2vertices.py`转换得到

62_quarter_single_dataset.npz: `62_quarter_single.csv`通过`transfer2vertices.py`转换得到

62_quarter_single_dataset_astcgn.npz: 通过`0_process_heat_data.py`转换得到

---

38_quarter_single_16t.csv： 38个节点，1/4模型，单层混凝土，水冷温度16

38_quarter_single_18t.csv： 38个节点，1/4模型，单层混凝土，水冷温度18

---
one_quarter_single: 62*3个单层混凝土节点温度（3中不同进水温度结合 186个节点）