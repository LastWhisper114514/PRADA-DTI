# import os
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# from sklearn.preprocessing import LabelEncoder
# import pandas as pd
# import torch.utils.data as data
# import torch
# import numpy as np
# from functools import partial
# from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
# from drug_ban.utils import integer_label_protein
# import dgl
#
# from datasets.utils.continual_dataset import ContinualDataset
#
# from typing import Tuple, Type
#
# base_path = './data/biosnap/'  # 确保这里是正确的数据路径
#
# class DTIDataset(ContinualDataset):
#     NAME = 'dtidataset'
#     N_CLASSES_PER_TASK = 2
#     N_TASKS = 5
#     INDIM = (1200,)
#     MAX_N_SAMPLES_PER_TASK = 300
#
#     def __init__(self, list_IDs, df, max_drug_nodes=290):
#         self.list_IDs = list_IDs
#         self.df = df
#         self.max_drug_nodes = max_drug_nodes
#
#         self.atom_featurizer = CanonicalAtomFeaturizer()
#         self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
#         self.fc = partial(smiles_to_bigraph, add_self_loop=True)
#
#     def __len__(self):
#         return len(self.list_IDs)
#
#     def __getitem__(self, idx):
#         index = self.list_IDs[idx]
#         # print('global index:',index,'index:',idx)
#         v_d = self.df.iloc[index]['SMILES']
#         v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
#         actual_node_feats = v_d.ndata.pop('h')
#         num_actual_nodes = actual_node_feats.shape[0]
#         num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
#         virtual_node_bit = torch.zeros([num_actual_nodes, 1])
#         actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
#         v_d.ndata['h'] = actual_node_feats
#         virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
#         v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
#         v_d = v_d.add_self_loop()
#
#         v_p = self.df.iloc[index]['Protein']
#         v_p = integer_label_protein(v_p)
#         y = self.df.iloc[index]["Y"]
#         # y = torch.Tensor([y])
#         return v_d, v_p, y, idx
# def graph_collate_func(x):
#     d, p, y, idx= zip(*x)
#     d = dgl.batch(d)
#     return d, torch.tensor(np.array(p)), torch.tensor(y),idx
#
# class BiosnapDataset(ContinualDataset):
#     """
#     This dataset contains protein sequ ences, SMILES strings, and the interaction labels.
#     """
#     NAME = 'biosnap'
#     N_CLASSES_PER_TASK = 2  # 二分类问题：相互作用与否
#     N_TASKS = 5 # 假设只有一个任务
#     MAX_N_SAMPLES_PER_TASK = 5400 # 数据集的最大样本数
#     INDIM = (1200,)  # 这是你之前提到的参数
#
#     def __init__(self, args):
#         self.args = args
#         self.label_encoder = LabelEncoder()  # 用于编码标签
#         self.data = self.load_data(args)
#         self.i = 0
#
#         self.setup_loaders()  # 初始化数据加载器
#     def load_data(self, args):
#         import pandas as pd
#
#         df = pd.read_csv('./data/biosnap/fulldata.csv')
#         df['drug_cluster'] = df['drug_cluster'].astype(str)
#         df['Protein'] = df['Protein'].astype(str)
#         df['SMILES'] = df['SMILES'].astype(str)
#         df['Y'] = df['Y'].astype(int)
#
#         # 直接将 target_domain 作为任务 id
#         df['task_id'] = df['target_domain'].astype(int)
#
#         return df
#
#     def setup_loaders(self):
#         self.train_loaders, self.test_loaders = [], []
#
#         all_task_ids = sorted(self.data['task_id'].unique())
#         for task_id in all_task_ids:
#             # 当前任务的数据
#             task_df = self.data[self.data['task_id'] == task_id]
#
#             # 划分训练集和测试集（8:2）
#             num_samples = len(task_df)
#             num_train = int(0.8 * num_samples)
#             indices = list(range(num_samples))
#             train_indices = indices[:num_train]
#             test_indices = indices[num_train:]
#
#             # 原始索引映射到 data 的行索引
#             df_indices = task_df.index.to_list()
#
#             train_dataset = DTIDataset([df_indices[i] for i in train_indices], self.data)
#             test_dataset = DTIDataset([df_indices[i] for i in test_indices], self.data)
#
#             train_loader = DataLoader(train_dataset,
#                                       batch_size=self.args.batch_size,
#                                       shuffle=True,
#                                       num_workers=self.args.num_workers,
#                                       collate_fn=graph_collate_func,
#                                       drop_last = True)
#             test_loader = DataLoader(test_dataset,
#                                      batch_size=self.args.batch_size,
#                                      shuffle=False,
#                                      num_workers=self.args.num_workers,
#                                      collate_fn=graph_collate_func,
#                                      drop_last = True)
#
#             self.train_loaders.append(train_loader)
#             self.test_loaders.append(test_loader)
#
#         #self.N_TASKS = len(self.train_loaders)  # 动态更新任务数
#
#     def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
#         current_train = self.train_loaders[self.i]
#         current_test = self.test_loaders[self.i]
#
#         next_train, next_test = None, None
#         if self.i + 1 < self.N_TASKS:
#             next_train = self.train_loaders[self.i + 1]
#             next_test = self.test_loaders[self.i + 1]
#
#         return current_train, current_test, next_train, next_test
#
#     def encode_protein_sequence(self, sequence: str):
#         amino_acid_map = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
#                           'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
#         sequence_encoded = [amino_acid_map.get(aa, 20) for aa in sequence]
#         return torch.tensor(sequence_encoded, dtype=torch.long)
#
#     def encode_smiles(self, smiles: str):
#         return torch.tensor([ord(c) for c in smiles], dtype=torch.long)
#
#     def __len__(self):
#         return len(self.data)
#
#     @staticmethod
#     def get_loss():
#         return torch.nn.BCEWithLogitsLoss()  # 适用于二分类任务
#
#     @staticmethod
#     def get_transform():
#         return None
#     @staticmethod
#     def get_batch_size() -> int:
#         return 32
#
#     @staticmethod
#     def get_minibatch_size() -> int:
#         return BiosnapDataset.get_batch_size()
#
#
