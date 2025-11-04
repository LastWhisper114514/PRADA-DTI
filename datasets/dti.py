
from torch.utils.data import Dataset, DataLoader, random_split
import dgl

from datasets.utils.continual_dataset import ContinualDataset

from typing import Tuple, Type
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.utils.data
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import torch
import pandas as pd
from torch_geometric.data import Data
import pickle
import torch.utils.data
from copy import deepcopy
import numpy as np


def graph_collate_func(x):
    d, p, y, idx= zip(*x)
    d = dgl.batch(d)
    return d, torch.tensor(np.array(p)), torch.tensor(y),idx

class BiosnapDataset(ContinualDataset):
    """
    This dataset contains protein sequ ences, SMILES strings, and the interaction labels.
    """
    NAME = 'dti'
    N_CLASSES_PER_TASK = 2  # 二分类问题：相互作用与否
    MAX_N_SAMPLES_PER_TASK = 1000 # 数据集的最大样本数
    INDIM = (400,)  # 这是你之前提到的参数

    # ✅ 建立映射表
    DATASET_TASK_MAP = {
        'biosnap': 15,  # 6个任务（domain）
        'bindingdb': 25,  # BindingDB 通常划分 25 个任务
        'human': 8,  # Human DTI 常见 8 个任务
        'davis': 1,  # Davis 没有 domain 划分
        'kiba': 1,  # KIBA 同样只有 1 个任务
    }

    def __init__(self, args):
        self.args = args
        self.i = 0
        self.device = 'cuda'

        self.DATASET_NAME = args.dataset_name
        self.N_TASKS = self.DATASET_TASK_MAP.get(self.DATASET_NAME, 1)

        self.data = pd.read_csv(f'./data/{self.DATASET_NAME}/fulldata.csv')
        self.data['task_id'] = self.data['new_domain_id'].astype(int)

        # 加载缓存的图数据
        self.mols = torch.load(open(f'./data/{self.DATASET_NAME}/ligand_raw.pt', 'rb'))
        self.prots = torch.load(f'./data/{self.DATASET_NAME}/protein.pt')

        self.train_loaders, self.test_loaders = self.setup_loaders()

    def setup_loaders(self):
        train_loaders, test_loaders = [], []
        task_ids = sorted(self.data['new_domain_id'].unique())

        for task_id in task_ids:
            task_df = self.data[ self.data['new_domain_id'] == task_id]

            n = len(task_df)
            n_train = int(n * 0.8)
            indices = np.random.permutation(n)
            train_df = task_df.iloc[indices[:n_train]]
            test_df = task_df.iloc[indices[n_train:]]

            train_dataset = ProteinMoleculeDataset(train_df, self.mols, self.prots, device=self.device)
            test_dataset = ProteinMoleculeDataset(test_df, self.mols, self.prots, device=self.device)

            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                     drop_last=False,follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])

            test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False,
                                     drop_last=False, follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])


            train_loaders.append(train_loader)
            test_loaders.append(test_loader)

        return train_loaders, test_loaders

    def setup_loaders_weighted(self):
        train_loaders, test_loaders = [], []
        task_ids = sorted(self.data['new_domain_id'].unique())

        # 计算每个任务的样本数
        task_counts = self.data.groupby('new_domain_id').size().to_dict()

        for task_id in task_ids:
            task_df = self.data[self.data['new_domain_id'] == task_id]

            n = len(task_df)
            n_train = int(n * 0.8)
            indices = np.random.permutation(n)
            train_df = task_df.iloc[indices[:n_train]]
            test_df = task_df.iloc[indices[n_train:]]

            train_dataset = ProteinMoleculeDataset(train_df, self.mols, self.prots, device=self.device)
            test_dataset = ProteinMoleculeDataset(test_df, self.mols, self.prots, device=self.device)

            # === 平滑反比权重计算 ===
            # sqrt(1/N_task) 归一化
            weight_value = 1.0 / np.sqrt(task_counts[task_id])
            weights = torch.full((len(train_dataset),), weight_value, dtype=torch.float)

            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(train_dataset),  # 采样次数等于数据集大小（过采样）
                replacement=True
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                sampler=sampler,
                drop_last=False,
                follow_batch=['mol_x', 'clique_x', 'prot_node_aa']
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                drop_last=False,
                follow_batch=['mol_x', 'clique_x', 'prot_node_aa']
            )

            train_loaders.append(train_loader)
            test_loaders.append(test_loader)

        return train_loaders, test_loaders

    def get_data_loaders(self):
        current_train = self.train_loaders[self.i]
        current_test = self.test_loaders[self.i]
        next_train, next_test = None, None

        if self.i + 1 < len(self.train_loaders):
            next_train = self.train_loaders[self.i + 1]
            next_test = self.test_loaders[self.i + 1]

        return current_train, current_test, next_train, next_test

    def encode_protein_sequence(self, sequence: str):
        amino_acid_map = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                          'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
        sequence_encoded = [amino_acid_map.get(aa, 20) for aa in sequence]
        return torch.tensor(sequence_encoded, dtype=torch.long)

    def encode_smiles(self, smiles: str):
        return torch.tensor([ord(c) for c in smiles], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_loss():
        return torch.nn.BCEWithLogitsLoss()  # 适用于二分类任务

    @staticmethod
    def get_transform():
        return None
    @staticmethod
    def get_batch_size() -> int:
        return 32

    @staticmethod
    def get_minibatch_size() -> int:
        return BiosnapDataset.get_batch_size()

class ProteinMoleculeDataset(Dataset):

    def __init__(self, sequence_data, mol_obj, prot_obj, device='cpu', cache_transform=True):
        super(ProteinMoleculeDataset, self).__init__()

        if isinstance(sequence_data, pd.core.frame.DataFrame):
            self.pairs = sequence_data
        elif isinstance(sequence_data, str):
            self.pairs = pd.read_csv(sequence_data)
        else:
            raise Exception("provide dataframe object or csv path")

        ## MOLECULES
        if isinstance(mol_obj, dict):
            self.mols = mol_obj
        elif isinstance(mol_obj, str):
            with open(mol_obj, 'rb') as f:
                self.mols = pickle.load(f)
        else:
            raise Exception("provide dict mol object or pickle path")

        ## PROTEINS
        if isinstance(prot_obj, dict):
            self.prots = prot_obj
        elif isinstance(prot_obj, str):
            self.prots = torch.load(prot_obj)
        else:
            raise Exception("provide dict mol object or pickle path")

        self.device = device
        self.cache_transform = cache_transform

        if self.cache_transform:
            for _, v in self.mols.items():
                v['atom_idx'] = v['atom_idx'].long().view(-1, 1)
                v['atom_feature'] = v['atom_feature'].float()
                adj = v['bond_feature'].long()
                mol_edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                v['atom_edge_index'] = mol_edge_index
                v['atom_edge_attr'] = adj[mol_edge_index[0], mol_edge_index[1]].long()
                v['atom_num_nodes'] = v['atom_idx'].shape[0]

                ## Clique
                v['x_clique'] = v['x_clique'].long().view(-1, 1)
                v['clique_num_nodes'] = v['x_clique'].shape[0]
                v['tree_edge_index'] = v['tree_edge_index'].long()
                v['atom2clique_index'] = v['atom2clique_index'].long()

            for _, v in self.prots.items():
                v['seq_feat'] = v['seq_feat'].float()
                v['token_representation'] = v['token_representation'].float()
                v['num_nodes'] = len(v['seq'])
                v['node_pos'] = torch.arange(len(v['seq'])).reshape(-1, 1)
                v['edge_weight'] = v['edge_weight'].float()

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return self.__len__()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Extract data
        mol_key = self.pairs.iloc[idx]['Ligand']
        prot_key = self.pairs.iloc[idx]['Protein']
        try:
            reg_y = self.pairs.loc[idx, 'regression_label']
            reg_y = torch.tensor(reg_y).float()
        except KeyError:
            reg_y = None

        try:
            cls_y = self.pairs.iloc[idx]['classification_label']
            cls_y = torch.tensor(cls_y).float()
        except KeyError:
            cls_y = None

        try:
            mcls_y = self.pairs.loc[idx, 'multiclass_label']
            mcls_y = torch.tensor(mcls_y + 1).float()
        except KeyError:
            mcls_y = None

        mol = self.mols[mol_key]
        prot = self.prots[prot_key]

        ## PROT
        if self.cache_transform:
            ## atom
            mol_x = mol['atom_idx']
            mol_x_feat = mol['atom_feature']
            mol_edge_index = mol['atom_edge_index']
            mol_edge_attr = mol['atom_edge_attr']
            mol_num_nodes = mol['atom_num_nodes']

            ## Clique
            mol_x_clique = mol['x_clique']
            clique_num_nodes = mol['clique_num_nodes']
            clique_edge_index = mol['tree_edge_index']
            atom2clique_index = mol['atom2clique_index']
            ## Prot
            prot_seq = prot['seq']
            prot_node_aa = prot['seq_feat']
            prot_node_evo = prot['token_representation']
            prot_num_nodes = prot['num_nodes']
            prot_node_pos = prot['node_pos']
            prot_edge_index = prot['edge_index']
            prot_edge_weight = prot['edge_weight']
        else:
            # MOL
            mol_x = mol['atom_idx'].long().view(-1, 1)
            mol_x_feat = mol['atom_feature'].float()
            adj = mol['bond_feature'].long()
            mol_edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            mol_edge_attr = adj[mol_edge_index[0], mol_edge_index[1]].long()
            mol_num_nodes = mol_x.shape[0]

            ## Clique
            mol_x_clique = mol['x_clique'].long().view(-1, 1)
            clique_num_nodes = mol_x_clique.shape[0]
            clique_edge_index = mol['tree_edge_index'].long()
            atom2clique_index = mol['atom2clique_index'].long()

            prot_seq = prot['seq']
            prot_node_aa = prot['seq_feat'].float()
            prot_node_evo = prot['token_representation'].float()
            prot_num_nodes = len(prot['seq'])
            prot_node_pos = torch.arange(len(prot['seq'])).reshape(-1, 1)
            prot_edge_index = prot['edge_index']
            prot_edge_weight = prot['edge_weight'].float()

        out = MultiGraphData(
            ## MOLECULE
            mol_x=mol_x, mol_x_feat=mol_x_feat, mol_edge_index=mol_edge_index,
            mol_edge_attr=mol_edge_attr, mol_num_nodes=mol_num_nodes,
            clique_x=mol_x_clique, clique_edge_index=clique_edge_index, atom2clique_index=atom2clique_index,
            clique_num_nodes=clique_num_nodes,
            ## PROTEIN
            prot_node_aa=prot_node_aa, prot_node_evo=prot_node_evo,
            prot_node_pos=prot_node_pos, prot_seq=prot_seq,
            prot_edge_index=prot_edge_index, prot_edge_weight=prot_edge_weight,
            prot_num_nodes=prot_num_nodes,
            ## Y output
            reg_y=reg_y, cls_y=cls_y, mcls_y=mcls_y,
            ## keys
            mol_key=mol_key, prot_key=prot_key
        )
        out.sample_idx = torch.tensor(idx, dtype=torch.long)

        return out


def maybe_num_nodes(index, num_nodes=None):
    # NOTE(WMF): I find out a problem here,
    # index.max().item() -> int
    # num_nodes -> tensor
    # need type conversion.
    # return index.max().item() + 1 if num_nodes is None else num_nodes
    return index.max().item() + 1 if num_nodes is None else int(num_nodes)


def get_self_loop_attr(edge_index, edge_attr, num_nodes):
    r"""Returns the edge features or weights of self-loops
    :math:`(i, i)` of every node :math:`i \in \mathcal{V}` in the
    graph given by :attr:`edge_index`. Edge features of missing self-loops not
    present in :attr:`edge_index` will be filled with zeros. If
    :attr:`edge_attr` is not given, it will be the vector of ones.

    .. note::
        This operation is analogous to getting the diagonal elements of the
        dense adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_weight = torch.tensor([0.2, 0.3, 0.5])
        >>> get_self_loop_attr(edge_index, edge_weight)
        tensor([0.5000, 0.0000])

        >>> get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
        tensor([0.5000, 0.0000, 0.0000, 0.0000])
    """
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[0][loop_mask]

    if edge_attr is not None:
        loop_attr = edge_attr[loop_mask]
    else:  # A vector of ones:
        loop_attr = torch.ones_like(loop_index, dtype=torch.float)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    full_loop_attr = loop_attr.new_zeros((num_nodes,) + loop_attr.size()[1:])
    full_loop_attr[loop_index] = loop_attr

    return full_loop_attr


class MultiGraphData(Data):
    def __inc__(self, key, item, *args):
        if key == 'mol_edge_index':
            return self.mol_x.size(0)
        elif key == 'clique_edge_index':
            return self.clique_x.size(0)
        elif key == 'atom2clique_index':
            return torch.tensor([[self.mol_x.size(0)], [self.clique_x.size(0)]])
        elif key == 'prot_edge_index':
            return self.prot_node_aa.size(0)
        elif key == 'prot_struc_edge_index':
            return self.prot_node_aa.size(0)
        elif key == 'm2p_edge_index':
            return torch.tensor([[self.mol_x.size(0)], [self.prot_node_aa.size(0)]])
        # elif key == 'edge_index_p2m':
        #     return torch.tensor([[self.prot_node_s.size(0)],[self.mol_x.size(0)]])
        else:
            return super(MultiGraphData, self).__inc__(key, item, *args)

