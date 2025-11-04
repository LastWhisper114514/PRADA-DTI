import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones.utils.continual_backbone import FwdContinualBackbone
from backbones.utils.modules import xavier
import ipdb
import os
# 导入 DrugBAN
from backbones.PSICHIC_.psichic_net import net as PSICHICbackbone # 替换为实际的 DrugBAN 导入路径

import os
import torch
from backbones.PSICHIC_.utils.utils import compute_pna_degrees  # 你已有的函数


class PSICHIC(FwdContinualBackbone):
    NAME = 'psichic'

    def __init__(self, indim, hiddim, outdim, args, config):
        super(PSICHIC, self).__init__()

        self.task_type = config['task_type']  # "regression", "classification", "multiclassification"
        self.softmax = torch.nn.Softmax(dim=1)

        # 存储中间层输出
        self.fwd_outputs = {}

        # 实例化 backbone
        self.net = PSICHICbackbone(
            mol_deg=config['mol_deg'],
            prot_deg=config['prot_dig'],
            mol_in_channels=config['params']['mol_in_channels'],
            prot_in_channels=config['params']['prot_in_channels'],
            prot_evo_channels=config['params']['prot_evo_channels'],
            hidden_channels=config['params']['hidden_channels'],
            pre_layers=config['params']['pre_layers'],
            post_layers=config['params']['post_layers'],
            aggregators=config['params']['aggregators'],
            scalers=config['params']['scalers'],
            total_layer=config['params']['total_layer'],
            K=config['params']['K'],
            t=config['params']['t'],
            heads=config['params']['heads'],
            dropout=config['params']['dropout'],
            dropout_attn_score=config['params']['dropout_attn_score'],
            drop_atom=config['params']['drop_atom'],
            drop_residue=config['params']['drop_residue'],
            dropout_cluster_edge=config['params']['dropout_cluster_edge'],
            gaussian_noise=config['params']['gaussian_noise'],
            regression_head=(self.task_type == "regression"),
            classification_head=(self.task_type == "classification"),
            multiclassification_head=(config['num_classes'] if self.task_type == "multiclassification" else 0),
            device=config['device'],
            use_prompt = getattr(args, 'use_prompt', False)  # ✅ 新增
        )

        # 注册钩子（如有需要，你也可以跳过）
        # def get_feature(name):
        #     def hook(model, input, output):
        #         self.fwd_outputs[name] = output
        #     return hook
        # self.net.mol_convs[0].register_forward_hook(get_feature("mol_conv0"))

    def extract_query(self, inputs):
        """
            Extracts the query vector used for prompt selection in L2P.
            Typically taken from cluster_x (before prompt injection).
            """
        batch = inputs
        with torch.no_grad():
            query = self.net(
                mol_x=batch['mol_x'],
                mol_x_feat=batch['mol_x_feat'],
                bond_x=batch['mol_edge_attr'],
                atom_edge_index=batch['mol_edge_index'],
                clique_x=batch['clique_x'],
                clique_edge_index=batch['clique_edge_index'],
                atom2clique_index=batch['atom2clique_index'],
                residue_x=batch['prot_node_aa'],
                residue_evo_x=batch['prot_node_evo'],
                residue_edge_index=batch['prot_edge_index'],
                residue_edge_weight=batch['prot_edge_weight'],
                mol_x_batch=batch['mol_x_batch'],
                prot_node_aa_batch=batch['prot_node_aa_batch'],
                clique_x_batch=batch['clique_x_batch'], returnt="query")
            return query  # [B, D]


    def forward(self, x, returnt='logits', prompt=None):
        batch = x  # x 是一个 dict，已经是整理好的 batch

        reg_pred, cls_pred, mcls_pred, spectral_loss, ortho_loss, cluster_loss, attention_dict  = self.net(
            mol_x=batch['mol_x'],
            mol_x_feat=batch['mol_x_feat'],
            bond_x=batch['mol_edge_attr'],
            atom_edge_index=batch['mol_edge_index'],
            clique_x=batch['clique_x'],
            clique_edge_index=batch['clique_edge_index'],
            atom2clique_index=batch['atom2clique_index'],
            residue_x=batch['prot_node_aa'],
            residue_evo_x=batch['prot_node_evo'],
            residue_edge_index=batch['prot_edge_index'],
            residue_edge_weight=batch['prot_edge_weight'],
            mol_x_batch=batch['mol_x_batch'],
            prot_node_aa_batch=batch['prot_node_aa_batch'],
            clique_x_batch=batch['clique_x_batch'],
            prompt=prompt
        )

        # 根据任务类型输出
        if returnt == 'features':
            raise NotImplementedError("PSICHIC目前未返回中间特征，如需支持请添加 attention_dict['interaction_fingerprint']")
        elif returnt == 'logits':
            if self.task_type == 'regression':
                return reg_pred
            elif self.task_type == 'classification':
                return cls_pred, attention_dict['interaction_fingerprint']
            elif self.task_type == 'multiclassification':
                return mcls_pred
        elif returnt == 'prob':
            if self.task_type == 'classification':
                return self.softmax(cls_pred)
            elif self.task_type == 'multiclassification':
                return self.softmax(mcls_pred)
            else:
                raise ValueError("回归任务不能输出概率")
        elif returnt == 'all':
            if self.task_type == 'regression':
                return reg_pred, reg_pred, None
            elif self.task_type == 'classification':
                prob = self.softmax(cls_pred)
                return prob, attention_dict['interaction_fingerprint'], cls_pred
            elif self.task_type == 'multiclassification':
                prob = self.softmax(mcls_pred)
                return mcls_pred, prob, None

    def reset_parameters(self) -> None:
        """
        调用 psichic backbone 自定义的 reset_parameters() 方法。
        """
        self.net.reset_parameters()

