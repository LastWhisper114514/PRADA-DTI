import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones.utils.continual_backbone import FwdContinualBackbone
from backbones.utils.modules import xavier
import ipdb

# 导入 DrugBAN
from backbones.drug_ban.models import DrugBANbackbone  # 替换为实际的 DrugBAN 导入路径


class DrugBAN(FwdContinualBackbone):
    NAME = 'drugban'

    def __init__(self, indim, hiddim, outdim, args, config):
        super(DrugBAN, self).__init__()

        self.softmax = torch.nn.Softmax(dim=1)

        # 存储中间层输出
        self.fwd_outputs = {}

        # 使用提供的 config 初始化 DrugBAN
        self.net = DrugBANbackbone(config=config)  # 假设 DrugBAN 类接收 config 参数

        # 如果需要，可以注册钩子来捕获中间层的输出
        def get_feature(name):
            def hook(model, input, output):
                self.fwd_outputs[name] = output

            return hook

        # 注册钩子以提取 DrugBAN 的中间层（如需要）
        # 注意：根据您的 DrugBAN 结构调整钩子的位置
        # 例如，您可以将 `DrugBAN` 的某些层连接到钩子以捕获中间输出。
        # 注册钩子以提取 DrugBAN 的中间层
        self.net.drug_extractor.gnn.register_forward_hook(get_feature('drug_features'))

        # 注册 ProteinCNN 中卷积层的钩子
        self.net.protein_extractor.conv1.register_forward_hook(get_feature('protein_conv1'))
        self.net.protein_extractor.conv2.register_forward_hook(get_feature('protein_conv2'))
        self.net.protein_extractor.conv3.register_forward_hook(get_feature('protein_conv3'))

    def forward(self, x, returnt='logits'):
        # 从 DrugBAN 获取输出
        v_p = x[0]
        bg_d = x[1]
        v_d, v_p, f, score = self.net(bg_d, v_p, mode='train')

        if returnt == 'features':
            # 假设 'f' 是来自 DrugBAN 的特征张量（根据需要调整）
            return f
        elif returnt == 'logits':
            # 'score' 是来自 DrugBAN 的分类器输出 logits
            return score
        elif returnt == 'prob':
            # 如果需要从 logits 计算概率
            return self.softmax(score)
        elif returnt == 'all':
            # 返回所有输出（logits，概率，特征）
            probs = self.softmax(score)
            return score, probs, f

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.net.apply(xavier)


if __name__ == '__main__':
    # 示例 config（根据您的实际配置修改）
    config = {
        "DRUG": {"NODE_IN_FEATS": 128, "NODE_IN_EMBEDDING": 64, "HIDDEN_LAYERS": [64, 128]},
        "PROTEIN": {"EMBEDDING_DIM": 128, "NUM_FILTERS": [32, 64], "KERNEL_SIZE": 3, "PADDING": 1},
        "DECODER": {"IN_DIM": 128, "HIDDEN_DIM": 64, "OUT_DIM": 50, "BINARY": True},
        "BCN": {"HEADS": 4},
    }

    # 用 config 初始化模型
    model = DrugBAN(indim=128, hiddim=64, outdim=50, args=None, config=config)

    # 示例输入张量（根据您的数据调整）
    bg_d = torch.ones([2, 3, 128, 128])  # 模拟药物输入
    v_p = torch.ones([2, 128])  # 模拟蛋白质输入

    # 前向传播
    logits, probs, features = model(bg_d, v_p, returnt='all')

    print("Logits:", logits)
    print("Probabilities:", probs)
    print("Features:", features)

    ipdb.set_trace()  # 调试点
