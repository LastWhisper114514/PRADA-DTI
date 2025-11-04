import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class DTI_Dataset(Dataset):
    """
    用于药物-靶标相互作用预测的 Dataset 类。
    """
    NAME = 'dti-core50'  # 修改为适应 DTI 的数据集名称
    N_CLASSES_PER_TASK = 2  # 假设每个任务是一个二分类问题（有或没有相互作用）
    N_TASKS = 10  # 假设有 10 个任务（可以根据需要调整）
    INDIM = (1, 128, 128)  # 输入维度可以根据实际情况修改，例如药物 SMILES 和靶标蛋白质序列
    MAX_N_SAMPLES_PER_TASK = 16000  # 每个任务的最大样本数

    def __init__(self, drug_data, target_data, labels, drug_encoder=None, target_encoder=None):
        """
        :param drug_data: 药物的 SMILES 字符串
        :param target_data: 靶标的蛋白质序列
        :param labels: 相互作用标签（0 或 1）
        :param drug_encoder: 用于编码药物的 LabelEncoder（可选）
        :param target_encoder: 用于编码靶标的 LabelEncoder（可选）
        """
        self.drug_data = drug_data
        self.target_data = target_data
        self.labels = labels
        self.drug_encoder = drug_encoder if drug_encoder else LabelEncoder()
        self.target_encoder = target_encoder if target_encoder else LabelEncoder()

        # 编码标签
        self.labels = self.drug_encoder.fit_transform(self.labels)

    def __getitem__(self, index):
        """
        获取数据集的一个样本：药物 SMILES、靶标蛋白质序列及其标签
        """
        drug_smiles = self.drug_data[index]  # 药物 SMILES
        target_sequence = self.target_data[index]  # 靶标蛋白质序列
        label = self.labels[index]  # 标签，0 或 1

        # 将药物 SMILES 和靶标蛋白质序列转换为数字表示
        drug_tensor = self.encode_drug(drug_smiles)
        target_tensor = self.encode_target(target_sequence)

        return drug_tensor, target_tensor, torch.tensor(label, dtype=torch.long)

    def encode_drug(self, smiles: str) -> torch.Tensor:
        """
        将 SMILES 字符串编码为数字表示。
        """
        smiles_encoded = [ord(c) for c in smiles]  # 使用 ASCII 值编码
        return torch.tensor(smiles_encoded, dtype=torch.long)

    def encode_target(self, sequence: str) -> torch.Tensor:
        """
        将靶标蛋白质序列编码为数字形式。
        """
        amino_acid_map = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                          'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
        sequence_encoded = [amino_acid_map.get(aa, 20) for aa in sequence]  # 20 为未知氨基酸的默认值
        return torch.tensor(sequence_encoded, dtype=torch.long)

    def __len__(self):
        return len(self.drug_data)

    @staticmethod
    def get_loss():
        """
        适用于二分类任务的损失函数。
        """
        return torch.nn.BCEWithLogitsLoss()  # 二分类任务使用 BCEWithLogitsLoss

    @staticmethod
    def get_batch_size():
        return 128

    @staticmethod
    def get_minibatch_size():
        return DTI_Dataset.get_batch_size()
