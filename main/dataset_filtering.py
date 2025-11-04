import os
import pandas as pd
import torch
from tqdm import tqdm
import sys

# 设置项目根路径，确保可以 import 成功
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'datasets'))
sys.path.append(os.path.join(project_root, 'backbones'))

from datasets.dti import ProteinMoleculeDataset  # ← 注意你的路径是否是这个


# === 设置路径 ===
DATA_PATH = './data/biosnap/fulldata.csv'
MOL_PATH = './data/biosnap/fullligand.pt'
PROT_PATH = './data/biosnap/fullprotein.pt'
OUTPUT_PATH = './data/biosnap/cleaned_fulldata.csv'

# === 加载原始数据 ===
data = pd.read_csv(DATA_PATH)
data['task_id'] = data['target_domain'].astype(int)

# === 只清洗 Task 5 ===
task_id = 5
task_df = data[data['task_id'] == task_id]
print(f"[*] Screening Task {task_id} for invalid samples...")

# === 加载图数据 ===
mol_graphs = torch.load(MOL_PATH)
prot_graphs = torch.load(PROT_PATH)

# === 初始化黑名单列表 ===
blacklist = []

# === 筛查每个样本 ===
for idx, row in tqdm(task_df.iterrows(), total=len(task_df)):
    try:
        dataset = ProteinMoleculeDataset(pd.DataFrame([row]), mol_graphs, prot_graphs)
        _ = dataset[0]  # 触发 __getitem__ 构建图
    except Exception as e:
        print(f"[!] Error at sample {idx}: {row['Ligand']} + {row['Protein']} => {e}")
        blacklist.append((row['Ligand'], row['Protein']))

# === 清洗掉黑名单样本 ===
if blacklist:
    print(f"\n[*] Found {len(blacklist)} invalid samples. Removing...")
    blacklist_set = set(blacklist)
    mask = data.apply(lambda row: (row['Ligand'], row['Protein']) not in blacklist_set, axis=1)
    cleaned_df = data[mask]
else:
    cleaned_df = data

# === 保存清洗后文件 ===
cleaned_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n[✓] Cleaned file saved to: {OUTPUT_PATH}")