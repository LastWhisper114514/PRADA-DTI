import os
import sys
import yaml
import torch
import numpy as np
import requests
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch.utils.data import ConcatDataset, DataLoader
from Bio.PDB import MMCIFParser, PDBParser
import py3Dmol
import matplotlib as mpl
from datasets import get_dataset, ContinualDataset, NAMES as DATASET_NAMES
from datasets.dti import MultiGraphData, BiosnapDataset
from preprocess.ligand_init import ligand_init
from preprocess.protein_init import protein_init
from backbones import get_backbone
from backbones.PSICHIC_.utils.utils import compute_pna_degrees
from models import get_all_models, get_model
from utils import get_loss
from utils.args import add_backbone_args, add_management_args
from utils.conf import set_random_seed
from utils.best_args import best_args

# === 代理设置（可选）===
proxies = {
    "http": "http://172.18.131.120:7890",
    "https": "http://172.18.131.120:7890",
}


# =====================================================
# 配置加载
# =====================================================
def load_config_for_backbone(backbone_name: str) -> dict:
    config_path = os.path.join("backbone_config", f"{backbone_name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_or_compute_degrees(config, train_loader):
    degree_path = os.path.join("data/", config["datafolder"], "degree.pt")

    if os.path.exists(degree_path):
        print("Loading cached PNA degrees...")
        degree_dict = torch.load(degree_path)
        mol_deg = degree_dict["ligand_deg"]
        prot_deg = degree_dict["protein_deg"]
    else:
        print("Computing training data degrees for PNA...")
        if isinstance(train_loader, (list, tuple)):
            combined_dataset = ConcatDataset([dl.dataset for dl in train_loader])
            train_loader = DataLoader(
                combined_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=train_loader[0].collate_fn
                if hasattr(train_loader[0], "collate_fn")
                else None,
            )
        mol_deg, clique_deg, prot_deg = compute_pna_degrees(train_loader)
        degree_dict = {
            "ligand_deg": mol_deg,
            "clique_deg": clique_deg,
            "protein_deg": prot_deg,
        }
        torch.save(degree_dict, degree_path)

    return mol_deg, prot_deg
# =====================================================
# 归因方法
# =====================================================
def grad_x_input_importance(model, batch, target_logit_idx=0):
    x = batch.prot_node_evo.detach().clone().requires_grad_(True)
    batch.prot_node_evo = x
    outputs = model(batch)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    logit = logits[0, target_logit_idx]
    logit.backward()
    grads = x.grad
    imp = (grads * x).sum(dim=-1)
    return imp.detach().cpu().numpy()
def integrated_gradients_importance(
    model, batch, steps=50, baseline="zero", target_logit_idx=0
):
    x = batch.prot_node_evo.detach().clone()
    if baseline == "zero":
        x0 = torch.zeros_like(x)
    elif baseline == "mean":
        x0 = x.mean(dim=0, keepdim=True).repeat(x.size(0), 1)
    else:
        x0 = baseline.to(x.device)

    total = torch.zeros_like(x)
    for s in range(1, steps + 1):
        alpha = float(s) / steps
        xt = (x0 + alpha * (x - x0)).detach().clone().requires_grad_(True)
        batch.prot_node_evo = xt
        model.zero_grad()
        outputs = model(batch)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        logit = logits[0, target_logit_idx]
        logit.backward()
        total += xt.grad

    avg_grad = total / steps
    ig = ((x - x0) * avg_grad).sum(dim=-1)
    return ig.detach().cpu().numpy()
# =====================================================
# PDB 查询与下载
# =====================================================
def search_pdb_by_sequence(seq, identity_cutoff=0.9, max_hits=3):
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 10,
                "identity_cutoff": identity_cutoff,
                "sequence_type": "protein",
                "value": seq,
            },
        },
        "return_type": "entry",
        "request_options": {
            "scoring_strategy": "sequence",
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }
    r = requests.post(url, json=query, proxies=proxies, timeout=20)
    if r.ok:
        data = r.json()
        return [result["identifier"] for result in data["result_set"]]
    else:
        print("Query failed:", r.text)
        return []
def download_pdb(pdb_id, out_dir="pdbs", fmt="cif", proxies=None):
    os.makedirs(out_dir, exist_ok=True)
    pdb_id = pdb_id.lower()
    if fmt == "pdb":
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    else:
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"

    try:
        r = requests.get(url, proxies=proxies, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to download {pdb_id}.{fmt}: {e}")
        return None

    out_path = os.path.join(out_dir, f"{pdb_id}.{fmt}")
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"[SUCCESS] Downloaded {fmt.upper()} to: {out_path}")
    return out_path


# =====================================================
# 可视化结构（用 py3Dmol 上色）
# =====================================================
def get_multi_gradient_color(score, color_stops):
    """
    根据分数计算多色渐变色
    :param score: 分数值，范围[0,1]
    :param color_stops: 颜色停靠点列表，格式为[(位置, 颜色), ...]
                        位置范围[0,1]，例如[(0, '#ff0000'), (0.5, '#ffff00'), (1, '#00ff00')]
    :return: 对应分数的渐变色，格式为#RRGGBB
    """
    # 确保颜色停靠点按位置排序
    sorted_stops = sorted(color_stops, key=lambda x: x[0])

    # 处理边界情况
    if score <= sorted_stops[0][0]:
        return sorted_stops[0][1]
    if score >= sorted_stops[-1][0]:
        return sorted_stops[-1][1]

    # 找到分数所在的颜色区间
    for i in range(len(sorted_stops) - 1):
        if sorted_stops[i][0] <= score <= sorted_stops[i + 1][0]:
            # 计算在当前区间内的相对位置
            start_pos, start_color = sorted_stops[i]
            end_pos, end_color = sorted_stops[i + 1]
            relative_pos = (score - start_pos) / (end_pos - start_pos)

            # 将十六进制颜色转换为RGB值
            def hex_to_rgb(hex_str):
                hex_str = hex_str.lstrip('#')
                return tuple(int(hex_str[j:j + 2], 16) for j in (0, 2, 4))

            # 线性插值计算RGB值
            r_start, g_start, b_start = hex_to_rgb(start_color)
            r_end, g_end, b_end = hex_to_rgb(end_color)

            r = int(r_start + (r_end - r_start) * relative_pos)
            g = int(g_start + (g_end - g_start) * relative_pos)
            b = int(b_start + (b_end - b_start) * relative_pos)

            # 转换回十六进制
            return '#%02x%02x%02x' % (r, g, b)

    # 默认为最后一个颜色
    return sorted_stops[-1][1]
# def hex_to_rgb(hex_color: str):
#     """Convert hex color (#rrggbb) to RGB tuple (0-255)."""
#     hex_color = hex_color.lstrip('#')
#     return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
# def custom_rgb(score, low="#0000ff", mid="#ffffff", high="#ff0000"):
#     """
#     自定义渐变: 低 -> 中 -> 高
#     输入颜色可以是 hex 形式 (#rrggbb)
#     """
#     low = hex_to_rgb(low)
#     mid = hex_to_rgb(mid)
#     high = hex_to_rgb(high)
#
#     if score <= 0.5:
#         t = score / 0.5
#         r = int(low[0] + t * (mid[0] - low[0]))
#         g = int(low[1] + t * (mid[1] - low[1]))
#         b = int(low[2] + t * (mid[2] - low[2]))
#     else:
#         t = (score - 0.5) / 0.5
#         r = int(mid[0] + t * (high[0] - mid[0]))
#         g = int(mid[1] + t * (high[1] - mid[1]))
#         b = int(mid[2] + t * (high[2] - mid[2]))
#     return r, g, b
def visualize_structure_with_scores(
    cif_path, seq, scores, chain_id="A",
    save_prefix=None, show=True,
    cmap_name="plasma"
):
    """
    在 3D 结构上可视化归因分数（主链渐变上色，其他链灰色透明，并展示配体）
    """
    import py3Dmol
    from Bio.PDB import MMCIFParser, PDBParser
    import numpy as np
    import matplotlib.pyplot as plt

    # 归一化 [0,1]
    scores = np.array(scores)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # 加载结构
    if cif_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", cif_path)

    # 检查是否有配体（非标准氨基酸）
    ligands = []
    for model in structure:
        for ch in model:
            for residue in ch:
                hetfield = residue.id[0]
                if hetfield.strip():  # HETATM = 配体
                    ligands.append((ch.id, residue.resname))
    if not ligands:
        print("[INFO] No ligand found, skip visualization.")
        return None

    # 设置py3Dmol视图
    view = py3Dmol.view(width=2400, height=1800)
    view.addModel(open(cif_path, "r").read(), "cif" if cif_path.endswith(".cif") else "pdb")

    # 默认所有链灰色透明
    view.setStyle({'cartoon': {'color': 'lightgrey', 'opacity': 0.3}})

    # 主链：渐变色上色
    cmap = plt.get_cmap(cmap_name)
    for i, residue in enumerate(seq):
        score = scores[i]
        r, g, b, _ = cmap(score)
        hex_color = '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))
        view.addStyle({'chain': chain_id, 'resi': i+1},
                      {'cartoon': {'color': hex_color, 'opacity': 1.0}})

    # 配体显示：sticks 样式
    for chain_id_lig, lig_resname in ligands:
        view.addStyle({'chain': chain_id_lig, 'resn': lig_resname},
                      {'stick': {'colorscheme': 'cyanCarbon', 'radius': 0.3}})

    view.zoomTo()

    # 保存交互式HTML
    if save_prefix:
        html_path = f"{save_prefix}.html"
        with open(html_path, "w") as f:
            f.write(view._make_html())
        print(f"[INFO] Saved interactive HTML to {html_path}")

    if show:
        return view





def save_attributions(model, loader, domain_id, save_path, methods=("grad", "ig")):
    """
    对 loader 里的蛋白做归因分析，并保存结果
    key = 蛋白序列 (seq)，值 = {grad, ig}
    """
    model.eval()
    device = next(model.parameters()).device
    results = {}

    for i, batch in enumerate(loader):
        batch = batch.to(device)

        # 获取序列
        if hasattr(batch, "prot_seq"):
            seq = batch.prot_seq[0]
        else:
            print(f"[WARN] batch {i} 没有 prot_seq，跳过")
            continue

        # grad × input
        grad_imp = grad_x_input_importance(model, batch, target_logit_idx=0) if "grad" in methods else None
        # integrated gradients
        ig_imp = integrated_gradients_importance(model, batch, steps=50, target_logit_idx=0) if "ig" in methods else None

        # 截断到真实长度
        true_len = len(seq)
        if grad_imp is not None: grad_imp = grad_imp[:true_len]
        if ig_imp is not None: ig_imp = ig_imp[:true_len]

        # 保存
        results[seq] = {}
        if grad_imp is not None:
            results[seq]["grad"] = grad_imp
        if ig_imp is not None:
            results[seq]["ig"] = ig_imp

        if i % 20 == 0:
            print(f"[INFO] Domain {domain_id}: processed {i} samples")

    # 保存为 npz
    np.savez_compressed(save_path, **results)
    print(f"[SUCCESS] Saved attributions for domain {domain_id} -> {save_path}")# =====================================================
# 参数
# =====================================================
def parse_args():
    parser = ArgumentParser(description="analyze sequence importance")
    parser.add_argument("--dataset", type=str, default="dti", choices=DATASET_NAMES)
    parser.add_argument("--model", type=str, default="ours", choices=get_all_models())
    parser.add_argument("--loss", type=str, default="ce")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--domain-id", type=int, default=2)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--C", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1208)
    add_management_args(parser)
    add_backbone_args(parser)
    args = parser.parse_args()
    if args.seed is not None:
        set_random_seed(args.seed)
    args.tpp_args = {"pe": 0.2, "pf": 0.2, "prompts": 1}
    args.d_data = 1313
    args.weight_decay = 1e-5
    return args

# =====================================================
# Main
# =====================================================
# def main(args=None):
#     if args is None:
#         args = parse_args()
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     dataset = get_dataset(args)
#     _, test_loader, _, _ = dataset.get_data_loaders()
#
#     config = load_config_for_backbone(args.backbone)
#     mol_deg, prot_dig = load_or_compute_degrees(config, dataset.train_loaders)
#     config["mol_deg"] = mol_deg
#     config["prot_dig"] = prot_dig
#
#     backbone = get_backbone(
#         backbone_name=args.backbone,
#         indim=dataset.INDIM,
#         hiddim=args.hiddim,
#         outdim=dataset.N_CLASSES_PER_TASK,
#         args=args,
#         configs=config,
#     )
#     args.config = config
#     loss = get_loss(loss_name=args.loss)
#     model = get_model(args, backbone, loss, dataset.get_transform()).to(device)
#
#     ckpt_path = f"/home/pan/LastWhisper/DTI_DIL/checkpoints/dti/ours/{args.backbone}/{args.seed}/domain-{args.domain_id}.pt"
#     print(f"[INFO] Loading checkpoint: {ckpt_path}")
#     state = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(state, strict=False)
#     model.eval()
#
#     # 用字典存所有归因结果
#     results = {}
#
#     for i, batch in enumerate(test_loader):
#         if i < args.sample_idx:
#             continue
#         if i >= args.sample_idx + args.num_samples:
#             break
#         batch = batch.to(device)
#
#         grad_imp = grad_x_input_importance(model, batch, target_logit_idx=0)
#         ig_imp = integrated_gradients_importance(model, batch, steps=50, target_logit_idx=0)
#
#         if hasattr(batch, "prot_seq"):
#             true_len = len(batch.prot_seq[0])
#             grad_imp = grad_imp[:true_len]
#             ig_imp = ig_imp[:true_len]
#             seq = batch.prot_seq[0]
#         else:
#             seq = f"sample{i}"
#
#         # === 保存归因结果 ===
#         results[seq] = {
#             "grad": grad_imp,
#             "ig": ig_imp
#         }
#
#         if i % 20 == 0:
#             print(f"[INFO] Domain {args.domain_id}: processed {i} samples")
#
#         # === 以下原有逻辑，暂时注释掉 ===
#         """
#         if hasattr(batch, "prot_seq"):
#             print(f"[INFO] Testing PDB download for sample {i}, seq length={len(seq)}")
#             hits = search_pdb_by_sequence(seq, identity_cutoff=0.9, max_hits=1)
#             if hits:
#                 pdb_id = hits[0]
#                 print(f"[SUCCESS] Found experimental PDB: {pdb_id}")
#                 pdb_path = download_pdb(pdb_id, out_dir="pdbs", fmt="cif", proxies=proxies)
#                 if pdb_path is not None:
#                     save_prefix = f"outputs/struct_domain{args.domain_id}_sample{i}"
#                     view = visualize_structure_with_scores(
#                         pdb_path, seq, grad_imp,
#                         chain_id="A",
#                         save_prefix=save_prefix,
#                         cmap_name="plasma"
#                     )
#                     # png_path = f"outputs/struct_domain{args.domain_id}_sample{i}.png"
#                     # with open(png_path, "wb") as f:
#                     #     f.write(view.png())
#                     # print(f"[INFO] Saved structure visualization: {png_path}")
#             else:
#                 print(f"[FAIL] No PDB found for sample {i}")
#
#         plt.figure(figsize=(12, 4))
#         plt.plot(grad_imp, label="Grad×Input", alpha=0.7)
#         plt.plot(ig_imp, label="Integrated Gradients", alpha=0.7)
#         plt.title(f"Protein attribution - Domain {args.domain_id}, Sample {i}")
#         plt.xlabel("Residue position")
#         plt.ylabel("Importance score")
#         plt.legend()
#         plt.tight_layout()
#         out_path = f"outputs/protein_attr_domain{args.domain_id}_sample{i}.png"
#         plt.savefig(out_path)
#         plt.close()
#         print(f"[INFO] Saved 1D attribution plot: {out_path}")
#         """
#
#     # === 循环结束后统一保存 ===
#     save_path = f"outputs/attributions_domain{args.domain_id}.npz"
#     np.savez_compressed(save_path, **results)
#     print(f"[SUCCESS] Saved attributions for domain {args.domain_id} -> {save_path}")

def run_all_domains(args, num_domains=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(args)
    _, test_loader, _, _ = dataset.get_data_loaders()

    config = load_config_for_backbone(args.backbone)
    mol_deg, prot_dig = load_or_compute_degrees(config, dataset.train_loaders)
    config["mol_deg"] = mol_deg
    config["prot_dig"] = prot_dig

    for domain_id in range(1, num_domains + 1):
        print(f"\n========== Domain {domain_id} ==========")
        args.domain_id = domain_id  # 动态更新 domain_id

        backbone = get_backbone(
            backbone_name=args.backbone,
            indim=dataset.INDIM,
            hiddim=args.hiddim,
            outdim=dataset.N_CLASSES_PER_TASK,
            args=args,
            configs=config,
        )
        args.config = config
        loss = get_loss(loss_name=args.loss)
        model = get_model(args, backbone, loss, dataset.get_transform()).to(device)

        ckpt_path = f"/home/pan/LastWhisper/DTI_DIL/checkpoints/dti/ours/{args.backbone}/{args.seed}/domain-{domain_id}.pt"
        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()

        # === 保存归因结果 ===
        results = {}

        for i, batch in enumerate(test_loader):
            if i < args.sample_idx:
                continue
            if i >= args.sample_idx + args.num_samples:
                break
            batch = batch.to(device)

            grad_imp = grad_x_input_importance(model, batch, target_logit_idx=0)
            ig_imp = integrated_gradients_importance(model, batch, steps=50, target_logit_idx=0)

            if hasattr(batch, "prot_seq"):
                true_len = len(batch.prot_seq[0])
                grad_imp = grad_imp[:true_len]
                ig_imp = ig_imp[:true_len]
                seq = batch.prot_seq[0]
            else:
                seq = f"sample{i}"

            results[seq] = {"grad": grad_imp, "ig": ig_imp}

            if i % 20 == 0:
                print(f"[INFO] Domain {domain_id}: processed {i} samples")

        save_path = f"outputs/attributions_domain{domain_id}.npz"
        np.savez_compressed(save_path, **results)
        print(f"[SUCCESS] Saved attributions for domain {domain_id} -> {save_path}")

def main(args=None):
    if args is None:
        args = parse_args()
    run_all_domains(args, num_domains=15)

if __name__ == "__main__":
    main()
