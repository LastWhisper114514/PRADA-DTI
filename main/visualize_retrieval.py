# visualize_retrieval.py
import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

dataset_name="human"


# ========== Step 1. 数据加载 ==========
def load_data(dataset_name="bindingdb"):
    df = pd.read_csv(f'./data/{dataset_name}/fulldata.csv')
    prots = torch.load(f'./data/{dataset_name}/protein.pt')
    return df, prots

def extract_embeddings(df, prots, method="mean"):
    embeddings, domain_ids = [], []
    for _, row in df.iterrows():
        key = row['Protein']
        domain_id = int(row['new_domain_id'])
        evo = prots[key]['token_representation']  # [L, D]

        if method == "mean":
            emb = evo.mean(0)   # [D]
        else:
            emb = evo[0]        # 取第一个token

        embeddings.append(emb.numpy())
        domain_ids.append(domain_id)

    return np.stack(embeddings, axis=0), np.array(domain_ids)

# ========== Step 2. Prototype 聚类 ==========
def compute_prototypes(embeddings, domain_ids, k=3, save_path="outputs/"):
    os.makedirs(save_path, exist_ok=True)  # ⬅️ 新增：自动建目录

    prototypes, proto_domain_ids = [], []
    for domain in np.unique(domain_ids):
        domain_emb = embeddings[domain_ids == domain]
        if len(domain_emb) < k:
            n_clusters = 1
        else:
            n_clusters = k
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(domain_emb)
        prototypes.append(kmeans.cluster_centers_)
        proto_domain_ids.extend([domain] * n_clusters)

    prototypes = np.vstack(prototypes)
    proto_domain_ids = np.array(proto_domain_ids)

    np.save(f"{save_path}/prototypes.npy", prototypes)
    np.save(f"{save_path}/proto_domain_ids.npy", proto_domain_ids)

    return prototypes, proto_domain_ids

# ========== Step 3. 检索 & 分析 ==========
def analyze_retrieval(embeddings, domain_ids, prototypes, proto_domain_ids,
                      k=1, weighting="softmax", alpha=1.0, return_soft=False):
    """
    Prototype retrieval analysis with weighted Top-k accuracy.

    Args:
        embeddings: [N, D]
        domain_ids: [N]
        prototypes: [P, D]
        proto_domain_ids: [P]
        k: Top-k
        weighting: 权重计算方式 ("uniform"|"inverse"|"softmax")
        alpha: softmax 温度系数 (alpha>1 会让分布更尖锐)
        return_soft: 是否返回 soft 矩阵

    Returns:
        top1_acc: Top-1 accuracy
        topk_weighted_acc: Top-k weighted accuracy
        cm_soft (optional): [D, D] soft similarity matrix
    """
    dist = euclidean_distances(embeddings, prototypes)      # [N, P]
    nearest_k = np.argsort(dist, axis=1)[:, :k]             # [N, k]
    nearest_domains = proto_domain_ids[nearest_k]           # [N, k]
    nearest_dist = np.take_along_axis(dist, nearest_k, axis=1)  # [N, k]

    # --- 权重计算 ---
    if weighting == "uniform":
        weights = np.ones_like(nearest_dist) / k
    elif weighting == "inverse":
        weights = 1 / (nearest_dist + 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)
    elif weighting == "softmax":
        exps = np.exp(-alpha * nearest_dist)
        weights = exps / exps.sum(axis=1, keepdims=True)
    else:
        raise ValueError("weighting must be 'uniform'|'inverse'|'softmax'")

    # --- Top-1 accuracy ---
    retrieved_domain = nearest_domains[:, 0]
    top1_acc = (retrieved_domain == domain_ids).mean()

    # --- Top-k Weighted Accuracy ---
    scores = []
    for true_d, doms, w in zip(domain_ids, nearest_domains, weights):
        mask = (doms == true_d).astype(float)
        score = (mask * w).sum() / w.sum()
        scores.append(score)
    topk_weighted_acc = np.mean(scores)

    print(f"Top-1 acc: {top1_acc:.4f}, Top-{k} weighted acc (alpha={alpha}): {topk_weighted_acc:.4f}")

    if not return_soft:
        return top1_acc, topk_weighted_acc

    # --- Soft weighted similarity matrix ---
    num_domains = len(np.unique(proto_domain_ids))
    cm_soft = np.zeros((num_domains, num_domains))

    for i, true_d in enumerate(domain_ids):
        for j, retrieved_d in enumerate(nearest_domains[i]):
            cm_soft[true_d, retrieved_d] += weights[i, j]

    cm_soft = cm_soft / cm_soft.sum(axis=1, keepdims=True)
    print(f"Soft Top-{k} similarity matrix (alpha={alpha}) computed.")

    return top1_acc, topk_weighted_acc, cm_soft

def run_experiments(embeddings, domain_ids, save_path=f"outputs/{dataset_name}/"):
    os.makedirs(save_path, exist_ok=True)

    n_proto_list = [1, 3, 5, 7, 9]
    k_list = [1, 3, 5]
    alpha_list = [1.0]  # 温度超参数

    results = []

    for n_proto in n_proto_list:
        prototypes, proto_domain_ids = compute_prototypes(
            embeddings, domain_ids, k=n_proto, save_path=save_path
        )

        for k in k_list:
            if k > n_proto:
                continue

            for alpha in alpha_list:
                print(f"\n=== Experiment: n_proto={n_proto}, topk={k}, alpha={alpha} ===")

                top1_acc, topk_weighted_acc, cm_soft = analyze_retrieval(
                    embeddings, domain_ids, prototypes, proto_domain_ids,
                    k=k, weighting="softmax", alpha=alpha, return_soft=True
                )

                # 保存 soft similarity matrix
                sns.heatmap(cm_soft, cmap="Blues", cbar=True)
                plt.xlabel("Retrieved domain")
                plt.ylabel("True domain")
                plt.title(f"Soft Similarity Matrix (n_proto={n_proto}, k={k}, alpha={alpha})")
                plt.savefig(f"{save_path}/{dataset_name}cm_{n_proto}protos_k={k}.png")
                plt.close()

                # 保存结果到表格
                results.append({
                    "n_proto": n_proto,
                    "topk": k,
                    "alpha": alpha,
                    "top1_acc": top1_acc,
                    "topk_weighted_acc": topk_weighted_acc
                })

    # 存结果表格
    df = pd.DataFrame(results)
    df.to_csv(f"{save_path}/retrieval_results.csv", index=False)
    print(f"\nAll results saved to {save_path}/retrieval_results.csv")

def plot_confusion(domain_ids, retrieved_domain=None, cm_soft=None, mode="hard"):
    """
    绘制检索混淆矩阵
    mode:
        "hard"       - 普通混淆矩阵（Top-1 vote）
        "normalized" - 每行归一化
        "soft"       - Soft/Weighted Top-k 矩阵
    参数:
        domain_ids: [N] 样本真实domain
        retrieved_domain: [N] 样本预测domain (仅hard/normalized模式用)
        cm_soft: [D, D] soft矩阵 (仅soft模式用)
    """
    if mode == "soft":
        cm = cm_soft
    else:
        cm = confusion_matrix(domain_ids, retrieved_domain)
        if mode == "normalized":
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(cm, cmap="Blues", annot=False, cbar=True)
    plt.xlabel("Retrieved domain")
    plt.ylabel("True domain")
    plt.title(f"Prototype Retrieval Matrix ({mode})")
    plt.show()

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def visualize_retrieval_vs_true(embeddings, domain_ids, prototypes, proto_domain_ids, k=1, save_path="outputs/"):
    # Step 1. 检索 (Top-1)
    dist = euclidean_distances(embeddings, prototypes)
    nearest = dist.argmin(axis=1)
    retrieved_domain = proto_domain_ids[nearest]  # [N]

    # Step 2. 降维到2D/3D
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne_2d.fit_transform(embeddings)

    pca_3d = PCA(n_components=3, random_state=42)
    emb_3d = pca_3d.fit_transform(embeddings)

    # Step 3. 颜色映射
    unique_domains = np.unique(domain_ids)
    num_domains = len(unique_domains)
    cmap = plt.cm.get_cmap("tab20", num_domains)
    color_map = {d: cmap(i) for i, d in enumerate(unique_domains)}

    # ===== 2D 可视化 =====
    def plot_2d(points, labels, title, filename):
        plt.figure(figsize=(7, 6))
        for d in unique_domains:
            idx = (labels == d)
            plt.scatter(points[idx, 0], points[idx, 1], s=8, color=color_map[d], label=f"{d}", alpha=0.7)
        plt.title(title)
        plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{save_path}/{filename}", dpi=300)
        plt.close()

    plot_2d(emb_2d, domain_ids, "Ground Truth Domains (2D)", "tsne_true.png")
    plot_2d(emb_2d, retrieved_domain, "Retrieved Domains (Top-1, 2D)", "tsne_retrieved.png")

    # ===== 3D 可视化 =====
    def plot_3d(points, labels, title, filename):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        for d in unique_domains:
            idx = (labels == d)
            ax.scatter(points[idx, 0], points[idx, 1], points[idx, 2],
                       s=8, color=color_map[d], label=f"{d}", alpha=0.7)
        ax.set_title(title)
        ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{save_path}/{filename}", dpi=300)
        plt.show()
        plt.close()

    plot_3d(emb_3d, domain_ids, "Ground Truth Domains (3D)", "tsne_true_3d.png")
    plot_3d(emb_3d, retrieved_domain, "Retrieved Domains (Top-1, 3D)", "tsne_retrieved_3d.png")

    print(f"Saved 2D and 3D plots to {save_path}/")



# ========== 主流程 ==========
def main():
    df, prots = load_data()
    embeddings, domain_ids = extract_embeddings(df, prots, method="mean")

    run_experiments(embeddings, domain_ids, save_path="outputs/")

    prototypes, proto_domain_ids = compute_prototypes(embeddings, domain_ids, k=9)
    visualize_retrieval_vs_true(embeddings, domain_ids, prototypes, proto_domain_ids, k=1, save_path="outputs/")


if __name__ == "__main__":
    main()
