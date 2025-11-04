import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import requests
import py3Dmol

# ----------------------
# 配置代理（可选）
proxies = {
    "http": "http://172.18.131.120:7890",
    "https": "http://172.18.131.120:7890",
}


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

def html_to_png(html_path, png_path, width=800, height=600, wait=2):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    from selenium.webdriver.chrome.service import Service

    service = Service("/usr/local/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)

    driver.set_window_size(width, height)
    driver.get("file://" + os.path.abspath(html_path))
    time.sleep(wait)  # 等待 py3Dmol 渲染完成
    driver.save_screenshot(png_path)
    driver.quit()


# ----------------------
# PDB 查询与下载
def search_pdb_by_sequence(seq, identity_cutoff=0.9, max_hits=1):
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

# ----------------------
# 可视化工具
def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def custom_rgb(score, low="#0000ff", mid="#ffffff", high="#ff0000"):
    low = hex_to_rgb(low)
    mid = hex_to_rgb(mid)
    high = hex_to_rgb(high)
    if score <= 0.5:
        t = score / 0.5
        r = int(low[0] + t * (mid[0] - low[0]))
        g = int(low[1] + t * (mid[1] - low[1]))
        b = int(low[2] + t * (mid[2] - low[2]))
    else:
        t = (score - 0.5) / 0.5
        r = int(mid[0] + t * (high[0] - mid[0]))
        g = int(mid[1] + t * (high[1] - mid[1]))
        b = int(mid[2] + t * (high[2] - mid[2]))
    return r, g, b

def visualize_structure_with_scores(
    cif_path, seq, scores, chain_id="A",
    save_prefix=None,
    low="#0000ff", mid="#ffffff", high="#ff0000",
    custom_view=None, save_png=False
):
    scores = np.array(scores)
    if scores.max() - scores.min() < 1e-8:
        scores = np.zeros_like(scores)
    else:
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    view = py3Dmol.view(width=800, height=600)
    view.addModel(open(cif_path, "r").read(), "cif" if cif_path.endswith(".cif") else "pdb")

    # 默认全链半透明
    view.setStyle({}, {"cartoon": {"color": "grey", "opacity": 0.2}})

    # 主链上色
    for i, residue in enumerate(seq):
        score = scores[i]
        r, g, b = custom_rgb(score, low, mid, high)
        hex_color = '#%02x%02x%02x' % (r, g, b)
        view.addStyle({'chain': chain_id, 'resi': i+1}, {'cartoon': {'color': hex_color, "opacity": 1.0}})

    # 配体
    view.addStyle({"hetflag": True, "resn": ["HOH"]}, {"hidden": True})
    view.addStyle({"hetflag": True}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.25}})

    # 固定视角
    if custom_view is not None:
        view.setView(custom_view)
    else:
        view.zoomTo()

    # 保存 HTML
    if save_prefix:
        html_path = f"{save_prefix}.html"
        with open(html_path, "w") as f:
            f.write(view._make_html())
        print(f"[INFO] Saved HTML: {html_path}")

        if save_png:
            png_path = f"{save_prefix}.png"
            html_to_png(html_path, png_path)
            print(f"[INFO] Saved PNG: {png_path}")


# ----------------------
# 平滑
def smooth_curve(scores, method="gaussian", sigma=2, window=5):
    if method == "gaussian":
        return gaussian_filter1d(scores, sigma=sigma)
    elif method == "moving":
        return np.convolve(scores, np.ones(window)/window, mode="same")
    else:
        return scores

# ----------------------
# 可视化单个蛋白（grad 和 ig）
def save_pdb_with_scores(cif_path, seq, scores, out_path, chain_id="A"):
    from Bio.PDB import MMCIFParser, PDBParser, PDBIO
    import numpy as np

    scores = np.array(scores)
    if scores.max() - scores.min() < 1e-8:
        scores = np.zeros_like(scores)
    else:
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    # 读取结构
    if cif_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", cif_path)

    # 给 chain A 残基打分
    idx = 0
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if idx < len(scores):
                    for atom in residue:
                        atom.set_bfactor(float(scores[idx] * 100))  # 放大到 0~100
                    idx += 1

    # 保存
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)
    print(f"[INFO] Saved PDB with scores -> {out_path}")


def visualize_protein(seq, npz_dir="outputs", out_root="outputs/vis"):
    os.makedirs(out_root, exist_ok=True)

    # 加载所有 domain 的结果
    all_scores = {"grad": {}, "ig": {}}
    for d in range(1, 16):
        data = np.load(f"{npz_dir}/attributions_domain{d}.npz", allow_pickle=True)
        if seq not in data:
            continue
        entry = data[seq].item()
        for method in ["grad", "ig"]:
            if method in entry:
                scores = smooth_curve(entry[method], method="gaussian", sigma=2)
                all_scores[method][d] = scores

    if not any(all_scores[m] for m in all_scores):
        print(f"[WARN] Sequence not found in any domain: {seq[:10]}...")
        return

    # 下载 PDB
    hits = search_pdb_by_sequence(seq, identity_cutoff=0.9, max_hits=1)
    if not hits:
        print("[FAIL] No PDB found.")
        return
    pdb_id = hits[0]
    pdb_path = download_pdb(pdb_id, out_dir="pdbs", fmt="cif", proxies=proxies)
    if pdb_path is None:
        print(f"[FAIL] Could not download PDB {pdb_id}, skipping {seq[:10]}...")
        return

    # 输出目录
    seq_tag = f"{seq[:10]}_{len(seq)}"
    base_dir = os.path.join(out_root, seq_tag)
    os.makedirs(base_dir, exist_ok=True)

    # === 分别保存 grad / ig ===
    for method in ["grad", "ig"]:
        method_dir = os.path.join(base_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # 保存 15 个 domain
        for d, scores in all_scores[method].items():
            out_path = os.path.join(method_dir, f"domain{d}.pdb")
            save_pdb_with_scores(pdb_path, seq, scores, out_path, chain_id="A")

        # 保存差值 (2..15 vs 1)
        if 1 in all_scores[method]:
            for d in range(2, 16):
                if d not in all_scores[method]:
                    continue
                diff = all_scores[method][d] - all_scores[method][d-1]

                out_path = os.path.join(method_dir, f"diff_{d}.pdb")
                save_pdb_with_scores(pdb_path, seq, diff, out_path, chain_id="A")

                if np.abs(diff).max() > 1e-8:
                    diff_norm = diff / np.abs(diff).max()
                else:
                    diff_norm = diff
                out_path_norm = os.path.join(method_dir, f"diff_{d}_norm.pdb")
                save_pdb_with_scores(pdb_path, seq, diff_norm, out_path_norm, chain_id="A")


# ----------------------
# 批量处理所有蛋白
def visualize_all_proteins(npz_dir="outputs", out_root="outputs/vis"):
    base_file = f"{npz_dir}/attributions_domain1.npz"
    base_data = np.load(base_file, allow_pickle=True)
    seqs = list(base_data.keys())

    print(f"[INFO] Found {len(seqs)} proteins from domain1.")
    for idx, seq in enumerate(seqs):
        print(f"\n[INFO] Visualizing protein {idx+1}/{len(seqs)}: {seq[:10]}... len={len(seq)}")
        visualize_protein(seq, npz_dir=npz_dir, out_root=out_root)

# ----------------------
if __name__ == "__main__":
    visualize_all_proteins(npz_dir="outputs", out_root="outputs/vis")
