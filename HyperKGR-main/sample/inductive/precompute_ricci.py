"""
预计算 Ollivier-Ricci 曲率，生成与 load_data.py 中 KG 数组等长的 ricci 数组。
用法: python3 precompute_ricci.py --data_path ./data/WN18RR_v1
"""
import argparse
import os
import numpy as np
import networkx as nx

try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    HAS_RICCI = True
except ImportError:
    HAS_RICCI = False
    print("WARNING: GraphRicciCurvature not installed. Install via: pip install GraphRicciCurvature")
    print("Will use Forman-Ricci approximation instead.")


def load_entities(path):
    entity2id = {}
    with open(os.path.join(path, 'entities.txt')) as f:
        for line in f:
            entity, eid = line.strip().split()
            entity2id[entity] = int(eid)
    return entity2id


def load_relations(path):
    relation2id = {}
    with open(os.path.join(path, 'relations.txt')) as f:
        for line in f:
            relation, rid = line.strip().split()
            relation2id[relation] = int(rid)
    return relation2id


def load_triples(path, filename, entity2id, relation2id):
    triples = []
    with open(os.path.join(path, filename)) as f:
        for line in f:
            h, r, t = line.strip().split()
            h, r, t = entity2id[h], relation2id[r], entity2id[t]
            triples.append([h, r, t])
    return triples


def build_graph(triples):
    """构建无向图（忽略关系类型，只看拓扑）"""
    G = nx.Graph()
    for h, r, t in triples:
        if h != t:
            G.add_edge(h, t)
    return G


def compute_ollivier_ricci(G):
    """计算 Ollivier-Ricci 曲率"""
    orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    ricci_dict = {}
    for u, v, data in orc.G.edges(data=True):
        rc = data.get('ricciCurvature', 0.0)
        ricci_dict[(u, v)] = rc
        ricci_dict[(v, u)] = rc
    return ricci_dict


def compute_forman_ricci(G):
    """Forman-Ricci 曲率（fallback，不需要额外库）"""
    ricci_dict = {}
    for u, v in G.edges():
        rc = 4.0 - G.degree(u) - G.degree(v)
        ricci_dict[(u, v)] = rc
        ricci_dict[(v, u)] = rc
    return ricci_dict


def build_ricci_array(triples, n_ent, n_rel, ricci_dict):
    """
    构建与 load_data.py 中 KG 数组完全等长的 ricci 数组。
    KG 的构成顺序（见 load_data.py load_graph）：
      1. 原始三元组 (h, r, t)
      2. 逆三元组 (t, r+n_rel, h)    — 与原始交替存储在 triples 中
      3. 自环 (e, 2*n_rel, e)         — 最后 n_ent 条
    """
    # triples 已经包含了正向+逆向（load_data.read_triples 中交替添加的）
    ricci_arr = []
    for h, r, t in triples:
        rc = ricci_dict.get((h, t), 0.0)
        ricci_arr.append(rc)

    # 自环：ricci = 0
    for e in range(n_ent):
        ricci_arr.append(0.0)

    return np.array(ricci_arr, dtype=np.float32)


def process_graph(data_path, entity2id, relation2id, tag='tra'):
    """处理一个图（transductive 或 inductive）"""
    n_ent = len(entity2id)
    n_rel = len(relation2id)

    # 读取三元组（与 load_data.read_triples 一致：正向+逆向交替）
    triples = []
    with open(os.path.join(data_path, 'train.txt')) as f:
        for line in f:
            h, r, t = line.strip().split()
            h, r, t = entity2id[h], relation2id[r], entity2id[t]
            triples.append([h, r, t])
            triples.append([t, r + n_rel, h])  # 逆边

    # 构建无向图
    raw_triples = [[h, r, t] for h, r, t in triples if r < n_rel]  # 只取正向边
    G = build_graph(raw_triples)
    print(f"  [{tag}] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 计算 Ricci 曲率
    if HAS_RICCI:
        print(f"  [{tag}] Computing Ollivier-Ricci curvature...")
        ricci_dict = compute_ollivier_ricci(G)
    else:
        print(f"  [{tag}] Computing Forman-Ricci curvature (fallback)...")
        ricci_dict = compute_forman_ricci(G)

    # 构建等长数组
    ricci_arr = build_ricci_array(triples, n_ent, n_rel, ricci_dict)
    print(f"  [{tag}] Ricci array length: {len(ricci_arr)}, "
          f"range: [{ricci_arr.min():.3f}, {ricci_arr.max():.3f}], "
          f"mean: {ricci_arr.mean():.3f}")

    return ricci_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset, e.g., ./data/WN18RR_v1')
    args = parser.parse_args()

    trans_dir = args.data_path
    ind_dir = args.data_path + '_ind'

    # 加载 entity 和 relation 映射
    entity2id_tra = load_entities(trans_dir)
    entity2id_ind = load_entities(ind_dir)
    relation2id = load_relations(trans_dir)

    print(f"Dataset: {args.data_path}")
    print(f"  Transductive: {len(entity2id_tra)} entities, {len(relation2id)} relations")
    print(f"  Inductive:    {len(entity2id_ind)} entities")

    # 处理 transductive 图
    print("\nProcessing transductive graph...")
    ricci_tra = process_graph(trans_dir, entity2id_tra, relation2id, tag='tra')
    save_path_tra = os.path.join(trans_dir, 'ricci_tra.npy')
    np.save(save_path_tra, ricci_tra)
    print(f"  Saved to {save_path_tra}")

    # 处理 inductive 图
    print("\nProcessing inductive graph...")
    ricci_ind = process_graph(ind_dir, entity2id_ind, relation2id, tag='ind')
    save_path_ind = os.path.join(ind_dir, 'ricci_ind.npy')
    np.save(save_path_ind, ricci_ind)
    print(f"  Saved to {save_path_ind}")

    print("\nDone!")


if __name__ == '__main__':
    main()
