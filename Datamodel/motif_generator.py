import networkx as nx
import math
import copy
# from pysmiles import read_smiles
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from collections import defaultdict
from tqdm import tqdm

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol
#
# with open(r'D:\work\MotifMol3D-main\data\train_smi.txt') as f:
#     smile_list = [line.strip().split('\t')[0] for line in f]


# class motif_generator(object):
#     def __init__(self,smile_list):
#         g_list=[]
#         self.smiles = smile_list
#         for i in self.smiles:
#             graph = read_smiles(i)
#             g_list.append(graph)
#         self.g_list = g_list
#         # self.mol_net = read_smiles(self.smiles)
#         self.vocab = {}
#         self.whole_node_count = {}
#         self.weight_vocab = {}
#         self.node_count = {}
#         self.edge_count = {}
#         self.g_e_weight = {}
#
#     @property
#     def get_motif_dict(self):
#         g_list = self.g_list
#         for g in tqdm(range(len(g_list)), desc='Get motif dict', unit='graph'):
#             clique_list = []
#             # 将环和键分离并记录
#             mcb = nx.cycle_basis(g_list[g])
#             mcb_tuple = [tuple(ele) for ele in mcb]
#             # print(g_list[g].edges)
#             # print(mcb_tuple)
#             edges = []
#             edges_mcb = []
#             for e in g_list[g].edges():
#                 count = 0
#                 for c in mcb_tuple:
#                     if e[0] in set(c) and e[1] in set(c):
#                         count += 1
#                         break
#                     elif e[0] in set(c) or e[1] in set(c):
#                         edges_mcb.append(e)
#                 if count == 0:
#                     edges.append(e)
#             # 记录分子中不属于环的边
#             edges = list(set(edges))
#             nodes_labels, g_smiles = self.get_node_labels(g)
#
#             element = nx.get_edge_attributes(g_list[g], name="order")
#             atoms = g_list[g].nodes
#
#             for e in edges:
#                 weight = element[tuple(e)]
#                 edge = ((nodes_labels[e[0]], nodes_labels[e[1]]), weight)
#                 clique_id = self.add_to_vocab(edge)
#                 clique_list.append(clique_id)
#                 if clique_id not in self.whole_node_count:
#                     self.whole_node_count[clique_id] = 1
#                 else:
#                     self.whole_node_count[clique_id] += 1
#
#             for m in mcb_tuple:
#                 weight = tuple(self.find_ring_weights(m, g_list[g],element))
#                 ring = []
#                 for i in range(len(m)):
#                     ring.append(nodes_labels[m[i]])
#                 cycle = (tuple(ring), weight)
#                 cycle_id = self.add_to_vocab(cycle)
#                 clique_list.append(cycle_id)
#                 if cycle_id not in self.whole_node_count:
#                     self.whole_node_count[cycle_id] = 1
#                 else:
#                     self.whole_node_count[cycle_id] += 1
#
#             for e in clique_list:
#                 self.add_weight(e, g)
#
#             c_list = tuple(set(clique_list))
#
#             for e in c_list:
#                 if e not in self.node_count:
#                     self.node_count[e] = 1
#                 else:
#                     self.node_count[e] += 1
#
#             e_weight = {}
#             for e in c_list:
#                 e_weight[e] = self.weight_vocab[(g, e)]/(len(edges) + len(mcb_tuple))
#             self.g_e_weight[g_smiles] = e_weight
#
#         for i in list(self.g_e_weight.keys()):
#             for m in list(self.g_e_weight[i].keys()):
#                 self.g_e_weight[i][m] = self.g_e_weight[i][m] * (math.log((len(self.g_list) + 1) / self.node_count[m]))
#         tf_tdf = []
#         for i in list(self.g_e_weight.keys()):
#             for m in list(self.g_e_weight[i].keys()):
#                 tf_tdf.append(self.g_e_weight[i][m])
#         tf_tdf_order = sorted(tf_tdf)
#         max_id = len(tf_tdf_order) - 1
#
#         for i in list(self.g_e_weight.keys()):
#             for m in list(self.g_e_weight[i].keys()):
#                 self.g_e_weight[i][m] = (self.g_e_weight[i][m] - tf_tdf_order[0]) / (tf_tdf_order[max_id] - tf_tdf_order[0])
#
#     def motif_id_get(self,smiles,cut_off):
#         motif_order_dict = sorted(self.g_e_weight[smiles].items(), key=lambda x: x[1], reverse=True)
#         motif_order = list(id for id, tf_tdf in motif_order_dict)
#         if len(motif_order) >= cut_off:
#             motif_order = motif_order[0:cut_off]
#         else:
#             motif_order = motif_order + [400 for i in range(cut_off-len(motif_order))]
#
#         return torch.tensor(motif_order, dtype=torch.long)
#
#     def motif_embedding(self, smiles):
#         motif_embedding = np.zeros((1, len(self.vocab)))
#         for motif_id in list(self.g_e_weight[smiles].keys()):
#             motif_embedding[0, int(motif_id)] = self.g_e_weight[smiles][motif_id]
#         motif_embedding = torch.tensor(motif_embedding, dtype=torch.float32)
#
#         return motif_embedding
#
#     def get_node_labels(self,g):
#         allowable_features = {
#             'possible_atomic_num_list': list(range(0, 119))}
#         atom_features_list = []
#         mol = get_mol(self.smiles[g])
#         for atom in mol.GetAtoms():
#             atom_feature = [allowable_features['possible_atomic_num_list'].index(
#                 atom.GetAtomicNum())]
#             atom_features_list.extend(atom_feature)
#
#         return atom_features_list, self.smiles[g]
#
#     def add_to_vocab(self, clique):
#         c = copy.deepcopy(clique[0])
#         weight = copy.deepcopy(clique[1])
#         for i in range(len(c)):
#             if (c, weight) in self.vocab:
#                 return self.vocab[(c, weight)]
#             else:
#                 c = self.shift_right(c)
#                 weight = self.shift_right(weight)
#         self.vocab[(c, weight)] = len(list(self.vocab.keys()))
#
#         return self.vocab[(c, weight)]
#
#     def add_weight(self, node_id, g):
#         if (g, node_id) not in self.weight_vocab:
#             self.weight_vocab[(g, node_id)] = 1
#         else:
#             self.weight_vocab[(g, node_id)] += 1
#
#     @staticmethod
#     def shift_right(l):
#         if type(l) == int:
#             return l
#         elif type(l) == tuple:
#             l = list(l)
#             return tuple([l[-1]] + l[:-1])
#         elif type(l) == list:
#             return tuple([l[-1]] + l[:-1])
#         else:
#             print('ERROR!')
#
#     @staticmethod
#     def find_ring_weights(ring, g, element):
#         weight_list = []
#         for i in range(len(ring) - 1):
#             try:
#                 weight = element[tuple([ring[i], ring[i+1]])]
#                 weight_list.append(weight)
#             except:
#                 weight = element[tuple([ring[i + 1], ring[i]])]
#                 weight_list.append(weight)
#
#         try:
#             weight = element[tuple([ring[-1], ring[0]])]
#             weight_list.append(weight)
#         except:
#             weight = element[tuple([ring[0], ring[-1]])]
#             weight_list.append(weight)
#
#         return weight_list

# class motif_generator_smidd(object):
#     def __init__(self, ori=None):
#         g_list=[]
#         self.smiles = ori
#         for i in self.smiles:
#             graph = read_smiles(i)
#             g_list.append(graph)
#         self.g_list = g_list
#         # self.mol_net = read_smiles(self.smiles)
#         self.vocab = {}
#         self.whole_node_count = {}
#         self.weight_vocab = {}
#         self.node_count = {}
#         self.edge_count = {}
#         self.g_e_weight = {}
#
#     @property
#     def get_motif_dict(self):
#         g_list = self.g_list
#         for g in tqdm(range(len(g_list)), desc='Get motif dict', unit='graph'):
#             clique_list = []
#             # 将环和键分离并记录
#             mcb = nx.cycle_basis(g_list[g])
#             mcb_tuple = [tuple(ele) for ele in mcb]
#
#             edges = []
#             edges_mcb = []
#             for e in g_list[g].edges():
#                 count = 0
#                 for c in mcb_tuple:
#                     if e[0] in set(c) and e[1] in set(c):
#                         count += 1
#                         break
#                     elif e[0] in set(c) or e[1] in set(c):
#                         edges_mcb.append(e)
#                 if count == 0:
#                     edges.append(e)
#             # 记录分子中不属于环的边
#             edges = list(set(edges))
#             nodes_labels, g_smiles = self.get_node_labels(g)
#
#             element = nx.get_edge_attributes(g_list[g], name="order")
#             atoms = g_list[g].nodes
#
#             for e in edges:
#                 weight = element[tuple(e)]
#                 edge = ((nodes_labels[e[0]], nodes_labels[e[1]]), weight)
#                 clique_id = self.add_to_vocab(edge)
#                 clique_list.append(clique_id)
#                 if clique_id not in self.whole_node_count:
#                     self.whole_node_count[clique_id] = 1
#                 else:
#                     self.whole_node_count[clique_id] += 1
#
#             for m in mcb_tuple:
#                 weight = tuple(self.find_ring_weights(m, g_list[g],element))
#                 ring = []
#                 for i in range(len(m)):
#                     ring.append(nodes_labels[m[i]])
#                 cycle = (tuple(ring), weight)
#                 cycle_id = self.add_to_vocab(cycle)
#                 clique_list.append(cycle_id)
#                 if cycle_id not in self.whole_node_count:
#                     self.whole_node_count[cycle_id] = 1
#                 else:
#                     self.whole_node_count[cycle_id] += 1
#
#             for e in clique_list:
#                 self.add_weight(e, g)
#
#             c_list = tuple(set(clique_list))
#
#             for e in c_list:
#                 if e not in self.node_count:
#                     self.node_count[e] = 1
#                 else:
#                     self.node_count[e] += 1
#
#             e_weight = {}
#             for e in c_list:
#                 e_weight[e] = self.weight_vocab[(g, e)]/(len(edges) + len(mcb_tuple))
#             self.g_e_weight[g_smiles] = e_weight
#
#         for i in list(self.g_e_weight.keys()):
#             for m in list(self.g_e_weight[i].keys()):
#                 self.g_e_weight[i][m] = self.g_e_weight[i][m] * (math.log((len(self.g_list) + 1) / self.node_count[m]))
#
#         tf_tdf = []
#         for i in list(self.g_e_weight.keys()):
#             for m in list(self.g_e_weight[i].keys()):
#                 tf_tdf.append(self.g_e_weight[i][m])
#         tf_tdf_order = sorted(tf_tdf)
#         max_id = len(tf_tdf_order) - 1
#
#         for i in list(self.g_e_weight.keys()):
#             for m in list(self.g_e_weight[i].keys()):
#                 self.g_e_weight[i][m] = (self.g_e_weight[i][m] - tf_tdf_order[0]) / (tf_tdf_order[max_id] - tf_tdf_order[0])
#
#     def motif_id_get(self,smiles,cut_off):
#         motif_order_dict = sorted(self.g_e_weight[smiles].items(), key=lambda x: x[1], reverse=True)
#         motif_order = list(id for id, tf_tdf in motif_order_dict)
#         if len(motif_order) >= cut_off:
#             motif_order = motif_order[0:cut_off]
#         else:
#             motif_order = motif_order + [400 for i in range(cut_off-len(motif_order))]
#
#         return torch.tensor(motif_order, dtype=torch.long)
#
#     def motif_embedding(self, smiles):
#         motif_embedding = np.zeros((1, len(self.vocab)))
#         for motif_id in list(self.g_e_weight[smiles].keys()):
#             motif_embedding[0, int(motif_id)] = self.g_e_weight[smiles][motif_id]
#         motif_embedding = torch.tensor(motif_embedding, dtype=torch.float32)
#
#         return motif_embedding
#
#     def get_node_labels(self,g):
#         allowable_features = {
#             'possible_atomic_num_list': list(range(0, 119))}
#         atom_features_list = []
#         mol = get_mol(self.smiles[g])
#         for atom in mol.GetAtoms():
#             atom_feature = [allowable_features['possible_atomic_num_list'].index(
#                 atom.GetAtomicNum())]
#             atom_features_list.extend(atom_feature)
#
#         return atom_features_list, self.smiles[g]
#
#     def add_to_vocab(self, clique):
#         c = copy.deepcopy(clique[0])
#         weight = copy.deepcopy(clique[1])
#         for i in range(len(c)):
#             if (c, weight) in self.vocab:
#                 return self.vocab[(c, weight)]
#             else:
#                 c = self.shift_right(c)
#                 weight = self.shift_right(weight)
#         self.vocab[(c, weight)] = len(list(self.vocab.keys()))
#
#         return self.vocab[(c, weight)]
#
#     def add_weight(self, node_id, g):
#         if (g, node_id) not in self.weight_vocab:
#             self.weight_vocab[(g, node_id)] = 1
#         else:
#             self.weight_vocab[(g, node_id)] += 1
#
#     @staticmethod
#     def shift_right(l):
#         if type(l) == int:
#             return l
#         elif type(l) == tuple:
#             l = list(l)
#             return tuple([l[-1]] + l[:-1])
#         elif type(l) == list:
#             return tuple([l[-1]] + l[:-1])
#         else:
#             print('ERROR!')
#
#     @staticmethod
#     def find_ring_weights(ring, g, element):
#         weight_list = []
#         for i in range(len(ring) - 1):
#             try:
#                 weight = element[tuple([ring[i], ring[i+1]])]
#                 weight_list.append(weight)
#             except:
#                 weight = element[tuple([ring[i + 1], ring[i]])]
#                 weight_list.append(weight)
#
#         try:
#             weight = element[tuple([ring[-1], ring[0]])]
#             weight_list.append(weight)
#         except:
#             weight = element[tuple([ring[0], ring[-1]])]
#             weight_list.append(weight)
#
#         return weight_list



class motif_generator_torch(object):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list
        self.vocab = {}  # 基序 -> 唯一ID映射
        self.whole_node_count = defaultdict(int)  # 基序在整个数据集中出现的总次数
        self.weight_vocab = {}  # 分子中基序的权重
        self.node_count = defaultdict(int)  # 包含每个基序的分子数
        self.g_e_weight = {}  # 每个分子的基序权重字典
        self.mol_cache = {}
        # 新增加的数据结构
        self.motif_atoms = defaultdict(dict)  # 基序 -> 分子SMILES -> 原子索引集合
        self.motif_definitions = {}  # 基序ID -> 基序定义（用于后续原子索引匹配）
        self.max_atoms = 0
        self.max_occurrence = 0
        self.process_molecules()
        self._calculate_max_values()

    def _calculate_max_values(self):
        """计算最大原子数和最大实例数"""
        for motif_id, motif_mols in self.motif_atoms.items():
            # 获取当前基序的原子数
            motif_type, atomic_nums, num_atoms = self.motif_definitions[motif_id]
            if num_atoms > self.max_atoms:
                self.max_atoms = num_atoms
            # 计算每个分子中当前基序的最大实例数
            for smiles, atoms_set in motif_mols.items():
                if len(atoms_set) > self.max_occurrence:
                    self.max_occurrence = len(atoms_set)

    def process_molecules(self):
        """处理所有分子，构建基序词典和原子索引映射"""
        print("Processing molecules to build motif vocabulary...")
        # 第一遍：构建词汇表
        for smiles in tqdm(self.smiles_list, desc="Building motif vocabulary"):
            mol = Chem.MolFromSmiles(smiles)
            self.mol_cache[smiles] = mol
            if not mol:
                continue

            # 获取所有环信息
            ssr = rdmolops.GetSSSR(mol)
            ring_systems = []
            for ring in ssr:
                ring_atoms = list(ring)
                ring_systems.append(tuple(sorted(ring_atoms)))

            # 获取所有非环键
            non_ring_bonds = []
            for bond in mol.GetBonds():
                if not bond.IsInRing():
                    a1 = bond.GetBeginAtomIdx()
                    a2 = bond.GetEndAtomIdx()
                    non_ring_bonds.append((a1, a2))

            # 处理环基序
            for ring_atoms in ring_systems:
                self._add_motif(smiles, ring_atoms, "ring")

            # 处理非环单键基序
            for bond in non_ring_bonds:
                self._add_motif(smiles, bond, "bond")

        # 第二遍：计算TF-IDF权重
        self._calculate_weights()

    def _add_motif(self, smiles, atoms, motif_type):
        """添加基序到词汇表并记录原子索引"""
        mol = self.mol_cache[smiles]
        # 创建基序的唯一标识符
        atomic_nums = tuple(sorted([
            mol.GetAtomWithIdx(atom).GetAtomicNum()
            for atom in atoms
        ]))
        motif_key = (motif_type, atomic_nums, len(atoms))

        # 添加到词汇表
        if motif_key not in self.vocab:
            motif_id = len(self.vocab)
            self.vocab[motif_key] = motif_id
            self.motif_definitions[motif_id] = motif_key

        motif_id = self.vocab[motif_key]

        # 记录原子索引（转换为元组保持可哈希性）
        atom_set = tuple(sorted(atoms))

        # 更新基序的原子索引映射
        if smiles not in self.motif_atoms[motif_id]:
            self.motif_atoms[motif_id][smiles] = set()

        self.motif_atoms[motif_id][smiles].add(atom_set)
        self.whole_node_count[motif_id] += 1

    def _calculate_weights(self):
        """计算每个分子的基序权重（TF-IDF）"""
        print("Calculating motif weights (TF-IDF)...")
        num_molecules = len(self.smiles_list)

        # 计算IDF部分：包含每个基序的分子数量
        for motif_id in self.motif_definitions:
            count = sum(1 for smiles in self.smiles_list
                        if smiles in self.motif_atoms[motif_id])
            self.node_count[motif_id] = count

        # 计算每个分子的TF-IDF权重
        for smiles in tqdm(self.smiles_list, desc="Calculating TF-IDF weights"):
            self.g_e_weight[smiles] = {}
            motif_count = defaultdict(int)

            # 计算每个基序在当前分子中出现的次数
            for motif_id in self.motif_definitions:
                if smiles in self.motif_atoms[motif_id]:
                    count = len(self.motif_atoms[motif_id][smiles])
                    motif_count[motif_id] = count

            # 总基序数
            total_motifs = sum(motif_count.values())

            # 计算每个基序的TF-IDF权重
            for motif_id, count in motif_count.items():
                tf = count / total_motifs
                idf = np.log((num_molecules + 1) /
                             (self.node_count[motif_id] + 1))
                self.g_e_weight[smiles][motif_id] = tf * idf

    # def motif_id_get(self, smiles, cut_off):
    #
    #     # 创建填充值张量
    #     motif_ids = torch.full((cut_off,), 400, dtype=torch.long)
    #     atom_tensors = torch.full((cut_off, self.max_occurrence, self.max_atoms),
    #                               -1, dtype=torch.long)
    #     """获取分子的前N个基序ID和对应的原子索引"""
    #     if smiles not in self.g_e_weight:
    #         return torch.zeros(cut_off, dtype=torch.long), [[] for _ in range(cut_off)]
    #
    #     # 按TF-IDF权重排序
    #     sorted_motifs = sorted(
    #         self.g_e_weight[smiles].items(),
    #         key=lambda x: x[1],
    #         reverse=True
    #     )

        # motif_ids = []
        # atom_sets = []
        #
        # # 收集前N个基序的ID和原子索引
        # for i, (motif_id, _) in enumerate(sorted_motifs):
        #     if i >= cut_off:
        #         break
        #     motif_ids.append(motif_id)
        #
        #     # 获取该基序在当前分子的所有原子集合
        #     atoms_in_motif = self.motif_atoms[motif_id].get(smiles, set())
        #     atom_sets.append([atoms for atoms in atoms_in_motif])
        #
        # # 填充不足部分
        # num_motifs = len(motif_ids)
        # if num_motifs < cut_off:
        #     motif_ids += [400] * (cut_off - num_motifs)
        #     atom_sets += [[] for _ in range(cut_off - num_motifs)]

        # 为每个选中的基序填充数据
        # for i, (motif_id, _) in enumerate(sorted_motifs):
        #     motif_ids[i] = motif_id
        #     atom_sets = self.motif_atoms[motif_id].get(smiles, set())
        #
        #     # 获取基序的原子数
        #     _, _, num_atoms = self.motif_definitions[motif_id]
        #
        #     # 填充每个实例的原子索引
        #     for j, atom_tuple in enumerate(atom_sets):
        #         if j >= self.max_occurrence:
        #             break
        #         # 将原子索引转换为张量
        #         indices = torch.tensor(list(atom_tuple), dtype=torch.long)
        #         atom_tensors[i, j, :num_atoms] = indices
        #
        # return torch.tensor(motif_ids, dtype=torch.long), atom_tensors

    def motif_id_get(self, smiles, cut_off):
        motif_ids = torch.full((cut_off,), 400, dtype=torch.long)
        atom_tensors = torch.full(
            (cut_off, self.max_occurrence, self.max_atoms),
            -1, dtype=torch.long
        )

        sorted_motifs = sorted(
            self.g_e_weight[smiles].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for i, (motif_id, _) in enumerate(sorted_motifs):
            if i >= cut_off:  # 确保不会超出cut_off范围
                break
            motif_ids[i] = motif_id
            occurrences = self.motif_atoms[motif_id].get(smiles, [])

            for j, atom_set in enumerate(occurrences):
                if j >= self.max_occurrence:
                    break
                atoms = list(atom_set)[:self.max_atoms]
                atom_tensors[i, j, :len(atoms)] = torch.tensor(atoms)

        return motif_ids, atom_tensors


class MotifGenerator(object):
    def __init__(self, smiles_list, max_atoms_per_motif=20, max_motifs=10):
        self.smiles_list = smiles_list
        self.max_atoms_per_motif = max_atoms_per_motif
        self.max_motifs = max_motifs

        # 基础数据结构
        self.vocab = {}  # 基序特征 -> ID
        self.id_to_feature = {}  # ID -> 基序特征
        self.motif_info = defaultdict(dict)  # smiles -> {motif_id: atom indices}
        self.tf_idf = {}  # smiles -> {motif_id: tf-idf score}

        # 合并后的数据结构
        self.unioned_motif_map = defaultdict(dict)

        self.process_molecules()

    def process_molecules(self):
        """处理所有分子，构建基序词典"""
        print("Building motif vocabulary...")

        # 第一遍：识别基础基序并记录原子索引
        base_motif_map = {}
        for smiles in tqdm(self.smiles_list, desc="Identifying base motifs"):
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue

            # 获取环系统和非环键
            rings, bonds = self._get_motif_atoms(mol)
            base_motifs = {}

            # 处理环基序
            for ring_atoms in rings:
                motif_id = self._register_motif(mol, ring_atoms)
                base_motifs[motif_id] = list(ring_atoms)

            # 处理非环键基序
            for bond in bonds:
                motif_id = self._register_motif(mol, bond)
                base_motifs[motif_id] = list(bond)

            # 保存基础基序信息
            self.motif_info[smiles] = base_motifs
            base_motif_map[smiles] = base_motifs

        # 第二遍：合并共享原子的基序
        print("Merging motifs with shared atoms...")
        for smiles, motifs in tqdm(base_motif_map.items(), desc="Merging motifs"):
            self.unioned_motif_map[smiles] = self._merge_shared_motifs(smiles,motifs)

        # 第三遍：计算TF-IDF权重
        print("Calculating TF-IDF weights...")
        self._calculate_tf_idf()

    def _get_motif_atoms(self, mol):
        """识别分子中的基序原子组"""
        # 获取最小环系统
        ri = mol.GetRingInfo()
        rings = [set(ring_atoms) for ring_atoms in ri.AtomRings()]

        # 合并环系统
        merged = []
        while rings:
            current = rings.pop(0)
            # 查找共享原子的环系统
            to_merge = []
            for i, ring in enumerate(rings):
                if current & ring:
                    to_merge.append(i)

            # 合并共享原子的环
            for idx in reversed(to_merge):
                current |= rings[idx]
                del rings[idx]

            merged.append(current)

        # 获取非环键
        non_ring_bonds = []
        for bond in mol.GetBonds():
            if not bond.IsInRing():
                a1 = bond.GetBeginAtomIdx()
                a2 = bond.GetEndAtomIdx()
                non_ring_bonds.append({a1, a2})

        return merged, non_ring_bonds

    def _register_motif(self, mol, atoms):
        """注册基序并返回ID"""
        # 创建特征签名 (原子类型 + 键信息)
        atom_types = sorted([
            mol.GetAtomWithIdx(atom).GetAtomicNum()
            for atom in atoms
        ])
        atom_counts = len(atoms)

        feature = (tuple(atom_types), atom_counts)

        # 注册新基序
        if feature not in self.vocab:
            motif_id = len(self.vocab)
            self.vocab[feature] = motif_id
            self.id_to_feature[motif_id] = feature
        else:
            motif_id = self.vocab[feature]

        return motif_id

    def _merge_shared_motifs(self, smiles,motifs):
        """合并共享原子的基序"""
        merged = {}
        motif_ids = list(motifs.keys())
        atom_sets = [set(motifs[mid]) for mid in motif_ids]

        # 使用并查集算法合并共享原子的基序
        n = len(motif_ids)
        parent = list(range(n))

        # 查找函数
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        # 合并函数
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        # 查找共享原子的基序对
        for i in range(n):
            for j in range(i + 1, n):
                if atom_sets[i] & atom_sets[j]:
                    union(i, j)

        # 按连通分量合并基序
        components = defaultdict(set)
        for i in range(n):
            root = find(i)
            components[root] |= atom_sets[i]

        # 为每个合并后的基序组创建新ID
        merged_motifs = {}
        for atom_set in components.values():
            # 从原始基序创建特征
            all_atoms = sorted(atom_set)
            atom_types = sorted([
                Chem.MolFromSmiles(smiles).GetAtomWithIdx(atom).GetAtomicNum()
                for atom in all_atoms
            ])
            atom_counts = len(all_atoms)
            feature = (tuple(atom_types), atom_counts)

            # 注册新基序
            if feature not in self.vocab:
                motif_id = len(self.vocab)
                self.vocab[feature] = motif_id
                self.id_to_feature[motif_id] = feature
            else:
                motif_id = self.vocab[feature]

            merged_motifs[motif_id] = all_atoms

        return merged_motifs

    def _calculate_tf_idf(self):
        """计算每个基序的TF-IDF权重"""
        # 计算IDF (出现基序的分子数量)
        motif_presence = defaultdict(int)
        for motifs in self.unioned_motif_map.values():
            for motif_id in motifs.keys():
                motif_presence[motif_id] += 1

        # 计算每个分子的TF-IDF
        num_molecules = len(self.smiles_list)
        for smiles, motifs in self.unioned_motif_map.items():
            total = len(motifs)
            self.tf_idf[smiles] = {}
            for motif_id, atoms in motifs.items():
                # 分子中基序频率
                tf = 1.0 / total
                # 逆文档频率
                idf = np.log(num_molecules / (1 + motif_presence[motif_id]))
                self.tf_idf[smiles][motif_id] = tf * idf

    def get_motif_data(self, smiles):
        """获取分子中权重最高的基序和对应原子索引"""
        if smiles not in self.tf_idf:
            # 返回空数据
            empty_motifs = torch.full((self.max_motifs,), 400, dtype=torch.long)
            empty_atoms = torch.full((self.max_motifs, self.max_atoms_per_motif),
                                     -1, dtype=torch.long)
            return empty_motifs, empty_atoms

        # 按TF-IDF权重排序
        sorted_motifs = sorted(
            self.tf_idf[smiles].items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.max_motifs]

        # 准备输出张量
        motif_ids = torch.full((self.max_motifs,), 400, dtype=torch.long)
        atom_indices = torch.full(
            (self.max_motifs, self.max_atoms_per_motif),
            -1,
            dtype=torch.long
        )

        # 填充数据
        for i, (motif_id, score) in enumerate(sorted_motifs):
            motif_ids[i] = motif_id
            atoms = torch.tensor(
                self.unioned_motif_map[smiles][motif_id],
                dtype=torch.long
            )
            num_atoms = min(len(atoms), self.max_atoms_per_motif)
            if num_atoms > 0:
                atom_indices[i, :num_atoms] = atoms[:num_atoms]

        return motif_ids, atom_indices

# if __name__ == '__main__':
#     g = motif_generator().get_motif_dict
#     print(g)
# smile_test = 'Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O'
# smile_test1 = 'Nc1ncnc2c1ncn2'
# gh = motif_generator()
# gh.get_motif_dict
# gh.motif_id_get(smile_test, 6)
# embedding = gh.motif_embedding(smile_test)
# print(gh.vocab)
# print(embedding)
# print(len(gh.vocab))

