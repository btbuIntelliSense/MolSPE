from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch

import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS
from torch_geometric.data import Data, Batch
import numpy as np

# from MaskModel.Datamodel.Maskpoint import mask_points
from Datamodel.chemutils import get_clique_mol
# from torchmdnet.datasets.Dictionary import Dictionary
import tqdm

from MaskModel.MaskPoint import mask_points


def generate_conformer(smiles):
    # 1. 通过SMILES生成分子对象
    mol = Chem.MolFromSmiles(smiles)
    # 2. 添加氢原子
    mol = Chem.AddHs(mol)
    # 3. 生成3D构象
    AllChem.EmbedMolecule(mol, randomSeed=42)
    # 4. 优化构象以获得低能量稳定构象
    AllChem.MMFFOptimizeMolecule(mol)
    # 5. 提取原子序数、化学键信息和原子坐标
    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    # 获取原子坐标
    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        # 将键类型转换为整数
        edge_attr.append({
            Chem.rdchem.BondType.SINGLE:1,
            Chem.rdchem.BondType.DOUBLE:2,
            Chem.rdchem.BondType.TRIPLE:3,
            Chem.rdchem.BondType.AROMATIC:4,
                         }[bond_type])
        edge_index.extend([[start, end], [end, start]])

    # cliques = motif_decomp(mol)
    # num_motif = len(cliques)
    # num_atoms = len(z)
    # if num_motif > 0:
    #     motif_x = torch.tensor([120]).repeat_interleave(num_motif, dim=0)
    #     # self.motif_x = torch.cat((self.z, self.motif_x), dim=0)
    #     # 构建motif边索引
    #     motif_edge_index = []
    #     for k, motif in enumerate(cliques):
    #         motif_edge_index = motif_edge_index + [[i, num_atoms + k] for i in motif]
    #         motif_edge_index = torch.tensor(np.array(motif_edge_index).T, dtype=torch.long).to(
    #         edge_index.device)
    #
    #     # 构建motif边特征
    #     motif_edge_attr = torch.zeros(motif_edge_index.size()[1])
    #     motif_edge_attr[:] = 6  # bond type for self-loop edge
    #     motif_edge_attr = motif_edge_attr.to(edge_attr.dtype).to(edge_attr.device)
    #     # self.motif_edge_attr = torch.cat((self.edge_attr, self.motif_edge_attr), dim = 0)
    # else:
    #     motif_x = torch.tensor([], dtype=torch.long)
    #     motif_edge_index = torch.tensor([], dtype=torch.long)
    #     motif_edge_attr = torch.tensor([], dtype=torch.float)
    # mask_z, coord, mask_coord, dist, mask_dist, targets, src_edge_type = apply_mask_and_noise()

    return {'z':z,'pos': pos,'edge_index':edge_index, "edge_attr":edge_attr,}

def apply_mask_and_noise(self):
    # 准备数据字典
    data_item = {
        "z": self.z,
        "pos": self.pos,
        # "dist": self.dist
    }

    # 应用掩码和噪声
    mask_z, coord, mask_coord, dist, mask_dist, targets, src_edge_type = mask_points(
        [data_item],
        self.vocab,
        self.pad_idx,
        self.mask_idx,
        self.noise_type,
        self.noise,
        self.mask_prob,
        self.max_atoms
    )

    return mask_z, coord, mask_coord, dist, mask_dist, targets, src_edge_type

def get_conformer(smiles):
    # 1. 通过SMILES生成分子对象
    mol = Chem.MolFromSmiles(smiles)
    # 2. 添加氢原子
    mol = Chem.AddHs(mol)
    # 3. 生成3D构象
    AllChem.EmbedMolecule(mol, randomSeed=42)
    # 4. 优化构象以获得低能量稳定构象
    AllChem.MMFFOptimizeMolecule(mol)
    # 5. 提取原子序数、化学键信息和原子坐标
    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    # 获取原子坐标
    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)

    return {
            'z': z,
            'pos': pos,
        }

def add_position_noise(data,noise_scale):
    noise = torch.randn_like(data['pos']) * noise_scale
    data['noise'] = noise
    data['pos_target'] = data['pos']
    data['pos'] = data['pos'] + noise
    return data


def motif_decomp(mol):  # 分解motif
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]]

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) != 0:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

            # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if n_atoms > len(c) > 0]

    num_cli = len(cliques)
    ssr_mol = Chem.GetSymmSSSR(mol)
    for i in range(num_cli):
        c = cliques[i]
        cmol = get_clique_mol(mol, c)
        ssr = Chem.GetSymmSSSR(cmol)
        if len(ssr) > 1:
            for ring in ssr_mol:
                if len(set(list(ring)) & set(c)) == len(list(ring)):
                    cliques.append(list(ring))
                    cliques[i] = list(set(cliques[i]) - set(list(ring)))

    cliques = [c for c in cliques if n_atoms > len(c) > 0]

    return cliques

# smiles_list=['C','N']
# data=get_conformer(smiles)
# print(data)
# dataset=[]
# for smiles in smiles_list:
#    dataset = get_conformer(smiles)
#    dataset.append(dataset)
# dataset=dataset
# print(dataset)
# processed_data = []
# for smiles in smiles_list:
# # 获取分子数据
#    data = get_conformer(smiles)
# # 添加噪声数据
#    data_with_noise = add_position_noise(data, noise_scale=0.1)
#    processed_data.append(data_with_noise)
# print(processed_data)
# from torch.utils.data import Dataset, DataLoader
#
# class CustomDataset(Dataset):
#     def __init__(self, data_list):
#         self.data_list = data_list
#
#     def __len__(self):
#         return len(self.data_list)
#
#     def __getitem__(self, idx):
#         return self.data_list[idx]
#
# # 将数据包装成 Dataset
# dataset = CustomDataset(processed_data)
#
# # 创建 DataLoader
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#
# # 使用 DataLoader 加载数据
# for batch in dataloader:
#     print(batch)
