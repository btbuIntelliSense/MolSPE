import os
import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem,BRICS
from tqdm import tqdm

from Datamodel.chemutils import get_clique_mol
from Datamodel.motif_generator import motif_generator_torch
from MaskModel.MaskPoint import mask_points
from MaskModel.MaskPointdef import mask_pointsdef

# 常量定义
FILE_NAME = ".pt"
# BOND_TYPE_MAP = {
#     Chem.rdchem.BondType.SINGLE: 1,
#     Chem.rdchem.BondType.DOUBLE: 2,
#     Chem.rdchem.BondType.TRIPLE: 3,
#     Chem.rdchem.BondType.AROMATIC: 4
# }
BOND_ORDER_MAP = {0: 0,
                  1: 1, 1.5: 2, 2: 3, 3: 4}   #一个字典，用于将化学键的类型映射到整数

class MolPretrainDataset(Dataset):
    @property
    def raw_file_names(self):
        """自动识别raw目录下所有CSV文件"""
        return [f for f in os.listdir(osp.join(self.root, "raw")) if f.endswith('.csv')]

    @property
    def processed_file_names(self):
        """生成处理后的文件名列表"""
        return [f"data_{i}{FILE_NAME}" for i in range(self.size)]

    @property
    def processed_dir(self) -> str:
        """定义处理后的数据存储目录"""
        return osp.join(self.root, "processed", f"{self.mode}_data")

    def __init__(self, root, mode='pretrain1', vocab=None, pad_idx=0,
                noise_type='uniform', noise=0.1,noise_scale=0.04, mask_prob=0.15, max_atoms=256
                , fast_read=True, save_cache=True):
        """
        参数说明：
        root: 数据根目录
        mode: 预训练模式 (pretrain1/pretrain2)
        noise_std: 坐标噪声标准差
        max_nodes: 最大原子数（用于padding）
        fast_read: 是否启用快速读取模式
        save_cache: 是否保存内存缓存
        """
        # 初始化参数
        self.mode = mode
        self.vocab=vocab
        self.denoise_scale = noise_scale
        self.pad_idx=pad_idx
        self.mask_idx = self.vocab.add_symbol("[MASK]", is_special=True)
        self.noise_type = noise_type
        self.mask_noise = noise
        self.mask_prob = mask_prob
        self.max_atoms = max_atoms
        self.fast_read = fast_read
        self.save_cache = save_cache

        # 路径设置
        self.base_path = osp.join(root, "processed")
        self.size_path = osp.join(self.base_path, f"num_files_{mode}{FILE_NAME}")
        self.smiles_path = osp.join(self.base_path, f"smiles_list_{mode}{FILE_NAME}")

        # 加载元数据
        if osp.exists(self.size_path):
            self.size = int(torch.load(self.size_path))
            self.smiles_list = torch.load(self.smiles_path)
        else:
            self.size = 0
            self.smiles_list = []

        super().__init__(root)

        # 快速读取模式处理
        if fast_read:
            cache_path = osp.join(self.base_path, f'cache_{mode}{FILE_NAME}')
            if osp.isfile(cache_path) and save_cache:
                print(f"Loading cache from {cache_path}...")
                self.data = torch.load(cache_path)
            else:
                self.fast_read = False
                self.data = [self[i] for i in tqdm(range(self.size))]
                if save_cache:
                    torch.save(self.data, cache_path)
        # self.fast_read = fast_read

    def process(self):
        """核心处理流程"""
        os.makedirs(self.processed_dir, exist_ok=True)

        # 遍历所有原始CSV文件
        processed_idx = 0
        for raw_file in self.raw_file_names:
            df = pd.read_csv(osp.join(self.root, 'raw', raw_file))
            smiles_list = df['smiles'].tolist()

            motif_g = motif_generator_torch(smiles_list)
            self.smiles_list = []
            for idx, smiles in enumerate(tqdm(smiles_list, desc=f"Processing {raw_file}")):
                try:
                    data = self._process_smiles(smiles,idx)
                    if data is not None:
                        # 保存处理结果
                        # 获取基序ID和对应的原子索引
                        motif_vocab, motif_atoms = motif_g.motif_id_get(smiles_list[idx], 7)
                        # motif_vocab = motif_g.motif_id_get(smiles_list[idx], 7)
                        data['motif_vocab'] = motif_vocab
                        data['motif_atom'] = motif_atoms
                        torch.save(data,osp.join(self.processed_dir, f"data_{processed_idx}{FILE_NAME}"))
                        self.smiles_list.append(smiles)
                        processed_idx += 1
                except Exception as e:
                    print(f"Error processing {smiles}: {str(e)}")
                    continue

        # 保存元数据
        self.size = processed_idx
        torch.save(self.size, self.size_path)
        torch.save(self.smiles_list, self.smiles_path)

    def _process_smiles(self, smiles,idx):
        """处理单个SMILES分子"""
        # 分子生成与校验
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDG()
        params.randomSeed = 42  # 固定随机种子
        params.numThreads = 4  # 多线程加速
        params.maxAttempts = 100  # 最大尝试次数
        params.useSmallRingTorsions = True  # 小环扭角约束
        # 生成3D坐标
        try:
            # 检查EmbedMolecule是否成功生成构象(返回0表示成功)
            if AllChem.EmbedMolecule(mol, params) != 0:
                return None

            # 尝试优化分子构象
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                pass

            # 检查是否成功获取构象坐标
            conformer = mol.GetConformer()
            if conformer is None:
                return None

        except:
            return None

        n_atom = len(mol.GetAtoms())
        # 原子特征
        z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()],
                         dtype=torch.long)
        pos = torch.tensor(mol.GetConformer().GetPositions(),
                           dtype=torch.float32)

        data = {'z':z, 'pos':pos,'n_atom':n_atom}

        noise = torch.randn_like(pos) * self.denoise_scale
        data['noise'] = noise
        data['pos_denoise'] = pos + noise

        n_atom = len(mol.GetAtoms())
        bond_adj = torch.zeros(n_atom, n_atom, dtype=torch.long)
        # edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()
            bond_adj[(i, j)] = bond_adj[(j, i)] = 1
            if bond_type == 2:
                bond_adj[(i, j)] = bond_adj[(j, i)] = bond_adj[(i, j)] + 2
            if bond_type == 3:
                bond_adj[(i, j)] = bond_adj[(j, i)] = bond_adj[(i, j)] + 4
            if bond_type == 1.5:
                bond_adj[(i, j)] = bond_adj[(j, i)] = bond_adj[(i, j)] + 8

        data['bond_adj'] = bond_adj

        data['src_z'], data['mask_z'], data['coord'], data['mask_coord'], data['dist'], data['mask_dist'], data[
            'targets'], \
            data['src_edge_type'], data['bond_adj'], data['mask_z_nopad'], data['mask_idx'] = mask_points(data,
                                                                                                          self.vocab,
                                                                                                          self.pad_idx,
                                                                                                          self.mask_idx,
                                                                                                          self.noise_type,
                                                                                                          self.mask_noise,
                                                                                                          self.mask_prob,
                                                                                                          self.max_atoms)

        return data

    def len(self):
        return self.size

    def get(self, idx):
        """获取单个样本"""
        if self.fast_read:
            return self.data[idx]
        else:
            return torch.load(osp.join(self.processed_dir, f"data_{idx}" + FILE_NAME))


class MolfineDataSet(Dataset):
    @property
    def raw_file_names(self):
        return [file_dir for file_dir in os.listdir(osp.join(self.root, "raw")) if ".csv" in file_dir]

    @property
    def processed_file_names(self):
        return [f"mol_fine_data_{idx}" + FILE_NAME for idx in range(self.size)]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed", 'mol_fine_data')

    def __init__(self, root, task_name, target,  base_path=None, mode='pretrain1', vocab=None,
                 pad_idx=0,noise_type='uniform', noise=0.1, noise_scale=0.04, mask_prob=0.15,
                 max_atoms=256,fast_read=False, save_cache=False):
        self.root = root
        self.task_name = task_name
        self.target = target
        # 初始化参数
        self.mode = mode
        self.vocab = vocab
        self.denoise_scale = noise_scale
        self.pad_idx = pad_idx
        self.mask_idx = self.vocab.add_symbol("[MASK]", is_special=True)
        self.noise_type = noise_type
        self.mask_noise = noise
        self.mask_prob = mask_prob
        self.max_atoms = max_atoms
        self.fast_read = fast_read
        self.save_cache = save_cache
        # self.prompt = prompt
        # self.max_node = -1

        if base_path is None:
            base_path = osp.join(self.root, "processed")
        self.base_path = base_path
        self.size_path = osp.join(base_path, "num_files" + FILE_NAME)
        if osp.exists(self.size_path):
            self.size = int(torch.load(self.size_path))
            self.smiles_list = torch.load(osp.join(base_path, "smiles_list" + FILE_NAME))
        else:
            self.size = 0
        super(MolfineDataSet, self).__init__(root)

        if fast_read:
            data_cache_path = osp.join(base_path, f'cache_{task_name}' + FILE_NAME)
            if osp.isfile(data_cache_path) and save_cache:
                print(f"read cache from {data_cache_path}...")
                # self.data = torch.load(data_cache_path)
                self.data = torch.load(data_cache_path)
            else:
                self.fast_read = False
                self.data = [mol_data for mol_data in tqdm(self)]
                if save_cache:
                    # torch.save(self.data, data_cache_path)
                    torch.save(self.data, data_cache_path)

    def process(self):
        # os.makedirs(self.processed_dir, exist_ok=True)

        for raw_file_name in self.raw_file_names:
            print(f"Processing the {raw_file_name} dataset to torch geometric format...\n")
            smiles_list, mol_info_list, labels = self.load_finetune_dataset(osp.join(self.root, 'raw', raw_file_name)
                                                                        , target=self.target)
            # motif_g = motif_generator(smiles_list) # 生成词典
            # motif_g.get_motif_dict
            motif_g = motif_generator_torch(smiles_list) # 生成词典

            cur_idx = 0
            self.smiles_list = []
            for idx, mol_info in enumerate(tqdm(mol_info_list)):
                if mol_info is None or mol_info['n_atom'] == 0 or mol_info['n_atom'] > self.max_atoms:
                    continue
                self.smiles_list.append(smiles_list[idx])
                # padding_mol_info(mol_info, self.max_node)
                mol_fine_data = mol_info
                mol_fine_data['labels']=labels[idx]
                # 获取基序ID和对应的原子索引
                motif_vocab, motif_atoms = motif_g.motif_id_get(smiles_list[idx], 7)
                # motif_vocab = motif_g.motif_id_get(smiles_list[idx], 7)
                mol_fine_data['motif_vocab']=motif_vocab
                mol_fine_data['motif_atom'] = motif_atoms
                torch.save(mol_fine_data, osp.join(self.processed_dir, f"mol_fine_data_{cur_idx}" + FILE_NAME))
                cur_idx += 1
            self.size = cur_idx
        torch.save(self.size, self.size_path)
        torch.save(self.smiles_list, osp.join(self.base_path, 'smiles_list' + FILE_NAME))
        # 生成并保存motif字典
        # motif_vocab = {smiles: idx for idx, smiles in enumerate(sorted(all_motifs))}
        # torch.save(motif_vocab, osp.join(self.base_path, "motif_vocab.pt"))
        # print(f"Generated motif vocabulary with {len(motif_vocab)} unique motifs")


    def _process_smiles(self, smiles):
        """处理单个SMILES分子"""
        # 分子生成与校验
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)

        # 生成3D坐标
        try:
            # 检查EmbedMolecule是否成功生成构象(返回0表示成功)
            if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
                return None

            # 尝试优化分子构象
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                pass

            # 检查是否成功获取构象坐标
            conformer = mol.GetConformer()
            if conformer is None:
                return None

        except:
            return None

        n_atom = len(mol.GetAtoms())
        # 原子特征
        z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()],
                         dtype=torch.long)
        pos = torch.tensor(mol.GetConformer().GetPositions(),
                           dtype=torch.float32)
        pos=pos-pos.mean(axis=0)
        # 模式分支处理
        data = {'z':z, 'pos':pos,'n_atom':n_atom}

        # if 'finetune1' in self.mode:
        noise = torch.randn_like(pos) * self.denoise_scale
        data['noise'] = noise
        data['pos_denoise'] = pos + noise

        n_atom = len(mol.GetAtoms())
        bond_adj = torch.zeros(n_atom, n_atom, dtype=torch.long)
        # edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()
            bond_adj[(i, j)] = bond_adj[(j, i)] = 1
            if bond_type == 2:
                bond_adj[(i, j)] = bond_adj[(j, i)] = bond_adj[(i, j)] + 2
            if bond_type == 3:
                bond_adj[(i, j)] = bond_adj[(j, i)] = bond_adj[(i, j)] + 4
            if bond_type == 1.5:
                bond_adj[(i, j)] = bond_adj[(j, i)] = bond_adj[(i, j)] + 8

        data['bond_adj'] = bond_adj

        data['src_z'], data['mask_z'], data['coord'], data['mask_coord'], data['dist'], data['mask_dist'], data[
            'targets'], \
            data['src_edge_type'], data['bond_adj'],data['mask_z_nopad'],data['mask_idx']= mask_points(data,
                                                                                                self.vocab,
                                                                                                self.pad_idx,
                                                                                                self.mask_idx,
                                                                                                self.noise_type,
                                                                                                self.mask_noise,
                                                                                                self.mask_prob,
                                                                                                self.max_atoms)

        return data


    def load_finetune_dataset(self,input_path, target):
        input_df = pd.read_csv(input_path, sep=',')
        smiles_list = input_df['smiles']
        labels = input_df[target]
        labels = labels.fillna(-1).values.tolist()
        real_smiles_list = []
        real_labels = []
        for idx, smile in enumerate(smiles_list):
            if Chem.MolFromSmiles(smile) is not None:
                real_smiles_list.append(smile)
                real_labels.append(labels[idx])
        smiles_list = real_smiles_list
        mol_info = [self._process_smiles(s) for s in tqdm(smiles_list)]

        # convert 0 to -1
        # labels = labels.replace(0, -1)
        # there are no nans

        labels = real_labels
        assert len(smiles_list) == len(mol_info)
        assert len(smiles_list) == len(labels)
        return smiles_list, mol_info, torch.tensor(labels)

    def get_motif_smiles(mol, cliques):
        motif_smiles = []
        for clique in cliques:
            try:
                # 获取子分子
                submol = get_clique_mol(mol, clique)
                # 规范化SMILES
                smiles = Chem.MolToSmiles(submol, canonical=True)
                motif_smiles.append(smiles)
            except:
                continue
        return motif_smiles

    def len(self):
        return self.size

    def get(self, idx):
        if self.fast_read:
            mol_fine_data = self.data[idx]
        else:
            # rxn_data = torch.load(osp.join(self.processed_dir, f"rxn_data_{idx}.pt"))[:-1]
            mol_fine_data = torch.load(osp.join(self.processed_dir, f"mol_fine_data_{idx}" + FILE_NAME))
        return mol_fine_data