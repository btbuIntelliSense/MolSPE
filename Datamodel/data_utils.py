import os
import pandas as pd
import torch
from torch import nn
from torch_scatter import scatter

from MaskModel.MultiTaskTransformer3D import GaussianLayer


def get_downstream_task_info(config):
    """
    Get task names of downstream dataset
    """
    if config['task_name'] == 'bace':
        target = ["Class"]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'bbbp':
        target = ["p_np"]
        task = 'classification'
        loss_type = 'bce'
    # elif config['task_name'] == 'flavor':
    #     target = ["flavor_id"]
    #     task = 'classification'
    #     loss_type = 'bce'
    elif config['task_name'] == 'clintox':
        target = ['CT_TOX', 'FDA_APPROVED']
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'hiv':
        target = ["HIV_active"]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'muv':
        target = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'flavor':
        target = [
            'sweet', 'fruity', 'green', 'bitter', 'floral', 'woody', 'herbal',
            'waxy', 'fatty', 'fresh'
        ]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'sider':
        target = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues",
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders",
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders",
            "Reproductive system and breast disorders",
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "General disorders and administration site conditions", "Endocrine disorders",
            "Surgical and medical procedures", "Vascular disorders",
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders",
            "Congenital, familial and genetic disorders", "Infections and infestations",
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders",
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions",
            "Ear and labyrinth disorders", "Cardiac disorders",
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'tox21':
        target = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'toxcast':
        # raw_path = os.path.join(config['root'], 'toxcast', 'raw')
        raw_path = os.path.join(config['root'], 'raw')
        csv_file = os.listdir(raw_path)[0]
        input_df = pd.read_csv(os.path.join(raw_path, csv_file), sep=',')
        target = list(input_df.columns)[1:]
        task = 'classification'
        loss_type = 'bce'
        # 回归数据集
    elif config['task_name'] == 'esol':
        target = ["measured log solubility in mols per litre"]
        task = 'regression'
        loss_type = 'mse'
    elif config['task_name'] == 'freesolv':
        target = ["expt"]
        task = 'regression'
        loss_type = 'mse'
    elif config['task_name'] == 'lipophilicity':
        target = ['exp']
        task = 'regression'
        loss_type = 'mse'
    elif config['task_name'] == 'qm7':
        target = ['u0_atom']
        task = 'regression'
        loss_type = 'l1'
    elif config['task_name'] == 'qm8':
        target = ['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2',
                  'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0',
                  'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM']
        task = 'regression'
        loss_type = 'l1'
    elif config['task_name'] == 'qm9':
        target = ['homo', 'lumo', 'gap']
        task = 'regression'
        loss_type = 'l1'
    elif config['task_name'] == 'physprop_perturb':
        target = ['LogP']
        task = 'regression'
        loss_type = 'mse'
    else:
        return None
    config['target'] = target
    config['task'] = task
    config['loss_type'] = loss_type
    config['num_tasks'] = len(target)
    return config


class AtomBondEncoder(nn.Module):
    """3D GNN原子与化学键编码器"""

    def __init__(self, hidden_channels, num_layers, cutoff_upper, num_rbf):
        super().__init__()
        # self.distance = Distance(cutoff_upper)
        self.embedding = nn.Embedding(100, hidden_channels)

        self.conv_layers = nn.ModuleList([
            EdgeConv3D(hidden_channels, num_rbf)
            for _ in range(num_layers)
        ])

    def forward(self, z, pos, batch, edge_index, edge_attr):
        # 原子初始嵌入
        x = self.embedding(z)

        # 生成3D边特征
        edge_dist = pos[edge_index[0]] - pos[edge_index[1]]

        # 多层3D卷积
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_dist, edge_attr)

        # 化学键特征生成
        bond_feature = scatter(x[edge_index[0]] + x[edge_index[1]],
                               edge_index[0], dim=0, reduce='mean')

        return x, bond_feature


class EdgeConv3D(nn.Module):
    """3D图卷积层"""

    def __init__(self, hidden_dim, num_rbf):
        super().__init__()
        self.dist_exp = GaussianLayer(num_rbf)
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_dist, edge_attr):
        src, dst = edge_index
        dist_feat = self.dist_exp(edge_dist)

        # 组合原子特征与几何特征
        messages = torch.cat([x[src], x[dst], dist_feat], dim=-1)
        messages = self.message_net(messages)

        # 聚合信息
        aggregated = scatter(messages, dst, dim=0, reduce='mean')
        return x + aggregated


class MotifAggregator(nn.Module):
    """原子与Motif特征聚合器"""

    def __init__(self, atom_dim, motif_dim, output_dim):
        super().__init__()
        self.motif_attention = nn.MultiheadAttention(
            embed_dim=motif_dim, num_heads=4
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(atom_dim + motif_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, atom_feats, motif_feats, batch):
        # 原子级聚合
        mol_atom_feats = scatter(atom_feats, batch, dim=0, reduce='mean')

        # Motif级聚合
        if motif_feats is not None:
            motif_attn, _ = self.motif_attention(
                motif_feats.unsqueeze(1),
                motif_feats.unsqueeze(1),
                motif_feats.unsqueeze(1)
            )
            mol_motif_feats = scatter(motif_attn.squeeze(1),
                                      torch.zeros_like(motif_feats),
                                      dim=0, reduce='mean')
        else:
            mol_motif_feats = torch.zeros_like(mol_atom_feats)

        # 特征融合
        combined = torch.cat([mol_atom_feats, mol_motif_feats], dim=-1)
        return self.fusion_layer(combined)

    def detect_motifs(self, z, pos, batch):
        """基于3D几何的motif检测"""
        # 这里可以集成RDKit的BRICS分解或自定义规则
        # 返回检测到的motif ID列表
        # 示例实现：
        motifs = []
        for mol_idx in batch.unique():
            mol_pos = pos[batch == mol_idx]
            mol_z = z[batch == mol_idx]
            # 调用motif检测逻辑
            # detected = find_3d_motifs(mol_z, mol_pos)
            # motifs.extend(detected)
        return motifs