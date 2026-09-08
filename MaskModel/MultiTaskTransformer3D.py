import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Dict, Any, List

from torch_geometric.nn import GATConv

from MaskModel.Transformer import TransformerEncoderWithPair
from model.Embeddings import EdgeEmbedding
from model.Transformerbond import get_activation_fn


# from torchmdnet.datasets import Dictionary
# from torchmdnet.models.SpecFormer_layers import get_activation_fn


class MultiTaskTransformer3D(nn.Module):
    def __init__(self, args,dictionary):
        super(MultiTaskTransformer3D, self).__init__()
        # base_architecture(args)
        self.args = args
        self.padding_idx = 0
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )

        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )

        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
            )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                args.encoder_attention_heads, 1, args.activation_fn
            )
        if args.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                args.encoder_attention_heads, args.activation_fn
            )
            # 新增GAT层用于处理基序特征
        self.gat = GATConv(
            in_channels=args.encoder_embed_dim,
            out_channels=args.encoder_embed_dim,
            heads=4,
            concat=False
        )
        self.edge_embedding = nn.Embedding(10, args.encoder_embed_dim)

    def forward(self, src_tokens, src_distance, src_coord, src_edge_type,motif_atoms,encoder_masked_tokens=None):
        # 嵌入原子序数
        x = self.embed_tokens(src_tokens)  #初始原子特征（32，128,256）
        #src_tokens=(32,128)

        def get_dist_features(dist, et):  #注意力偏置
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result.permute(0,3,1,2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            # graph_attn_bias += bond_attn_bias  # 合并距离和键的注意力偏置
            return graph_attn_bias


        graph_attn_bias = get_dist_features(src_distance, src_edge_type)

        # 编码器
        encoder_rep,encoder_pair_rep,delta_encoder_pair_rep,x_norm,delta_encoder_pair_rep_norm=self.encoder(x,padding_mask=None, attn_mask=graph_attn_bias)
        #encoder_rep （32,128,256）   encoder_pair_rep（32,128,128,256）

        # 原子序数预测
        logits = self.lm_head(encoder_rep, encoder_masked_tokens)#（32,120）

        # 坐标预测
        coords_emb = src_coord #(32,128,3)
        delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)  #(32,128,128,3)
        attn_probs = self.pair2coord_proj(encoder_pair_rep)
        coord_update = delta_pos * attn_probs
        coord_update = torch.sum(coord_update, dim=2)
        encoder_coord = coords_emb + coord_update

        # 距离预测
        encoder_distance = self.dist_head(encoder_pair_rep)

        ##########################################################################
        # 新增部分：基序特征提取和聚合
        ##########################################################################
        # 获取输入形状
        batch_size, num_motifs, max_instances, max_atoms = motif_atoms.shape
        _, seq_len, embed_dim = encoder_rep.shape

        # 初始化基序表示张量
        motif_reprs = torch.zeros(
            batch_size, num_motifs, embed_dim,
            device=encoder_rep.device
        )
        # 遍历每个分子
        for i in range(batch_size):
            # 遍历每种基序类型
            for j in range(num_motifs):
                instance_reprs = []  # 存储当前基序类型的所有实例表示

                # 遍历该基序类型的每个实例
                for k in range(max_instances):
                    # 获取当前实例的原子索引
                    atom_indices = motif_atoms[i, j, k]

                    # 过滤无效索引（填充值为-1）
                    valid_indices = [idx.item() for idx in atom_indices if idx != -1]

                    # 如果没有有效原子，跳过
                    if not valid_indices:
                        continue

                    # 提取原子表示
                    atom_reprs = encoder_rep[i, valid_indices]  # [num_atoms, embed_dim]

                    # 构建子图的邻接矩阵
                    num_atoms = len(valid_indices)
                    edge_index = []
                    edge_attr = []

                    # 遍历子图中的所有原子对
                    for m_idx, m in enumerate(valid_indices):
                        for n_idx, n in enumerate(valid_indices):
                            # 仅处理上三角部分（避免重复）
                            if m_idx < n_idx:
                                # 检查原始图中是否存在边
                                bond_type = src_edge_type[i, m, n]

                                # 如果存在有效边
                                if bond_type > 0:
                                    # 添加两条边（无向图）
                                    edge_index.append([m_idx, n_idx])
                                    edge_index.append([n_idx, m_idx])

                                    # 边类型属性（相同类型添加两次）
                                    edge_attr.append(bond_type)
                                    edge_attr.append(bond_type)

                    # 如果没有边，添加自环
                    if not edge_index:
                        for idx in range(num_atoms):
                            edge_index.append([idx, idx])
                            edge_attr.append(4)  # 自环的特殊类型

                    # 转换为张量
                    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(encoder_rep.device)
                    edge_attr = torch.tensor(edge_attr, dtype=torch.long).to(encoder_rep.device)

                    # 嵌入边属性
                    edge_emb = self.edge_embedding(edge_attr)

                    # 应用GAT更新原子表示
                    updated_atom_reprs = self.gat(atom_reprs, edge_index, edge_emb)

                    # 平均池化得到实例表示
                    instance_repr = torch.mean(updated_atom_reprs, dim=0)
                    instance_reprs.append(instance_repr)

                # 如果有有效实例，计算该基序类型的平均表示
                if instance_reprs:
                    motif_repr = torch.mean(torch.stack(instance_reprs), dim=0)
                    motif_reprs[i, j] = motif_repr

        new_encoder_rep=torch.cat([encoder_rep,motif_reprs],dim=1)
        return logits, encoder_distance, encoder_coord, x_norm,delta_encoder_pair_rep_norm,encoder_rep,new_encoder_rep



# 辅助类
class MaskLMHead(nn.Module):
    def __init__(self, embed_dim, output_dim, activation_fn):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)
        self.weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None):
        if masked_tokens is not None:
            features = features[masked_tokens, :]
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x


class DistanceHead(nn.Module):
    def __init__(self, heads, activation_fn):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).squeeze(-1)
        return x


class NonLinearHead(nn.Module):
    def __init__(self, input_dim, out_dim, activation_fn):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, out_dim)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        # 重塑为原子级特征
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1,-1,self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # 计算均值和方差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # 归一化
        x = (x - mean) / (std + self.eps)
        # 应用可学习的权重和偏置
        x = self.weight * x + self.bias
        return x

def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)

